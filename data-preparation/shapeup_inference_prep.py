# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
This module implements utility functions for sampling points from
batches of meshes.
"""
import sys
import gc
from typing import Tuple, Union
import torch
import torch.nn.functional as tfunc
import math
import numpy as np 
import argparse
from pysdf import SDF
import trimesh
import os 
from os.path import join as pjoin 
import fpsample
import pandas as pd
from bpyrenderer import bpy_ops
from sharp_edge_sampling.to_watertight_mesh import to_watertight
from datetime import datetime
import shutil
from pathlib import Path
# import pytorch3d
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import check_sign, point_mesh_face_distance

# os.environ["POLYSCOPE_ALLOW_HEADLESS"] = "1"
# import polyscope as ps 

def calc_edge_length(
        vertices: torch.Tensor,  # V,3 first may be dummy
        edges: torch.Tensor,  # E,2 long, lower vertex index first, (0,0) for unused
) -> torch.Tensor:  # E

    full_vertices = vertices[edges]  # E,2,3
    a, b = full_vertices.unbind(dim=1)  # E,3
    return torch.norm(a - b, p=2, dim=-1)

def calc_edges(
        faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
        with_edge_to_face: bool = False,
        with_dummies=True
):
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |
    e0---->-o
    """

    F = faces.shape[0]

    # make full edges, lower vertex index first
    face_edges = torch.stack((faces, faces.roll(-1, 1)), dim=-1)  # F*3,3,2
    full_edges = face_edges.reshape(F * 3, 2)
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2 TODO min/max faster?

    # make unique edges
    edges, full_to_unique = torch.unique(input=sorted_edges, sorted=True, return_inverse=True, dim=0)  # (E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F, 3)  # F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:, 0] != sorted_edges[:, 0]  # F*3
    edge_to_face = torch.zeros((E, 2, 2), dtype=torch.long, device=faces.device)  # E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(0, F, device=faces.device),
                                       torch.arange(0, 3, device=faces.device))  # F*3,2
    edge_to_face.reshape(2 * E, 2).scatter_(dim=0, index=(2 * full_to_unique + is_right)[:, None].expand(F * 3, 2),
                                            src=scatter_src)  # E,LR=2,S=2
    if with_dummies:
        edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face  # =EF

def sample_points_from_meshes(
    verts, faces,
    num_samples: int = 10000,
    return_normals: bool = True,
    return_barycentric: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
            coordinates of sampled points for each mesh in the batch. For empty
            meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
            to each sampled point. Only returned if return_normals is True.
            For empty meshes the corresponding row in the normals array will
            be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
            texture vector to each sampled point. Only returned if return_textures is True.
            For empty meshes the corresponding row in the textures array will
            be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    # if meshes.isempty():
    #     raise ValueError("Meshes are empty.")

    # dont support batch for now.
    # if not torch.isfinite(verts).all():
    #     raise ValueError("Meshes contain nan or inf.")

    # if return_textures and meshes.textures is None:
    #     raise ValueError("Meshes do not contain textures.")

    device = verts.device

    # faces = soa[6][0]#meshes.faces_packed()
    #mesh_to_face = meshes.mesh_to_faces_packed_first_idx()

    # Initialize samples tensor with fill value 0 for empty meshes.

    # Only compute samples for non empty meshes
    with torch.no_grad():
        # areas, _ = mesh_face_areas_normals(verts, faces[:-1])  # Face areas can be zero.
        face_normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                    verts[faces[:, 2]] - verts[faces[:, 1]])
        areas = torch.linalg.norm(face_normals, dim=-1) / 2
        # assert(torch.all(torch.isclose(areas, areas2)))
        assert not torch.all(torch.isclose(areas, torch.zeros_like(areas)))
        # max_faces = meshes.num_faces_per_mesh().max().item()
        # areas_padded = packed_to_padded(
        #     areas, mesh_to_face[meshes.valid], max_faces
        #)  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        #sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces[:]]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(1, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples = w0[0, :, None] * a + w1[0, :, None] * b + w2[0, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, vert_normals, sample_face_idxs
    if return_barycentric:
        return samples, sample_face_idxs, torch.concat([w0,w1,w2], dim=0).T
    return samples, sample_face_idxs

def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
    #  got `Tuple[float, typing.Any, typing.Any]`.
    return w0, w1, w2

def sample_points_from_edges(verts, edges, num_samples, edge_normals = None):

    device = verts.device

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_samples, 3), device=device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        lengths = calc_edge_length(verts, edges)
        assert not torch.all(torch.isclose(lengths, torch.zeros_like(lengths)))

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_edge_idxs = lengths.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        # sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    edge_verts = verts[edges[:]]
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Randomly generate barycentric coords.
    t = torch.rand(num_samples, device=verts.device)

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_edge_idxs]  # (N, num_samples, 3)
    b = v1[sample_edge_idxs]
    samples = t[:, None] * a + (1-t[:, None]) * b

    if edge_normals is not None:
        normals = torch.zeros((num_samples, 3), device=device)
        normal_idx = torch.round(torch.rand(num_samples, device=verts.device)).long()
        normals = edge_normals[sample_edge_idxs][normal_idx]

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if edge_normals is not None:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals, sample_edge_idxs
    return samples, sample_edge_idxs

def calculate_centeroid(vs):
    return torch.mean(vs,dim=1)

def bounding_sphere_radius(vs):
    centered_vs = vs - calculate_centeroid(vs)
    distances = torch.norm(centered_vs,dim=-1)
    return torch.max(distances)

def sample_pertrubed_points_from_mesh(vs,f,num_samples):
    samples, _ = sample_points_from_meshes(vs,f,num_samples=num_samples)
    r = bounding_sphere_radius(vs)
    pertrubation_vectors = torch.randn_like(samples, device=samples.device)*(r/1024)
    return samples + pertrubation_vectors

def orient_edges(edges, f_for_orientation):
    ev0 = edges[:, 0]
    ev1 = edges[:, 1]
    fv0 = f_for_orientation[:, 0]
    fv1 = f_for_orientation[:, 1]
    fv2 = f_for_orientation[:, 2]
    flip_flag = ((ev1 == fv0) & (ev0 == fv1)) | ((ev1 == fv1) & (ev0 == fv2)) | ((ev1 == fv2) & (ev0 == fv0))
    oriented_edges = torch.tensor(edges, device=edges.device, dtype=torch.long)
    oriented_edges[flip_flag, 0] = edges[flip_flag, 1]
    oriented_edges[flip_flag, 1] = edges[flip_flag, 0]
    return oriented_edges

def calc_face_normals(
        vertices: torch.Tensor,  # V,3 first vertex may be unreferenced
        faces: torch.Tensor,  # F,3 long, first face may be all zero   
        normalize: bool = False,
) -> torch.Tensor:  # F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces]  # F,C=3,3
    v0, v1, v2 = full_vertices.unbind(dim=1)  # F,3
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # F,3
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1)  # TODO inplace?
    return face_normals  # F,3

def calculate_dihedral_angles(v, f, edges, ef, face_normals, return_normals_and_orientation=False):
    oriented_edges = orient_edges(edges, f[ef[:, 0, 0]])
    fn0 = face_normals[ef[:, 0, 0]]
    fn1 = face_normals[ef[:, 1, 0]]
    edge_vec = tfunc.normalize(v[oriented_edges[:, 1]] - v[oriented_edges[:, 0]])
    # todo: calculate angle to rotate first normal to second normal through oriented edge
    dot = torch.sum(fn0 * fn1, dim=-1, keepdim=True)
    det = torch.sum(edge_vec * torch.cross(fn0, fn1, dim=-1), dim=-1, keepdim=True)
    angles = torch.atan2(det, dot)  # from -180 to 180
    # angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    if return_normals_and_orientation:
        return angles, fn0, fn1, edge_vec
    else:
        return torch.abs(angles)
    
def sample_points_in_cube(num_samples, edge_length, device = 'cuda'):
    """
    returns $num_samples points on the cube with edge_length=2*$half_length centered at the origin
    """
    samples = (torch.rand([num_samples,3],device=device)-0.5) * edge_length
    return samples

def rotate_vectors_by_angles_around_axis(vectors, angles, axes):
    """
    batched operation, uses the Rodriguez formula
    :param vectors:
    :param angles: in radians
    :param axes:
    :return:
    """
    K2v = torch.cross(axes, torch.cross(axes, vectors, dim=-1), dim=-1)
    Kv = torch.cross(axes, vectors, dim=-1)
    return vectors + torch.sin(angles) * Kv + (1 - torch.cos(angles)) * K2v

def calculate_edge_normals(angles, fn0, edge_vec):
    edge_normals = rotate_vectors_by_angles_around_axis(fn0, angles / 2, edge_vec)
    return edge_normals

def process_mesh(v, f, sharpness_threshold, n_coarse, n_sharp):
    edges, fe, ef = calc_edges(f, with_edge_to_face=True, with_dummies=False)
    face_normals = calc_face_normals(v, f, normalize=True)
    angles, fn0, fn1, edge_vec = calculate_dihedral_angles(v, f, edges, ef, face_normals, return_normals_and_orientation=True)
    edge_normals = calculate_edge_normals(angles, fn0, edge_vec)
    sharp_edges = edges[torch.abs(angles).squeeze() > sharpness_threshold]
    if len(sharp_edges) == 0: # no sharp edges depected. sample uniformly
        print("No sharp edges detected. Sample edges uniformely")
        sharp_edges = edges
    sharp_samples, sharp_normals, sharp_sample_edge_idxs = sample_points_from_edges(v, sharp_edges, n_sharp, edge_normals = edge_normals)
    sharp_surface = torch.cat((sharp_samples, sharp_normals), dim=-1)
    
    sharp_near_surface_samples= [
                    sharp_samples + torch.randn(len(sharp_samples), 3) * 0.001,
                    sharp_samples + torch.randn(len(sharp_samples), 3) * 0.005,
                    sharp_samples + torch.randn(len(sharp_samples), 3) * 0.007,
                    sharp_samples + torch.randn(len(sharp_samples), 3) * 0.01
        ]
    sharp_near_surface_samples = torch.cat(sharp_near_surface_samples)
    #torch imp test 
    # mesh_for_sdf = Meshes(verts=[v], faces=[f])
    # squared_dists = point_mesh_face_distance(mesh_for_sdf, sharp_near_surface_samples)
    # dists = squared_dists.sqrt().transpose(1, 0)  # shape: (N, 1)
    # inside_mask = check_sign(sharp_near_surface_samples, mesh_for_sdf) 
    # signed_dists = dists.squeeze(1)
    # signed_dists[inside_mask.squeeze(0)] *= -1
    # signed_dists = signed_dists.unsqueeze(1)
    
    
    mesh_sdf = SDF(v, f)
    sharp_sdf = mesh_sdf(sharp_near_surface_samples).reshape(-1,1) # numpy array
    sharp_near_surface = np.concatenate([sharp_near_surface_samples, sharp_sdf], axis=1) # numpy array
    
    
    coarse_samples_for_random, coarse_sample_face_idxs_for_random = sample_points_from_meshes(v, f, 200000,return_normals=False) 
    
    coarse_near_surface_points= [
            coarse_samples_for_random + torch.randn(len(coarse_samples_for_random), 3) * 0.001,
            coarse_samples_for_random + torch.randn(len(coarse_samples_for_random), 3) * 0.005
        ]
    
    coarse_near_surface_points = torch.cat(coarse_near_surface_points)
    space_points = torch.rand(200000, 3) * 2.1 - 1.05
    rand_points = torch.cat([coarse_near_surface_points, space_points], dim=0)
    coarse_sdf = mesh_sdf(rand_points).reshape(-1,1) # numpy array

    rand_points = np.concatenate([rand_points, coarse_sdf], axis=1) 
    
    coarse_samples, coarse_normals, coarse_sample_face_idxs = sample_points_from_meshes(v, f, 200000,return_normals=True) 
    coarse_surface = torch.cat([coarse_samples, coarse_normals], dim=-1)
    fps_coarse_surface_list=[]
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_samples, n_coarse, h=5)
    fps_coarse_surface = coarse_surface[kdline_fps_samples_idx].reshape(-1,1,6)
    fps_coarse_surface_list.append(fps_coarse_surface) 
    fps_coarse_surface = np.concatenate(fps_coarse_surface_list, axis=1)
    
    return sharp_surface.numpy()[:, None, :], sharp_near_surface, fps_coarse_surface, rand_points
    


def get_parent_chain(scene, node):
    """Return list of parents from node up to root (node included)."""
    chain = [node]
    while True:
        parent = scene.graph.transforms.parents.get(node)
        if not parent:
            break
        node = parent  # trimesh usually has one parent, but handle multiple if needed
        chain.append(node)
    return chain


def first_geometry_nodes_common_parent(scene):
    geometry_nodes = list(scene.geometry.keys())
    # Get parent chains for all nodes
    chains = [get_parent_chain(scene, n) for n in geometry_nodes]

    # Intersect chains: walk from root downward
    common = set(chains[0])
    for ch in chains[1:]:
        common &= set(ch)

    if not common:
        return None, None

    # Find the lowest (closest to the nodes) common ancestor
    common_node = None
    for node in chains[0]:
        if node in common:
            common_node = node
            break

    part2geometry = dict()
    for chain_ind, chain in enumerate(chains):
        common_parent_ind = chain.index(common_node)
        if common_parent_ind == 0:  # next node in chain in the geometry node
            part2geometry[geometry_nodes[chain_ind]] = [geometry_nodes[chain_ind]]
        else:
            part = chain[common_parent_ind - 1]
            if part in part2geometry.keys():
                part2geometry[part].append(geometry_nodes[chain_ind])
            else:
                part2geometry[part] = [geometry_nodes[chain_ind]]

    return common_node, part2geometry


def delete_hierarchy(scene: trimesh.Scene, root: str) -> trimesh.Scene:
    """
    Remove root and all its descendants (by node name) from a trimesh.Scene,
    returning a new scene. Geometry smoothness is preserved by leaving meshes
    and their normals untouched; we only drop instances and then prune orphans.
    """
    s = scene.copy()

    # 1) Collect subtree nodes (root + all descendants)
    forest = s.graph.transforms
    to_remove = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n in s.graph.nodes:
            to_remove.append(n)
            stack.extend(forest.children.get(n, []))

    if not to_remove:
        return s  # nothing to remove

    # 2) Remove nodes (instances); this does NOT delete mesh data
    for n in to_remove:
        if n in s.graph.nodes:
            forest.remove_node(n)

    # 3) Prune geometry that is no longer referenced by any node
    unused = [g for g, nodes in s.graph.geometry_nodes.items() if not nodes]
    if unused:
        s.delete_geometry(unused)

    for geom in s.geometry.values():
        geom.merge_vertices()  # weld duplicate vertices
        geom.vertex_normals
    return s


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="render mv for texture",
    )
    parser.add_argument(
        "--out_dir",
        type= str,
        default="sampled_meshes",
        help="watertight obj files out directory",
    )
    parser.add_argument(
        "--angle_threshold",
        type=int,
        default=15,
        help="angle threhold for sharp edge",
    )

    parser.add_argument(
        "--point_number",
        type=int,
        default=65536,
        help="number of points",
    )

    parser.add_argument(
        "--local_glb_dir",
        type=str,
        default="glb_dir",
        help="If set, ingest local .glb files from this directory instead of downloading from Objaverse.",
    )

    args, extras = parser.parse_known_args()
    sharpness_threshold = math.radians(args.angle_threshold)
    skip_uids = list()
    render_only = args.render_only

    local_mode = bool(args.local_glb_dir)
    glb_path_map = {}
    local_root = Path(args.local_glb_dir).expanduser().resolve()
    glb_paths = sorted(local_root.glob("**/*.glb"))
    glb_paths = [p for p in glb_paths if p.is_file()]
    uids = []
    for p in glb_paths:
        rel = p.relative_to(local_root)
        # If your data is laid out like <mesh_name>/*.glb, use <mesh_name> as uid.
        # Otherwise (files directly under local_root), fall back to filename stem.
        base_uid = rel.parent.as_posix() if rel.parent.as_posix() != "." else p.stem
        base_uid = base_uid.replace("/", "__")
        uid = base_uid
        if uid in glb_path_map:
            uid = f"{base_uid}__{p.stem}"
            if uid in glb_path_map:
                uid = f"{uid}__{abs(hash(str(p)))}"
        glb_path_map[uid] = str(p)
        uids.append(uid)
    save_dir = args.out_dir
    bpy_manager = bpy_ops(save_dir)
    for uid in uids:
        print(f"processing uid: {uid}")
        # check if alredy processed
        out_path=pjoin(args.out_dir, uid)
        if os.path.exists(out_path):
            if not render_only:
                shutil.rmtree(out_path, ignore_errors=True)
            else:
                shutil.rmtree(pjoin(out_path,"source", 'mv'), ignore_errors=True)
                shutil.rmtree(pjoin(out_path, "source", 'depth'), ignore_errors=True)
                shutil.rmtree(pjoin(out_path,"source" , 'normal'), ignore_errors=True)
                try:
                    os.remove(pjoin(out_path,"source", 'meta.json'), ignore_errors=True)
                except:
                    pass
            
        glb_path = glb_path_map[uid]
        print("Using local object path:", glb_path)
        
        try:
            bpy_manager.load_scene(glb_path, ignore_components=["Icosphere"]) 
        except:
            reason = "failed loading scene"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
            continue

        bpy_manager.out_path = os.path.join(save_dir, uid)

        os.makedirs(bpy_manager.out_path, exist_ok=True)
        bpy_manager.set_camera()
        bpy_manager.render("source")
        if not render_only:
            # Need to figure our how to use the same scale and center for multistep editing sequence. 
            scale=None
            center=None
            dirname = "source"
            scene = trimesh.load(glb_path, force='mesh')
            try:
                watertight_mesh, scale, center = to_watertight(scene, resolution=512, scale=scale, center=center)# convert to waterproof + sample 
                # mesh = list(watertight_mesh.geometry.values())[0]
                v = torch.from_numpy(watertight_mesh.vertices).float()
                f = torch.from_numpy(watertight_mesh.faces).long()
                sharp_surface, sharp_near_surface, coarse_surface, rand_points = process_mesh(v, f, sharpness_threshold, args.point_number//2, args.point_number//2)
                
                npz_output_path = pjoin(bpy_manager.out_path, dirname, "pc.npz")
                np.savez(
                    npz_output_path,
                    fps_sharp_surface = sharp_surface.astype(np.float32),
                    sharp_near_surface = sharp_near_surface.astype(np.float32),
                    fps_coarse_surface = coarse_surface.astype(np.float32),
                    rand_points = rand_points.astype(np.float32),
                ) 
                del v, f, sharp_surface, sharp_near_surface, coarse_surface, rand_points, watertight_mesh
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                reason = f"failed converting to watertights mesh or sampling. problematic part: {dirname} exception: {e}"
                print(f"skipping uid: {uid}, reason: {reason} exception: {e}")
                skip_uids.append({"uid": uid, "reason": reason})
                    # ADD THESE LINES:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                break 
    
            
    print(f'Skipped {len(skip_uids)} uids.')
    skip_ids_df = pd.DataFrame(skip_uids)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = pjoin(args.out_dir, f"skip_uids_{timestamp}.csv")
    skip_ids_df.to_csv(timestamped_filename, index=False)