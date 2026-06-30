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
import ctypes
import shutil
import math
import argparse
from typing import Tuple, Union
from os.path import join as pjoin
from datetime import datetime

import torch
import torch.nn.functional as tfunc
import numpy as np
import pandas as pd
import trimesh
import fpsample
import objaverse
from pysdf import SDF

from bpyrenderer import bpy_ops
from sharp_edge_sampling.to_watertight_mesh import to_watertight

import os


def trim_memory():
    """Reclaim freed memory back to the OS so long batch runs don't accumulate RSS.

    Runs the garbage collector, empties the CUDA cache, and (on Linux) asks glibc
    to return free heap pages with ``malloc_trim``. Without this, processing many
    meshes in one process slowly leaks resident memory and eventually OOMs.
    """
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


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
    sorted_edges, _ = full_edges.sort(dim=-1)  # F*3,2

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
    Uniformly sample points on the surface of a single mesh, with probability
    proportional to face area.

    Args:
        verts, faces: the mesh vertices (V,3) and faces (F,3).
        num_samples: number of points to sample.
        return_normals: also return per-point normals.
        return_barycentric: also return the sampled barycentric coordinates.
    """
    device = verts.device

    with torch.no_grad():
        face_normals = torch.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                    verts[faces[:, 2]] - verts[faces[:, 1]])
        areas = torch.linalg.norm(face_normals, dim=-1) / 2
        assert not torch.all(torch.isclose(areas, torch.zeros_like(areas)))
        sample_face_idxs = areas.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)

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
        # Per-point normals are the face normals of the sampled faces.
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]

    if return_normals:
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
    return w0, w1, w2

def sample_points_from_edges(verts, edges, num_samples, edge_normals = None):

    device = verts.device
    samples = torch.zeros((num_samples, 3), device=device)

    with torch.no_grad():
        lengths = calc_edge_length(verts, edges)
        assert not torch.all(torch.isclose(lengths, torch.zeros_like(lengths)))
        sample_edge_idxs = lengths.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)

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

    if edge_normals is not None:
        return samples, normals, sample_edge_idxs
    return samples, sample_edge_idxs

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
    r"""
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
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1)
    return face_normals  # F,3

def calculate_dihedral_angles(v, f, edges, ef, face_normals, return_normals_and_orientation=False):
    oriented_edges = orient_edges(edges, f[ef[:, 0, 0]])
    fn0 = face_normals[ef[:, 0, 0]]
    fn1 = face_normals[ef[:, 1, 0]]
    edge_vec = tfunc.normalize(v[oriented_edges[:, 1]] - v[oriented_edges[:, 0]])
    dot = torch.sum(fn0 * fn1, dim=-1, keepdim=True)
    det = torch.sum(edge_vec * torch.cross(fn0, fn1, dim=-1), dim=-1, keepdim=True)
    angles = torch.atan2(det, dot)  # from -180 to 180
    if return_normals_and_orientation:
        return angles, fn0, fn1, edge_vec
    else:
        return torch.abs(angles)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type= str,
        default="csvs/unified_dataset.csv",
        help="watertight obj files out directory",
    )
    
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="render mv for texture",
    )
    parser.add_argument(
        "--data_dir",
        type= str,
        default="bpy-renderer/outputs",
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
        "--start_ind",
        type=int,
        default=0,
        help="for running in parallel",
    )
    parser.add_argument(
        "--end_ind",
        type=int,
        default=4000,
        help="for running in parallel",
    )

    args, extras = parser.parse_known_args()
    print(f"args.start_ind: {args.start_ind}, args.end_ind: {args.end_ind}")
    render_only = args.render_only
    sharpness_threshold = math.radians(args.angle_threshold)
    skip_uids = list()
    uids = pd.read_csv(args.csv_path)
    uids = uids[uids.motion == 1.0]
    uids = uids[uids.texture_compatible == 1.0]
    uids = uids['uid'].to_list()[args.start_ind:args.end_ind]
    save_dir = args.data_dir
    bpy_manager = bpy_ops(save_dir)
    for uid in uids:
        print(f"processing uid: {uid}")
        out_path=pjoin(args.data_dir, uid)
        if os.path.exists(out_path):
            processed_motion_dirs = [pjoin(out_path, d) for d in os.listdir(out_path) if os.path.isdir(pjoin(out_path, d)) and d.startswith('frame_')] 
            for dir in processed_motion_dirs:
                if not render_only:
                    shutil.rmtree(dir)
                else:
                    shutil.rmtree(pjoin(dir, 'mv'), ignore_errors=True)
                    shutil.rmtree(pjoin(dir, 'depth'), ignore_errors=True)
                    shutil.rmtree(pjoin(dir, 'normal'), ignore_errors=True)
                    try:
                        os.remove(pjoin(dir, 'meta.json'), ignore_errors=True)
                    except:
                        pass
        glb_path_dict = objaverse.load_objects(
        uids=[uid]
        )
        print("Downloaded object paths:", glb_path_dict[uid])
        try:
            bpy_manager.load_scene_armature(glb_path_dict[uid], ignore_components=["Icosphere", "polySurface"]) 
        except:
            reason = "failed loading scene armature"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
            continue
        num_frames = bpy_manager.get_frames_count()
        if num_frames == 1:
            reason = "single frame"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
            continue
        os.makedirs(bpy_manager.out_path, exist_ok=True) 
        print(f"getting most different keyframes: {num_frames}")
        chosen_frames_indices = bpy_manager.get_most_different_keyframes(num_frames=3)
        print(f"chosen frames indices: {chosen_frames_indices}")
        chosen_frames_indices.sort(key=lambda x: bpy_manager.get_frame_bbox(x)[2], reverse=True)
        scale=None
        for frame_ind in chosen_frames_indices: 
            dirname = str(f'frame_{frame_ind}')
            bpy_manager.copy_frame_to_end(frame_ind)
            bpy_manager.set_frame(bpy_manager.get_frames_count())
            bpy_manager.gc()
            # calc camera center and radius
            bbox_min, bbox_max = bpy_manager.scene_manager.get_scene_bbox()
            center = (bbox_min + bbox_max) / 2

            bpy_manager.set_camera(center=center)  # Set up 6 cameras ONCE at render frames 0-5
            print(f">>> Rendering {dirname}", flush=True)
            bpy_manager.render(dirname)
            print(f">>> Finished rendering {dirname}", flush=True)
            bpy_manager.gc()
            if not render_only:
                save_path = bpy_manager.export_current_pose(fname="tmp_glb.glb")
                scene = trimesh.load(save_path, force='mesh')
                os.remove(save_path)
                try:
                    watertight_mesh, scale, _ = to_watertight(scene, scale=scale, resolution=512)  # convert to watertight + sample
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
                    
                except Exception as e:
                    reason = f"failed converting to watertight mesh or sampling. problematic part: {dirname} exception: {e}"
                    print(f"skipping uid: {uid}, reason: {reason}")
                    skip_uids.append({"uid": uid, "reason": reason})
                    # this frame failed; keep already-sampled frames and free memory before the next one
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    trim_memory()

        try:
            os.remove(glb_path_dict[uid])
            os.rmdir(path=os.path.dirname(glb_path_dict[uid]))
        except OSError:
            # downloaded file already removed, or directory not empty: ignore
            pass
        trim_memory()

    print(f'Processed uids {args.start_ind}-{args.end_ind}. Skipped {len(skip_uids)} uids.')
    skip_ids_df = pd.DataFrame(skip_uids)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = pjoin(args.data_dir, f"skip_uids_motion_{args.start_ind}_{args.end_ind}_{timestamp}.csv")
    skip_ids_df.to_csv(timestamped_filename, index=False)

    
