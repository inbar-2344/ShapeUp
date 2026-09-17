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

# Shared sampling code lives in sampling_utils.py so the three preprocessing
# entry points cannot drift apart.
from sampling_utils import (
    _rand_barycentric_coords,
    calc_edge_length,
    calc_edges,
    calc_face_normals,
    calculate_dihedral_angles,
    calculate_edge_normals,
    orient_edges,
    process_mesh,
    rotate_vectors_by_angles_around_axis,
    sample_points_from_edges,
    sample_points_from_meshes,
)





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



    
def sample_points_in_cube(num_samples, edge_length, device = 'cuda'):
    """
    returns $num_samples points on the cube with edge_length=2*$half_length centered at the origin
    """
    samples = (torch.rand([num_samples,3],device=device)-0.5) * edge_length
    return samples



    


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


        
# --------------------------------------------------------------------------- #
#  Reusable single-mesh steps.
#
#  The __main__ loop below and geometry_inference.py both go through these, so
#  there is exactly one implementation of "render a source view set" and
#  "sample the point cloud" in the project.
# --------------------------------------------------------------------------- #
def render_source_views(glb_path, out_path, bpy_manager=None, save_dir=None,
                        ignore_components=("Icosphere",), dirname="source"):
    """Render the multi-view / depth / normal set for one mesh into out_path.

    Returns the bpy_ops manager so a caller looping over many meshes can reuse
    it (constructing one per mesh is wasteful and leaks Blender state).
    """
    if bpy_manager is None:
        bpy_manager = bpy_ops(save_dir if save_dir is not None else out_path)
    bpy_manager.load_scene(glb_path, ignore_components=list(ignore_components))
    bpy_manager.out_path = out_path
    os.makedirs(bpy_manager.out_path, exist_ok=True)
    bpy_manager.set_camera()
    bpy_manager.render(dirname)
    return bpy_manager


def sample_and_save_pc(glb_path, npz_output_path, sharpness_threshold,
                       point_number=65536, resolution=512, scale=None,
                       center=None):
    """to_watertight + process_mesh -> pc.npz. Returns (scale, center).

    The scale/center are returned so a multistep editing sequence can pin the
    same normalisation across its frames.
    """
    scene = trimesh.load(glb_path, force="mesh")
    watertight_mesh, scale, center = to_watertight(scene, resolution=resolution,
                                                   scale=scale, center=center)
    v = torch.from_numpy(watertight_mesh.vertices).float()
    f = torch.from_numpy(watertight_mesh.faces).long()
    sharp_surface, sharp_near_surface, coarse_surface, rand_points = process_mesh(
        v, f, sharpness_threshold, point_number // 2, point_number // 2
    )
    os.makedirs(os.path.dirname(npz_output_path), exist_ok=True)
    np.savez(
        npz_output_path,
        fps_sharp_surface=sharp_surface.astype(np.float32),
        sharp_near_surface=sharp_near_surface.astype(np.float32),
        fps_coarse_surface=coarse_surface.astype(np.float32),
        rand_points=rand_points.astype(np.float32),
    )
    del v, f, sharp_surface, sharp_near_surface, coarse_surface, rand_points
    del watertight_mesh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return scale, center


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
            render_source_views(glb_path, os.path.join(save_dir, uid),
                                bpy_manager=bpy_manager)
        except:
            reason = "failed loading scene"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
            continue
        if not render_only:
            # Need to figure our how to use the same scale and center for multistep editing sequence. 
            scale=None
            center=None
            dirname = "source"
            try:
                scale, center = sample_and_save_pc(
                    glb_path, pjoin(bpy_manager.out_path, dirname, "pc.npz"),
                    sharpness_threshold, point_number=args.point_number,
                    resolution=512, scale=scale, center=center)

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