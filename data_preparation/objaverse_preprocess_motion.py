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
    trim_memory,
)














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

    
