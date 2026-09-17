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
    sharpness_threshold = math.radians(args.angle_threshold)
    skip_uids = list()
    render_only = args.render_only
    uids = pd.read_csv(args.csv_path)
    uids = uids[uids.texture_compatible == 1.0]
    uids = uids['uid'].to_list()[args.start_ind:args.end_ind]
    save_dir = args.data_dir
    bpy_manager = bpy_ops(save_dir)
    for uid in uids:
        print(f"processing uid: {uid}")
        # check if alredy processed
        out_path=pjoin(args.data_dir, uid)
        if os.path.exists(out_path):
            processed_parts_dirs = [pjoin(out_path, d) for d in os.listdir(out_path) if not d.startswith('frame_')] 
            for dir in processed_parts_dirs:
                if not render_only:
                    shutil.rmtree(dir, ignore_errors=True)
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
            bpy_manager.load_scene(glb_path_dict[uid], ignore_components=["Icosphere"]) 
        except:
            reason = "failed loading scene"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
            continue
        if bpy_manager.count_geometry() == 1: #single mesh scene, not relevant 
            reason = "single mesh"
            print(f"skipping uid: {uid}, reason: {reason}")
            skip_uids.append({"uid": uid, "reason": reason}) 
        else:
            parts, parts_bbox_volumes = bpy_manager.get_removal_candidates() # return also volumes to maybe set a threshold on part volume
            if not parts: 
                reason = "no common parent for geometry nodes"
                print(f"skipping uid: {uid}, reason: {reason}")
                skip_uids.append({"uid": uid, "reason": reason}) 
                continue
            os.makedirs(bpy_manager.out_path, exist_ok=True)
            sorted_parts = sorted(zip(parts, parts_bbox_volumes), key=lambda x: x[1], reverse=True)[:3] # process only the first 3 parts with max volume
            print(">>> Setting camera", flush=True)
            bpy_manager.set_camera()
            print(">>> Starting source render", flush=True)
            bpy_manager.render("source")
            print(">>> Finished source render", flush=True)
            n_parts = len(sorted_parts)
            scale=None
            center=None
            for ind in range(-1, n_parts): # max(n_parts) = 3
                if ind > -1:
                    dirname = str(ind)
                    bpy_manager.remove_part_from_scene(parts[ind])
                    bpy_manager.gc()
                    print(f">>> Rendering part {dirname}", flush=True)
                    bpy_manager.render(dirname)
                    print(f">>> Finished rendering part {dirname}", flush=True)
                    bpy_manager.gc()
                else:
                    dirname = "source"
                save_path = bpy_manager.export_glb(fname="tmp_glb.glb")
                scene = trimesh.load(save_path, force='mesh')
                os.remove(save_path)
                if not render_only:
                    try:
                        watertight_mesh, scale, center = to_watertight(scene, resolution=512, scale=scale, center=center)  # convert to watertight + sample
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
                        trim_memory()

                    except Exception as e:
                        reason = f"failed converting to watertight mesh or sampling. problematic part: {dirname} exception: {e}"
                        print(f"skipping uid: {uid}, reason: {reason}")
                        skip_uids.append({"uid": uid, "reason": reason})
                        # drop the partial output dir so we don't leave a half-processed uid behind
                        shutil.rmtree(out_path, ignore_errors=True)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        trim_memory()
                        break

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
    timestamped_filename = pjoin(args.data_dir, f"skip_uids_{args.start_ind}_{args.end_ind}_{timestamp}.csv")
    skip_ids_df.to_csv(timestamped_filename, index=False)

