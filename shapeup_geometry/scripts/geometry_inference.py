"""Edit a mesh with ShapeUp: sample the source mesh, then run geometry inference.

    python geometry_inference.py --mesh chair.glb --image edit.png --save_dir out/chair

Writes the prepared inputs and the edited mesh into --save_dir:

    <save_dir>/source/src.glb        the source mesh (converted if it was .obj)
    <save_dir>/source/pc.npz         sampled point cloud fed to the model
    <save_dir>/source/mv|depth|normal, meta.json   renders (skip with --no_render)
    <save_dir>/edited_mv/<image>     the edit image
    <save_dir>/<image name>.obj      the edited mesh (raw marching cubes)
"""
import argparse
import math
import os
import shutil
import sys
import tempfile
from os.path import join as pjoin

import numpy as np
import torch
import trimesh
from PIL import Image

# Scripts live in shapeup_geometry/scripts/, so put the repo root on sys.path.
# data_preparation/ holds the sampling code and is put on sys.path too.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "data_preparation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from shapeup_inference_prep import render_source_views, sample_and_save_pc

from shapeup_geometry.geometry_spec import GEOMETRY_SPEC

from shapeup_geometry.models.pipelines.pipeline import (
    SHAPEUP_HUB_REPO,
    ShapeUpGeometryPipeline,
)

# Downloaded from the Hub on first use; pass a local directory to override.
ADAPTER = SHAPEUP_HUB_REPO
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")



def prepare_inputs(mesh_path, image_path, save_dir, angle_threshold=15,
                   point_number=65536, resolution=512, render=True):
    """Sample the source mesh and lay out the inputs. Returns (pc.npz, edit image)."""
    source_dir, edited_dir = pjoin(save_dir, "source"), pjoin(save_dir, "edited_mv")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)

    src_glb = pjoin(source_dir, "src.glb")
    if os.path.splitext(mesh_path)[1].lower() == ".glb":
        shutil.copyfile(mesh_path, src_glb)
    else:
        trimesh.load(mesh_path).export(src_glb)

    if render:
        print("[prep] rendering source views")
        render_source_views(src_glb, save_dir)

    print(f"[prep] sampling {point_number} points (watertight res {resolution})")
    pc_path = pjoin(source_dir, "pc.npz")
    sample_and_save_pc(src_glb, pc_path, math.radians(angle_threshold),
                       point_number=point_number, resolution=resolution)

    edit_image = pjoin(edited_dir, os.path.basename(image_path))
    shutil.copyfile(image_path, edit_image)
    return pc_path, edit_image


def build_pipeline(adapter=ADAPTER, guidance_scale_visual=3.5,
                   guidance_scale_shape=2.5, num_inference_steps=30,
                   octree_resolution=256):
    spec = dict(GEOMETRY_SPEC)
    spec["guidance_scale_visual"] = guidance_scale_visual
    spec["guidance_scale_shape"] = guidance_scale_shape
    spec["num_inference_steps"] = num_inference_steps
    spec["octree_resolution"] = octree_resolution
    return ShapeUpGeometryPipeline.from_spec(spec, adapter_path=adapter,
                                             device="cuda", dtype=torch.float32)


def generate(pipe, pc_path, image_path, post_process=False, seed=10,
             num_inference_steps=30,
             octree_resolution=256, guidance_scale_visual=3.5,
             guidance_scale_shape=2.5):
    """Run the model on one prepared point cloud + edit image. Returns a trimesh."""
    pc = np.load(pc_path)
    batch = {
        "surface": torch.from_numpy(pc["fps_coarse_surface"][:, 0]).float().unsqueeze(0),
        "sharp_surface": torch.from_numpy(pc["fps_sharp_surface"][:, 0]).float().unsqueeze(0),
        "image": [Image.open(image_path)],
    }
    out = pipe(batch, seed=seed,
               num_inference_steps=num_inference_steps,
               octree_resolution=octree_resolution,
               cfg_type="separate_visual",
               guidance_scale_visual=guidance_scale_visual,
               guidance_scale_shape=guidance_scale_shape,
               match_reference=False,
               do_remove_floater=post_process,
               do_reduce_face=post_process,
               do_shade_smooth=post_process,
               output_type="trimesh")
    return out.mesh[0]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="Source mesh to edit (.glb or .obj).")
    p.add_argument("--image", required=True, help="Image describing the edit.")
    p.add_argument("--save_dir", required=True,
                   help="Where the prepared inputs and the edited mesh are written.")
    p.add_argument("--guidance_scale_visual", type=float, default=3.5,
                   help="CFG scale for the edit image.")
    p.add_argument("--guidance_scale_shape", type=float, default=2.5,
                   help="CFG scale for the source shape.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--octree_resolution", type=int, default=256,
                   help="Marching-cubes grid. Higher = finer, slower, more memory.")
    p.add_argument("--adapter", default=ADAPTER,
                   help="Adapter: a local directory or a Hub repo id. "
                        f"Default: {ADAPTER} (downloaded on first use).")
    p.add_argument("--angle_threshold", type=int, default=15,
                   help="Dihedral angle (degrees) above which an edge counts as sharp.")
    p.add_argument("--point_number", type=int, default=65536,
                   help="Points sampled, split evenly between coarse and sharp.")
    p.add_argument("--watertight_resolution", type=int, default=512,
                   help="Grid resolution for the watertight conversion.")
    p.add_argument("--post_process", action="store_true",
                   help="Clean the mesh before export: remove floaters, decimate to "
                        "200k faces, smooth shading. Off by default -- the raw "
                        "marching-cubes surface is what the texture stage expects.")
    p.add_argument("--no_render", action="store_true",
                   help="Skip the multi-view/depth/normal renders; inference needs "
                        "only pc.npz.")
    p.add_argument("--tmpdir", help="Scratch dir for the mesh post-processing, which "
                                    "writes temporary .ply files. Use it when the "
                                    "machine's /tmp is small or full.")
    return p.parse_args()


def main():
    args = parse_args()
    for path, what in ((args.mesh, "mesh"), (args.image, "image")):
        if not os.path.isfile(path):
            raise SystemExit(f"No such {what}: {path}")
    if args.tmpdir:
        os.makedirs(args.tmpdir, exist_ok=True)
        tempfile.tempdir = os.environ["TMPDIR"] = args.tmpdir

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"mesh   : {args.mesh}\nimage  : {args.image}\nout    : {args.save_dir}")
    pc_path, edit_image = prepare_inputs(
        args.mesh, args.image, args.save_dir,
        angle_threshold=args.angle_threshold, point_number=args.point_number,
        resolution=args.watertight_resolution, render=not args.no_render)

    print(f"[infer] gs_visual={args.guidance_scale_visual} "
          f"gs_shape={args.guidance_scale_shape} seed={args.seed} "
          f"steps={args.num_inference_steps} octree={args.octree_resolution}")
    pipe = build_pipeline(adapter=args.adapter,
                          guidance_scale_visual=args.guidance_scale_visual,
                          guidance_scale_shape=args.guidance_scale_shape,
                          num_inference_steps=args.num_inference_steps,
                          octree_resolution=args.octree_resolution)
    mesh = generate(pipe, pc_path, edit_image, post_process=args.post_process,
                    seed=args.seed,
                    num_inference_steps=args.num_inference_steps,
                    octree_resolution=args.octree_resolution,
                    guidance_scale_visual=args.guidance_scale_visual,
                    guidance_scale_shape=args.guidance_scale_shape)
    if mesh is None:
        raise SystemExit("Model produced an empty mesh.")

    name = os.path.splitext(os.path.basename(edit_image))[0]
    out_path = pjoin(args.save_dir, f"{name}.obj")
    mesh.export(out_path)
    print("V%d F%d area %.5f vol %.5f" % (len(mesh.vertices), len(mesh.faces),
                                          mesh.area, mesh.volume))
    print(f"Exported {out_path}")


if __name__ == "__main__":
    main()
