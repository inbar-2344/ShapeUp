"""Run ShapeUp on one of the bundled examples: edit the shape, then texture it.

    python demo.py --out_dir outputs/demo                  # fox
    python demo.py --example robot --out_dir outputs/robot

Each example already ships a sampled point cloud and the source multi-view
renders, so this skips data preparation and runs the two model stages only:

    <out_dir>/<edit image name>.obj   edited geometry, untextured
    <out_dir>/<edit image name>.glb   edited geometry, textured   <- the result

Both checkpoints are downloaded from the Hugging Face Hub on first use and cached.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "shapeup_texture" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shapeup_geometry.geometry_spec import GEOMETRY_SPEC
from shapeup_geometry.models.pipelines.pipeline import (
    SHAPEUP_HUB_REPO,
    ShapeUpGeometryPipeline,
)
import texture_inference as tex

EXAMPLES = ROOT / "data" / "editing_examples"
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example", default="fox",
                   help=f"Example directory name under {EXAMPLES}. Default: fox.")
    p.add_argument("--out_dir", default="outputs/demo",
                   help="Where the edited and textured meshes are written.")
    # geometry
    p.add_argument("--guidance_scale_visual", type=float, default=3.5)
    p.add_argument("--guidance_scale_shape", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--geometry_adapter", default=SHAPEUP_HUB_REPO,
                   help="Geometry adapter: a local directory or a Hub repo id.")
    # texture
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--guidance_scale_img", type=float, default=2.5)
    p.add_argument("--guidance_scale_src_mv", type=float, default=3.5)
    p.add_argument("--texture_checkpoint", default=None,
                   help="Local directory holding step1x-3d-ig2v.safetensors.")
    return p.parse_args()


def main():
    args = parse_args()
    sample = EXAMPLES / args.example
    pc_path = sample / "source" / "pc.npz"
    if not pc_path.is_file():
        available = ", ".join(sorted(d.name for d in EXAMPLES.iterdir() if d.is_dir()))
        raise SystemExit(f"No {pc_path} -- available examples: {available}")
    if not (sample / "source" / "mv").is_dir():
        raise SystemExit(f"No {sample / 'source' / 'mv'} -- the texture stage needs "
                         "the source multi-view renders.")
    images = sorted(p for p in (sample / "edited_mv").iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No edit images in {sample / 'edited_mv'}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"example : {sample}")
    print(f"images  : {', '.join(p.name for p in images)}")
    print(f"out_dir : {args.out_dir}")

    # ---------------- geometry ----------------
    print(f"\n{'='*78}\n1/2 GEOMETRY\n{'='*78}", flush=True)
    spec = dict(GEOMETRY_SPEC)
    spec["guidance_scale_visual"] = args.guidance_scale_visual
    spec["guidance_scale_shape"] = args.guidance_scale_shape
    geo_pipe = ShapeUpGeometryPipeline.from_spec(
        spec, adapter_path=args.geometry_adapter, device="cuda", dtype=torch.float32)

    pc = np.load(pc_path)
    surface = torch.from_numpy(pc["fps_coarse_surface"][:, 0]).float().unsqueeze(0)
    sharp_surface = torch.from_numpy(pc["fps_sharp_surface"][:, 0]).float().unsqueeze(0)

    meshes = {}
    for image_path in images:
        out = geo_pipe({"surface": surface, "sharp_surface": sharp_surface,
                        "image": [Image.open(image_path)]},
                       seed=args.seed, num_inference_steps=30, octree_resolution=256,
                       cfg_type="separate_visual",
                       guidance_scale_visual=args.guidance_scale_visual,
                       guidance_scale_shape=args.guidance_scale_shape,
                       # Raw marching cubes: the texture stage takes this surface
                       # directly, and cleaning it first degrades the result.
                       do_remove_floater=False,
                       do_reduce_face=False,
                       do_shade_smooth=False,
                       output_type="trimesh")
        mesh = out.mesh[0]
        if mesh is None:
            print(f"{image_path.name}: empty mesh, skipping")
            continue
        geo_path = os.path.join(args.out_dir, f"{image_path.stem}.obj")
        mesh.export(geo_path)
        meshes[image_path] = geo_path
        print(f"{image_path.name}: V{len(mesh.vertices)} F{len(mesh.faces)} -> {geo_path}")
    del geo_pipe
    if not meshes:
        raise SystemExit("Geometry stage produced no meshes.")

    # ---------------- texture ----------------
    print(f"\n{'='*78}\n2/2 TEXTURE\n{'='*78}", flush=True)
    tex_pipe = tex.build_pipeline(args.texture_checkpoint)
    for image_path, geo_path in meshes.items():
        textured, _, _, _ = tex.texture(
            tex_pipe, geo_path, str(image_path), str(sample / "source"),
            guidance_scale=args.guidance_scale,
            guidance_scale_img=args.guidance_scale_img,
            guidance_scale_src_mv=args.guidance_scale_src_mv)
        out_path = os.path.join(args.out_dir, f"{image_path.stem}.glb")
        textured.export(out_path)
        print(f"{image_path.name}: -> {out_path}")

    print(f"\n{'='*78}\nDone: {args.out_dir}")


if __name__ == "__main__":
    main()
