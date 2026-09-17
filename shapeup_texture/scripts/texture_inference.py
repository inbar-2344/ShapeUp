"""Texture one edited mesh.

    python texture_inference.py \
        --source_dir data/editing_examples/batman/source \
        --image      data/editing_examples/batman/edited_mv/bronze_cape.png \
        --mesh       outputs/FINAL_RES_PIPE/geometry/batman/bronze_cape.obj \
        --out_dir    outputs/my_texture

Writes <out_dir>/<edit image name>.glb.

--source_dir is the *source* shape's directory and must contain mv/ -- the
multi-view renders of the unedited shape, which condition the texture model.
--mesh is the edited, untextured geometry; the raw marching-cubes surface that
geometry inference writes is what this stage expects.
"""
import argparse
import os
from os.path import join as pjoin

import numpy as np
import trimesh
from PIL import Image

# Scripts live in shapeup_texture/scripts/, so put the repo root on sys.path.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapeup_geometry.models.pipelines.pipeline_utils import (
    reduce_face,
    remove_degenerate_face,
)
from shapeup_texture.pipelines.step1x_3d_texture_synthesis_shapeup_pipeline import (
    Step1X3DTexturePipeline,
)

# The texture checkpoint is downloaded from the Hub on first use and cached.
# Point this at a local directory holding step1x-3d-ig2v.safetensors to override.
CHECKPOINT_ENV = "SHAPEUP_TEXTURE_CHECKPOINT"


def export_tensor_as_pngs(tensor, output_dir):
    """Write a [B, H, W, C] tensor in [0, 1] as render_XXXX.png."""
    os.makedirs(output_dir, exist_ok=True)
    images = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    for i in range(images.shape[0]):
        Image.fromarray(images[i]).save(pjoin(output_dir, f"render_{i:04d}.png"))


def build_pipeline(checkpoint=None):
    """Build the texture pipeline. `checkpoint` overrides the Hub download."""
    if checkpoint:
        os.environ[CHECKPOINT_ENV] = checkpoint
    return Step1X3DTexturePipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture",
        cond_type="cross_attention")


def texture(pipeline, mesh_path, image_path, source_dir, guidance_scale=6.5,
            cfg_type="separate", guidance_scale_img=2.5,
            guidance_scale_src_mv=3.5):
    """Untextured mesh + edit image + source views -> textured mesh."""
    mesh = trimesh.load(mesh_path)
    mesh = reduce_face(remove_degenerate_face(mesh))
    textured_mesh, edited_view_attr, pos_images, normal_images = pipeline(
        image_path, mesh, source_dir,
        guidance_scale=guidance_scale,
        cfg_type=cfg_type,
        guidance_scale_img=guidance_scale_img,
        guidance_scale_src_mv=guidance_scale_src_mv)
    return textured_mesh, edited_view_attr, pos_images, normal_images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_dir", required=True,
                   help="Source shape directory; must contain mv/.")
    p.add_argument("--image", required=True, help="Image describing the edit.")
    p.add_argument("--mesh", required=True,
                   help="Edited, untextured mesh (.obj/.glb) to texture.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cfg_type", default="separate",
                   help='"single" uses one joint guidance_scale; anything else guides '
                        "the edit image and the source views independently.")
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--guidance_scale_img", type=float, default=2.5,
                   help="CFG scale for the edit image.")
    p.add_argument("--guidance_scale_src_mv", type=float, default=3.5,
                   help="CFG scale for the source multi-view renders.")
    p.add_argument("--checkpoint", default=None,
                   help="Local directory holding step1x-3d-ig2v.safetensors. By "
                        "default the checkpoint is fetched from the Hugging Face Hub "
                        "on first use and cached.")
    p.add_argument("--save_views", action="store_true",
                   help="Also write the generated views, position and normal maps.")
    return p.parse_args()


def main():
    args = parse_args()
    for path, what in ((args.mesh, "mesh"), (args.image, "image")):
        if not os.path.isfile(path):
            raise SystemExit(f"No such {what}: {path}")
    if not os.path.isdir(pjoin(args.source_dir, "mv")):
        raise SystemExit(f"No {pjoin(args.source_dir, 'mv')} -- --source_dir must be "
                         "the source shape directory containing its mv/ renders.")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"mesh   : {args.mesh}\nimage  : {args.image}\nsource : {args.source_dir}")
    print(f"cfg    : cfg_type={args.cfg_type} "
          f"guidance_scale={args.guidance_scale} "
          f"guidance_scale_img={args.guidance_scale_img} "
          f"guidance_scale_src_mv={args.guidance_scale_src_mv}")

    pipeline = build_pipeline(args.checkpoint)
    textured_mesh, views, pos_images, normal_images = texture(
        pipeline, args.mesh, args.image, args.source_dir,
        guidance_scale=args.guidance_scale, cfg_type=args.cfg_type,
        guidance_scale_img=args.guidance_scale_img,
        guidance_scale_src_mv=args.guidance_scale_src_mv)

    name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = pjoin(args.out_dir, f"{name}.glb")
    textured_mesh.export(out_path)
    print(f"Exported {out_path}")

    if args.save_views:
        for tensor, sub in ((views, "edited_views"), (pos_images, "position"),
                            (normal_images, "normal")):
            if tensor is not None:
                export_tensor_as_pngs(tensor, pjoin(args.out_dir, sub))
                print(f"Wrote {pjoin(args.out_dir, sub)}")


if __name__ == "__main__":
    main()
