"""ShapeUp end to end: a mesh plus an edit image in, a textured mesh out.

    python full_inference.py --mesh chair.glb --image edit.png --out_dir outputs/chair

Three stages, all in one process:

  1. prepare  -- watertight conversion + point sampling, and the source
                 multi-view renders that the texture stage conditions on
  2. geometry -- the edited, untextured mesh (raw marching cubes)
  3. texture  -- paints it, guided by the same edit image

    <out_dir>/source/          src.glb, pc.npz, mv/ depth/ normal/, meta.json
    <out_dir>/edited_mv/       the edit image
    <out_dir>/<image>.obj      edited geometry, untextured
    <out_dir>/<image>.glb      edited geometry, textured   <- the result

The renders in stage 1 are required: the texture model reads <out_dir>/source/mv.
"""
import argparse
import os
import tempfile
from os.path import join as pjoin

# geometry_inference lives under shapeup_geometry/scripts/, this file under
# shapeup_texture/scripts/; both plus the repo root go on sys.path.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "shapeup_geometry" / "scripts", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geometry_inference as geom
import texture_inference as tex


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, help="Source mesh to edit (.glb or .obj).")
    p.add_argument("--image", required=True, help="Image describing the edit.")
    p.add_argument("--out_dir", required=True)
    # geometry
    p.add_argument("--guidance_scale_visual", type=float, default=3.5,
                   help="Geometry CFG scale for the edit image.")
    p.add_argument("--guidance_scale_shape", type=float, default=2.5,
                   help="Geometry CFG scale for the source shape.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--octree_resolution", type=int, default=256)
    p.add_argument("--geometry_adapter", default=geom.ADAPTER,
                   help="Geometry adapter directory or HF repo id.")
    # preparation
    p.add_argument("--angle_threshold", type=int, default=15)
    p.add_argument("--point_number", type=int, default=65536)
    p.add_argument("--watertight_resolution", type=int, default=512)
    # texture
    p.add_argument("--cfg_type", default="separate")
    p.add_argument("--guidance_scale", type=float, default=6.5,
                   help="Texture guidance scale.")
    p.add_argument("--guidance_scale_img", type=float, default=2.5,
                   help="Texture CFG scale for the edit image.")
    p.add_argument("--guidance_scale_src_mv", type=float, default=3.5,
                   help="Texture CFG scale for the source multi-view renders.")
    p.add_argument("--texture_checkpoint", default=None,
                   help="Local directory holding step1x-3d-ig2v.safetensors. By "
                        "default it is fetched from the Hugging Face Hub and cached.")
    p.add_argument("--tmpdir", help="Scratch dir for the mesh post-processing "
                                    "filters, which write temporary .ply files.")
    return p.parse_args()


def main():
    args = parse_args()
    for path, what in ((args.mesh, "mesh"), (args.image, "image")):
        if not os.path.isfile(path):
            raise SystemExit(f"No such {what}: {path}")
    if args.tmpdir:
        os.makedirs(args.tmpdir, exist_ok=True)
        tempfile.tempdir = os.environ["TMPDIR"] = args.tmpdir
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"{'='*78}\n1/3 PREPARE\n{'='*78}", flush=True)
    pc_path, edit_image = geom.prepare_inputs(
        args.mesh, args.image, args.out_dir,
        angle_threshold=args.angle_threshold,
        point_number=args.point_number,
        resolution=args.watertight_resolution,
        render=True)          # the texture stage needs source/mv

    print(f"\n{'='*78}\n2/3 GEOMETRY\n{'='*78}", flush=True)
    print(f"cfg    : gs_visual={args.guidance_scale_visual} "
          f"gs_shape={args.guidance_scale_shape} seed={args.seed} "
          f"steps={args.num_inference_steps} octree={args.octree_resolution}")
    geo_pipe = geom.build_pipeline(
        adapter=args.geometry_adapter,
        guidance_scale_visual=args.guidance_scale_visual,
        guidance_scale_shape=args.guidance_scale_shape,
        num_inference_steps=args.num_inference_steps,
        octree_resolution=args.octree_resolution)
    mesh = geom.generate(geo_pipe, pc_path, edit_image, seed=args.seed,
                         num_inference_steps=args.num_inference_steps,
                         octree_resolution=args.octree_resolution,
                         guidance_scale_visual=args.guidance_scale_visual,
                         guidance_scale_shape=args.guidance_scale_shape)
    if mesh is None:
        raise SystemExit("Geometry stage produced an empty mesh.")
    geo_path = pjoin(args.out_dir, f"{name}.obj")
    mesh.export(geo_path)
    print("V%d F%d area %.5f vol %.5f" % (len(mesh.vertices), len(mesh.faces),
                                          mesh.area, mesh.volume))
    print(f"Exported {geo_path}")
    del geo_pipe

    print(f"\n{'='*78}\n3/3 TEXTURE\n{'='*78}", flush=True)
    print(f"cfg    : cfg_type={args.cfg_type} "
          f"guidance_scale={args.guidance_scale} "
          f"guidance_scale_img={args.guidance_scale_img} "
          f"guidance_scale_src_mv={args.guidance_scale_src_mv}")
    tex_pipe = tex.build_pipeline(args.texture_checkpoint)
    textured_mesh, _, _, _ = tex.texture(
        tex_pipe, geo_path, edit_image, pjoin(args.out_dir, "source"),
        guidance_scale=args.guidance_scale, cfg_type=args.cfg_type,
        guidance_scale_img=args.guidance_scale_img,
        guidance_scale_src_mv=args.guidance_scale_src_mv)
    out_path = pjoin(args.out_dir, f"{name}.glb")
    textured_mesh.export(out_path)
    print(f"Exported {out_path}")
    print(f"\n{'='*78}\nDone: {out_path}")


if __name__ == "__main__":
    main()
