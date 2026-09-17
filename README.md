# ShapeUp

Official implementation of **ShapeUP**.

[![arXiv](https://img.shields.io/badge/arXiv-2602.05676-b31b1b.svg)](https://arxiv.org/abs/2602.05676)
[![Project page](https://img.shields.io/badge/Project-page-blue.svg)](https://inbar-2344.github.io/ShapeUp-page/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Inbar2344/ShapeUP-yellow)](https://huggingface.co/Inbar2344/ShapeUP)


Image-conditioned 3D editing: given a source mesh and an *edit image*, ShapeUp
produces an edited, textured mesh. Two stages, both built on
[Step1X-3D](https://github.com/stepfun-ai/Step1X-3D):

1. **Geometry** — a rectified-flow diffusion model (a LoRA adapter) edits the shape.
2. **Texture** — an IG2MV SDXL model paints it, guided by the same edit image and
   the source shape's multi-view renders.

## Layout

```
data_preparation/          watertight conversion, point sampling, Blender rendering
    objaverse_preprocess.py       build training data from Objaverse assets
    shapeup_inference_prep.py     prepare local meshes (used by the inference scripts)
    sampling_utils.py             the shared sampling code both of the above call
demo.py                    run a bundled example through both stages
shapeup_geometry/          the geometry package
    geometry_spec.py       architecture + sampling values used at inference
    scripts/
        geometry_inference.py  mesh + edit image -> edited mesh
        train.py               training entry point
        create_vae_data.py     precompute VAE latents for training
        extract_geometry_adapter.py   slim a checkpoint down to the adapter
shapeup_texture/           the texture package
    scripts/
        texture_inference.py   edited mesh + edit image + source views -> textured
        full_inference.py      mesh + edit image -> textured mesh (all stages)
configs/
    train-geometry-diffusion/   geometry training config
    geometry-autoencoder/       config for precomputing VAE latents
    train-texture-ig2mv/        texture training config
data/
    editing_examples/      validation examples (source/pc.npz + edited_mv/)
    train.json, val.json, test.json, train_motion.json
```

## Setup

See **[`SETUP-combined.md`](SETUP-combined.md)** — one environment that runs data
preparation, geometry and texture. Data preparation and inference have different
dependency sets, and the guide explains how they reconcile (they agree on Python
3.10 and numpy 1.26.4; only torch conflicts, and the inference build wins).

## Demo

The bundled examples ship a sampled point cloud and the source renders, so this
skips data preparation and runs both model stages:

```bash
python demo.py --out_dir outputs/demo                    # fox
python demo.py --example robot --out_dir outputs/robot
```

```
<out_dir>/<edit image name>.obj   edited geometry, untextured
<out_dir>/<edit image name>.glb   edited geometry, textured   <- the result
```

Both checkpoints come from the Hugging Face Hub on first use and are cached:
the geometry adapter from `Inbar2344/ShapeUP` (`geometry/adapter-1024-550k`) and
the texture checkpoint from the same repo (`texture/ig2mv-cross-attention`).
`--geometry_adapter` and `--texture_checkpoint` point at local copies instead.

The `.obj` is the **raw marching-cubes surface**, which is what the texture stage
expects. It carries floaters and is not decimated, so it looks rough on its own;
`geometry_inference.py --post_process` cleans it (floater removal, decimation to
200k faces, smooth shading) if you want the untextured mesh as a standalone asset.

## Data preparation

### For inference

Nothing to do by hand. `full_inference.py` and `geometry_inference.py` call the
data-preparation code themselves: watertight conversion, point sampling, and the
Blender renders of the source shape. They write a directory in the same layout as
`data/editing_examples/*`:

```
<out_dir>/source/src.glb     the source mesh
<out_dir>/source/pc.npz      sampled points: coarse + sharp surface, SDF samples
<out_dir>/source/mv/         multi-view renders   <- the texture stage reads these
<out_dir>/source/depth/ normal/ meta.json
<out_dir>/edited_mv/         the edit image
```

To prepare a batch of meshes up front instead:

```bash
python data_preparation/shapeup_inference_prep.py \
    --local_glb_dir my_meshes/ --out_dir prepared/
```

### For training

Training data comes from Objaverse assets, rendered and sampled into the same
per-object layout:

```bash
python data_preparation/objaverse_preprocess.py \
    --csv_path data_preparation/csvs/<uid list>.csv \
    --data_dir data/surfaces --point_number 65536
```

`objaverse_preprocess_motion.py` is the equivalent for animated assets, picking the
three most-different keyframes. `run_batched.sh` drives either over a uid range in
batches, restarting between them to release memory. See
`data_preparation/README.md` for the full pipeline description.

The geometry trainer expects `<root_dir>/surfaces/<uid>/` plus these lists at
`<root_dir>/`:

| file | contents |
| --- | --- |
| `train.json` | uids used for training |
| `val.json` / `test.json` | uids used for validation / test |
| `train_motion.json` | uids that are motion samples, oversampled by `motion_weight` |

Validation reads `<validation_root_dir>/<uid>/source/pc.npz` and every image in
`<validation_root_dir>/<uid>/edited_mv/`, so the bundled `data/editing_examples`
works as-is.

**Geometry only:** precompute the VAE latents the diffusion model trains against:

```bash
python shapeup_geometry/scripts/create_vae_data.py \
    --config configs/geometry-autoencoder/michelangelo_data_creation_shapeup.yaml --gpu 0
```

## Inference — everything at once

```bash
python shapeup_texture/scripts/full_inference.py \
    --mesh chair.glb --image edit.png --out_dir outputs/chair
```

Prepares the inputs, edits the geometry, then textures it:

```
outputs/chair/source/       src.glb, pc.npz, mv/ depth/ normal/, meta.json
outputs/chair/edited_mv/    the edit image
outputs/chair/<image>.obj   edited geometry, untextured
outputs/chair/<image>.glb   edited geometry, textured   <- the result
```

The source renders are not optional here: the texture model reads
`<out_dir>/source/mv`.

## Inference — texture only

If you already have an edited mesh:

```bash
python shapeup_texture/scripts/texture_inference.py \
    --source_dir data/editing_examples/batman/source \
    --image      data/editing_examples/batman/edited_mv/cross_bow.png \
    --mesh       outputs/batman/cross_bow.obj \
    --out_dir    outputs/batman_textured
```

The texture checkpoint is fetched from `Inbar2344/ShapeUP`
(`texture/ig2mv-cross-attention`) on first use. `--checkpoint <dir>` overrides it.

## Inference — geometry only

```bash
python shapeup_geometry/scripts/geometry_inference.py \
    --mesh chair.glb --image edit.png --save_dir outputs/chair
```

This prepares the inputs first (watertight conversion, point sampling, optional
renders) and then runs the model, so it needs the combined environment. Use
`--no_render` to skip the multi-view renders, which inference does not use.

## Training

Both trainers take `--config`, `--train`, and `--gpu` (a comma-separated list).
Each config carries its measured parameter counts in a header comment.

### Geometry

```bash
python shapeup_geometry/scripts/train.py \
    --config configs/train-geometry-diffusion/shapeup-train.yaml --train --gpu 0
```

Only the adapter trains: the shape model and the visual encoder are frozen, and the
DiT is frozen apart from its LoRA layers — **75.6 M trainable of 2.27 B**. Point
`data.root_dir` at the directory holding `train.json` and `surfaces/`, and
`data.validation_root_dir` at the examples to validate on.

Motion samples listed in `train_motion.json` are oversampled by `motion_weight`
(3.0) through a weighted sampler.

When a run finishes, slim the checkpoint down to just the trained weights before
sharing it:

```bash
python shapeup_geometry/scripts/extract_geometry_adapter.py \
    --ckpt <trial_dir>/ckpts/last.ckpt --out checkpoints/my_adapter
```

### Texture

```bash
python shapeup_texture/scripts/train_ig2mv.py \
    --config configs/train-texture-ig2mv/shapeup_ca_train.yaml --train --gpu 0
```

Cross-attention conditioning. The VAE and both text encoders are frozen; what trains
is the multi-view / reference attention in the UNet plus the condition encoder —
**901 M trainable of 4.37 B**. The base IG2MV adapter it starts from is pulled from
`stepfun-ai/Step1X-3D` on first use, so no local checkpoint is needed.

Training writes `step1x-3d-ig2v.safetensors` at every checkpoint, already containing
only the trained weights. It is saved in fp32; converting to fp16 halves the file
with no practical loss for inference.

### Where runs are written

`outputs/<name>/<tag>`, holding `ckpts/`, `save/`, `configs/`, and TensorBoard and
CSV logs. The timestamp is **disabled when training on more than one GPU**, so give
each multi-GPU run a unique `tag` or it will overwrite the previous one:

```bash
python shapeup_geometry/scripts/train.py --config <cfg> --train --gpu 0,1,2,3 tag=my-run
```

Set `resume: <path to .ckpt>` in the config to continue from a checkpoint.
