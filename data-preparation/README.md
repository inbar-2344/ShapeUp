# Data Preparation

Data-generation pipeline for the **ShapeUP** project. It turns raw
[Objaverse](https://objaverse.allenai.org/) 3D assets into the multi-view
renders and sampled point clouds used to train and evaluate the shape models.

For every object it:

1. **Downloads** the asset from Objaverse by `uid`.
2. **Renders** 6 orthographic views (color / depth / normal) with headless
   Blender, for the full object and for several part-removed / animated variants.
3. **Converts** each variant to a watertight mesh
   ([`sharp_edge_sampling/to_watertight_mesh.py`](sharp_edge_sampling/to_watertight_mesh.py)).
4. **Samples** a point cloud per variant — sharp-edge points, coarse surface
   points (with normals, farthest-point sampled), near-surface SDF samples and
   random in-volume SDF samples — saved as a single `pc.npz`.

The sampling strategy follows the Dora-VAE data pipeline combined with sharp-edge
sampling; rendering is done through a local fork of
[`bpy-renderer`](https://github.com/huanngzh/bpy-renderer).

---

## Repository layout

| Path | Description |
| --- | --- |
| [`objaverse_preprocess.py`](objaverse_preprocess.py) | **Main pipeline for static objects.** Decomposes multi-part objects and renders/samples the full object plus part-removed variants. |
| [`objaverse_preprocess_motion.py`](objaverse_preprocess_motion.py) | **Main pipeline for animated objects.** Picks the 3 most-different keyframes and renders/samples each pose. |
| [`shapeup_inference_prep.py`](shapeup_inference_prep.py) | Benchmark / evaluation-set preparation (documented separately). |
| [`run_batched.sh`](run_batched.sh) | Runs a preprocessing script over a `uid` index range in fixed-size batches, restarting the process between batches to release memory. |
| [`setup.sh`](setup.sh) | Installs the system libraries headless Blender needs. |
| `bpy-renderer/` | Local **modified** fork of `bpy-renderer`. Adds `bpy_ops.py`, the high-level renderer used by every script. Installed in editable mode. |
| `sharp_edge_sampling/` | Watertight-mesh conversion and sharp-edge sampling utilities. Only the source files are tracked; bundled sample data is git-ignored. |
| `csvs/` | Objaverse `uid` lists and metadata used as input. |

---

## Installation

Tested with **Python 3.10 + CUDA 12.1 on Linux**. Headless Blender (`bpy`)
requires a Linux environment with the system libraries installed by `setup.sh`.

> **A plain `pip install -r requirements.txt` is not enough.** `cubvh` and
> `diso` are CUDA extensions compiled from source: they need a build toolchain
> (`nvcc` 12.1 + a host **gcc ≤ 12**) and `torch` installed **before** them, and
> must be installed with `--no-build-isolation`. The ordered steps below handle
> all of this. (Steps 3–4 install the toolchain into the conda env; skip them if
> your machine already provides `nvcc` 12.1 and `gcc ≤ 12` on `PATH`.)

```bash
# 1. System libraries for headless Blender (run once; needs root/sudo).
#    Skip if they are already present (e.g. inside a prepared container).
bash setup.sh

# 2. Create and activate an environment
conda create -n dataprep python=3.10 -y && conda activate dataprep

# 3. Build toolchain for the CUDA extensions (cubvh, diso): nvcc 12.1 + gcc <= 12
#    (system gcc 13+ is rejected by the CUDA 12.1 headers).
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 \
                                binutils_linux-64 sysroot_linux-64
export CUDA_HOME="$CONDA_PREFIX"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"   # make nvcc use the conda gcc<=12
export TORCH_CUDA_ARCH_LIST="8.0"         # your GPU's compute capability (8.0 = A100)

# 4. PyTorch first — the CUDA extensions are built against the installed torch.
#    setuptools<81 is required: cubvh's setup.py imports the removed pkg_resources.
pip install "setuptools<81" wheel ninja pybind11
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Everything else (run from THIS directory so `-e ./bpy-renderer` resolves).
#    --no-build-isolation lets cubvh/diso compile against the torch from step 4.
#    The package indexes (PyTorch + Blender's bpy server) are declared inside
#    requirements.txt, so no --extra-index-url is needed here.
pip install -r requirements.txt --no-build-isolation
```

`requirements.txt` is a curated, grouped dependency list. For a byte-exact
reproduction of the original environment use
[`requirements-lock.txt`](requirements-lock.txt) instead (the same build
prerequisites from steps 1–4 apply).

Key non-PyPI dependencies handled by `requirements.txt`:

- **`bpyrenderer`** — installed editable from the local `./bpy-renderer`. This
  copy is required: it contains `bpy_ops.py`, which the upstream package lacks.
- **`cubvh`** — built from GitHub (`ashawkey/cubvh`); needs `--no-build-isolation`.
- **`bpy==4.0.0`** — **not on PyPI** for Python 3.10; it resolves from Blender's
  own index (`https://download.blender.org/pypi/`), declared at the top of
  `requirements.txt`.
- **`diso`**, **`pysdf`**, **`fpsample`**, **`open3d`** — meshing, SDF and
  sampling backends.

The renderer loads an HDRI environment map from
`bpy-renderer/assets/env_textures/brown_photostudio_02_1k.exr`. Override it with
the `BPYRENDERER_ENV_MAP` environment variable if needed.

---

## Input data

Both scripts read a CSV of Objaverse `uid`s (default
[`csvs/unified_dataset.csv`](csvs/unified_dataset.csv)) with these columns:

- `uid` — Objaverse object id.
- `texture_compatible` — `1` selects an object for the **static** pipeline
  (`objaverse_preprocess.py`).
- `motion` — `1` marks an animated object; the **motion** pipeline
  (`objaverse_preprocess_motion.py`) processes objects with `motion == 1`
  **and** `texture_compatible == 1`.

---

## Usage

### Static objects

```bash
python objaverse_preprocess.py \
    --csv_path csvs/unified_dataset.csv \
    --data_dir outputs \
    --start_ind 0 --end_ind 1000 \
    --point_number 65536 \
    --angle_threshold 15
```

### Animated objects

```bash
python objaverse_preprocess_motion.py \
    --csv_path csvs/unified_dataset.csv \
    --data_dir outputs \
    --start_ind 0 --end_ind 1000
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--csv_path` | `csvs/unified_dataset.csv` | Input CSV; needs `uid`, `texture_compatible`, `motion` columns. |
| `--data_dir` | `bpy-renderer/outputs` | Output root directory. |
| `--start_ind` / `--end_ind` | `0` / `4000` | Slice of the `uid` list to process (used for parallel runs). |
| `--point_number` | `65536` | Total points sampled per mesh (split between sharp and coarse). |
| `--angle_threshold` | `15` | Dihedral-angle threshold (degrees) for detecting sharp edges. |
| `--render_only` | off | Only render multi-view images; skip watertight conversion and point sampling. |

### Batch processing

Restart the process between batches so memory is reliably released:

```bash
# ./run_batched.sh START END LOG [SCRIPT]
./run_batched.sh 0 1000 render_0_1000.log
./run_batched.sh 1000 2000 render_1000_2000.log objaverse_preprocess.py
```

Run several ranges in parallel with `nohup ... &` to use multiple GPUs/workers.
`BATCH_SIZE`, `PYTHON` and the target `SCRIPT` are configurable (see the script
header).

---

## Output structure

Each processed `uid` produces a directory under `--data_dir`:

```
<data_dir>/<uid>/
├── source/                 # full object (static)  — or  frame_<i>/ (motion)
│   ├── mv/                 # 6 multi-view color renders (.webp)
│   ├── depth/              # depth maps
│   ├── normal/             # normal + render maps (.webp)
│   ├── meta.json           # camera metadata (positions, elevations, azimuths)
│   └── pc.npz              # sampled point cloud (omitted with --render_only)
├── 0/  1/  2/              # static: same layout, with the top-N parts removed
└── ...
```

Each `pc.npz` contains four `float32` arrays:

| Key | Meaning |
| --- | --- |
| `fps_sharp_surface` | Sharp-edge surface points with normals. |
| `sharp_near_surface` | Near-sharp-edge points with signed distance. |
| `fps_coarse_surface` | Farthest-point-sampled surface points with normals. |
| `rand_points` | Near-surface and random in-volume points with signed distance. |

A skip report listing skipped objects and the reason (e.g. *single mesh* /
*single frame*) is written after each run: `skip_uids_<start>_<end>_<timestamp>.csv`
for the static pipeline, `skip_uids_motion_<start>_<end>_<timestamp>.csv` for the
motion pipeline.

---

## Acknowledgements

- [Dora-VAE](https://github.com/Seed3D/Dora) — point-cloud sampling strategy.
- [bpy-renderer](https://github.com/huanngzh/bpy-renderer) — Blender rendering backend.
- [Objaverse](https://objaverse.allenai.org/) — source 3D assets.
- [cubvh](https://github.com/ashawkey/cubvh), [diso](https://github.com/SarahWeiii/diso) — watertight-mesh conversion.
