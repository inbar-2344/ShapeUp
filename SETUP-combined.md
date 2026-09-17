# Environment setup

One environment that runs everything: data preparation, geometry inference and
training, and texture inference and training. `full_inference.py` goes from a raw
mesh to a textured one in a single process, so it needs all of it at once.

## Why it can be merged

The two halves already agree on everything that is hard to change:

| | inference | data_preparation (`data_preparation/requirements.txt`) |
|---|---|---|
| Python | 3.10 | 3.10 |
| numpy | 1.26.4 | 1.26.4 |
| torch | **2.5.1+cu124** | **2.4.0+cu121** |

Only torch differs, and the conflict resolves in one direction: keep the
inference build. The inference side has prebuilt CUDA extensions compiled against
torch 2.5.1+cu124 (`nvdiffrast`, `torch-cluster`, `custom_rasterizer`,
`differentiable_renderer`); downgrading torch breaks all of them. The
prep side's only compiled extensions are `cubvh` and `diso`, and those are built
from source anyway, so they can be compiled against 2.5.1+cu124 instead.

> **Do not** run `pip install -r data_preparation/requirements.txt` in this
> environment. It pins `torch==2.4.0+cu121` and would silently downgrade torch,
> breaking the inference extensions. Install the prep packages individually, as
> below.

## Installation

Steps 1-3 build the inference side; steps 4-7 add data preparation.

### 1. Blender's system libraries (once per machine, needs root)

Headless Blender cannot import without these:

```bash
sudo bash data_preparation/setup.sh
```

### 2. Create the environment in ONE conda transaction

Create python, the CUDA toolkit, the compilers **and git together**. Do not split
this across two `conda install` calls:

```bash
conda create -y -n shapeup -c conda-forge -c nvidia \
    python=3.10 \
    "cuda-version=12.4" "cuda-toolkit=12.4" "cuda-nvcc=12.4" \
    gcc=13 gxx=13 ninja cmake pkg-config git
conda activate shapeup
```

> **Why one transaction.** Installing the CUDA toolkit first and the compilers
> second records the transitive `cuda-compiler` build as an explicit spec in the
> environment's history. When a later `conda install` re-solves, that exact build
> may no longer be on the channel and conda aborts with
> `InvalidSpec: The package "nvidia/linux-64::cuda-compiler==12.6.2=0" is not
> available for the specified platform`. This is not a gcc-vs-CUDA conflict and
> pinning `cuda-version` does not fix it — the two-step sequence is the problem.
> Installing the toolkit and the compilers separately has the same trap.

> **Why `git`.** `cubvh` installs from a `git+https://` URL, so pip needs a git
> binary. Compute nodes often have none; getting it from conda avoids that.

### 3. PyTorch, then the rest of the stack

Install torch **before** `requirements.txt`, and `nvdiffrast` after it with
`--no-build-isolation` so its build can see the installed torch:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

`torch-cluster` is required by geometry inference
(`michelangelo_autoencoder` imports `torch_cluster.fps`); `nvdiffrast` by the
texture stage.

### 4. Build environment variables for the CUDA extensions


`cubvh`, `diso` and the texture extensions compile at install time and need the
CUDA toolkit already in the conda prefix from step 2:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
# 8.0=A100, 8.6=A5000/A6000/RTX 3090, 8.9=L40S. Trim to your GPUs to build faster.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
```

### 5. Pure-Python prep dependencies

```bash
# cubvh's setup.py imports pkg_resources, removed in setuptools 81.
pip install "setuptools<81" wheel ninja pybind11
pip install pysdf fpsample==0.3.3 open3d==0.18.0 objaverse==0.1.7 \
            imageio==2.37.0 imageio-ffmpeg==0.6.0
```

`open3d` and `bpy` both require numpy < 2. The inference env is already on
1.26.4 — keep it there.

### 6. Headless Blender

`bpy` 4.0.0 for Python 3.10 is not on PyPI; it comes from Blender's own index:

```bash
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install -e ./data_preparation/bpy-renderer   # provides bpy_ops
```

The local fork is required — upstream `bpyrenderer` does not contain `bpy_ops`.

### 7. The two CUDA extensions

Build these last, after torch, and without build isolation so their setup step
can see the installed torch:

```bash
pip install --no-build-isolation --no-deps diso==0.1.4
pip install --no-build-isolation --no-deps \
    "cubvh @ git+https://github.com/ashawkey/cubvh@4c0de5fc7f9f836fd9c562616bdfeb77a942ba5d"
```

`--no-deps` matters: without it pip is free to "satisfy" a dependency by
installing a different torch or numpy, silently undoing the version decision this
whole document is built on. Confirm the pins survived:

```bash
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
# expect: 2.5.1+cu124 1.26.4
```

### 8. Texture CUDA extensions

The texture stage rasterises and bakes with two compiled extensions:

```bash
pip install --no-build-isolation ./shapeup_texture/custom_rasterizer
pip install --no-build-isolation ./shapeup_texture/differentiable_renderer
```

> conda keeps the CUDA headers under `targets/<arch>/include`, not `include/`, so
> without `CPATH` (step 4) these fail with
> `fatal error: cuda_runtime.h: No such file or directory`.

Both link against torch's shared libraries **at import time**, so this must be set
whenever you run the texture stage, not just to build it:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
```

Without it the import fails with `libc10.so: cannot open shared object file`.

## Verify

```bash
python - <<'PY'
import importlib
prep  = ["bpy", "bpyrenderer", "cubvh", "diso", "pysdf", "fpsample", "open3d"]
infer = ["torch", "diffusers", "transformers", "pymeshlab", "trimesh",
         "shapeup_geometry.models.pipelines.pipeline"]
texture = ["nvdiffrast", "custom_rasterizer", "mesh_processor",
           "shapeup_texture.pipelines.step1x_3d_texture_synthesis_shapeup_pipeline"]
for group, mods in (("prep", prep), ("inference", infer), ("texture", texture)):
    for m in mods:
        try:
            importlib.import_module(m); print(f"OK   [{group}] {m}")
        except Exception as e:
            print(f"FAIL [{group}] {m} -> {type(e).__name__}: {e}")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

Then an end-to-end run:

```bash
python geometry_inference.py \
    --mesh  data/editing_examples/fox/source/src.glb \
    --image data/editing_examples/fox/edited_mv/pixelated.png \
    --save_dir outputs/fox_e2e
```

## Known friction

* **`trimesh` version.** Prep is pinned to 4.8.2; the inference env ships 4.3.2.
  Both halves work on 4.3.2 in practice, so the merged environment leaves it
  alone. Upgrade only if a prep call needs it, and re-check mesh export
  afterwards.
* **GPU memory.** The watertight conversion builds a 512^3 grid; use a 24 GB card
  or larger, or lower `--watertight_resolution`. Geometry inference itself peaks
  around 11.5 GB.
* **`/tmp` space.** The mesh post-processing filters write temporary `.ply`
  files. On a shared node with a full `/tmp` this fails with `OSError: [Errno 28]`;
  pass `--tmpdir` to put scratch somewhere with room.
* **Reproducibility.** float32 inference is not bit-reproducible across GPU
  models — the same input gives a ~7% surface-area difference between an A6000
  and an RTX 3090. Pin a GPU type for reference outputs.
