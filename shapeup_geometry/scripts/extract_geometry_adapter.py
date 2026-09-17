"""Extract the trained adapter (LoRA + shape-condition projection) from a ShapeUp
geometry DeepSpeed ZeRO checkpoint into a small, self-contained adapter checkpoint.

The full DeepSpeed checkpoint stores every parameter (frozen base transformer,
frozen VAE, frozen image encoder, ...) which is wasteful: only the LoRA weights on
the DiT and the `proj_shape_condtion` projection are actually trained by this
pipeline. Everything else is loaded from the original Step1X-3D weights on the Hub.

The trainable parameters live in the intact optimizer-state shard as fp32 master
copies (`single_partition_of_fp32_groups`) together with a name->slice mapping
(`param_slice_mappings`), so we can recover them without the (large) model-state
shard.

Usage:
    python extract_geometry_adapter.py \
        --config configs/train-geometry-diffusion/step1x-3d-geometry-shapeup-lr1e-parts-reconstruction-train-new-data-zero-edit-inference.yaml \
        --ckpt checkpoints/1024_540k.ckpt \
        --out checkpoints/1024_540k_adapter

Produces:
    <out>/adapter_config.json     LoRA config (rank/alpha/target_modules/init)
    <out>/adapter_model.safetensors   adapter weights, keyed relative to the FluxDenoiser
"""
import argparse
import glob
import json
import os
import warnings
from os.path import join as pjoin

warnings.filterwarnings("ignore")

import torch
from safetensors.torch import save_file

# Scripts live in shapeup_geometry/scripts/, so put the repo root on sys.path.
# data_preparation/ holds the sampling code and is put on sys.path too.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "data_preparation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shapeup_geometry
from shapeup_geometry.utils.config import load_config


DENOISER_PREFIX = "denoiser_model."

# Default LoRA target modules used when the training config does not set `lora_layers`
# (mirrors ShapeupRectifiedFlowSystem.configure()).
DEFAULT_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


def find_optim_state(ckpt_dir):
    matches = glob.glob(pjoin(ckpt_dir, "checkpoint", "*optim_states.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No *optim_states.pt found under {pjoin(ckpt_dir, 'checkpoint')}"
        )
    return sorted(matches)[0]


def build_denoiser_with_adapter(cfg):
    """Instantiate the FluxDenoiser (base weights load from the Hub) and attach the
    LoRA adapter exactly as during training, so we get the trained parameters' names
    and shapes."""
    denoiser = shapeup_geometry.find(cfg.system.denoiser_model_type)(
        cfg.system.denoiser_model
    )

    if not cfg.system.get("use_lora", False):
        raise ValueError("Config has use_lora=False; nothing to extract.")

    from peft import LoraConfig

    # rank/alpha/lora_layers default to the ShapeupRectifiedFlowSystem.Config values
    # when not overridden in the yaml (that is how they were set at training time).
    lora_layers = cfg.system.get("lora_layers", None)
    if lora_layers is not None:
        target_modules = [l.strip() for l in lora_layers.split(",")]
    else:
        target_modules = DEFAULT_TARGET_MODULES

    lora_config = LoraConfig(
        r=cfg.system.get("rank", 128),
        lora_alpha=cfg.system.get("alpha", 128),
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    denoiser.dit_model.add_adapter(lora_config)
    return denoiser, lora_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="inference/training yaml")
    parser.add_argument(
        "--ckpt", required=True, help="DeepSpeed checkpoint directory (…/*.ckpt)"
    )
    parser.add_argument(
        "--out", required=True, help="output directory for the adapter checkpoint"
    )
    args, extras = parser.parse_known_args()

    cfg = load_config(args.config, cli_args=extras, n_gpus=1)

    print(f"Loading optimizer states from {args.ckpt}")
    optim_path = find_optim_state(args.ckpt)
    optim = torch.load(optim_path, map_location="cpu", weights_only=False)
    osd = optim["optimizer_state_dict"]
    flats = osd["single_partition_of_fp32_groups"]
    mappings = osd["param_slice_mappings"]

    # The checkpoint may hold several optimizer param groups; every one of them is
    # trainable state and must end up in the adapter. Bail out loudly rather than
    # silently extracting only the first group.
    if len(flats) != len(mappings):
        raise RuntimeError(
            f"{len(flats)} fp32 param groups but {len(mappings)} slice mappings"
        )
    print(f"  {len(flats)} optimizer param group(s)")
    for gi, (flat, mapping) in enumerate(zip(flats, mappings)):
        mapped = sum(f.numel for f in mapping.values())
        print(
            f"    group {gi}: fp32 partition {tuple(flat.shape)}  |  "
            f"{len(mapping)} trainable tensors  |  {mapped} mapped elements"
        )
        if mapped != flat.numel():
            # A shortfall means part of this rank's partition is not described by the
            # name->slice mapping, so we cannot claim to have recovered everything.
            raise RuntimeError(
                f"group {gi}: slice mappings cover {mapped} of {flat.numel()} "
                "elements; the adapter would be incomplete"
            )

    print("Building FluxDenoiser + LoRA adapter to resolve parameter shapes")
    denoiser, lora_config = build_denoiser_with_adapter(cfg)

    # Every trainable tensor must live inside the denoiser: that is the only module the
    # pipeline loads the adapter into. Anything else (an unfrozen encoder, say) would be
    # trained state we silently drop, so refuse to write a checkpoint in that case.
    outside_denoiser = [
        k for mapping in mappings for k in mapping if not k.startswith(DENOISER_PREFIX)
    ]
    if outside_denoiser:
        raise RuntimeError(
            f"{len(outside_denoiser)} trainable tensors live outside "
            f"'{DENOISER_PREFIX}' and would be lost, e.g. {outside_denoiser[:5]}"
        )

    # name (relative to denoiser) -> (shape, group index), for every mapped parameter
    denoiser_params = dict(denoiser.named_parameters())
    param_info = {}
    for gi, mapping in enumerate(mappings):
        for full in mapping:
            name = full[len(DENOISER_PREFIX):]
            if name not in denoiser_params:
                continue
            param_info[name] = (denoiser_params[name].shape, gi)

    missing_in_model = [
        k[len(DENOISER_PREFIX):]
        for mapping in mappings
        for k in mapping
        if k[len(DENOISER_PREFIX):] not in param_info
    ]
    if missing_in_model:
        raise RuntimeError(
            f"{len(missing_in_model)} checkpoint params not found in the model, e.g. "
            f"{missing_in_model[:5]}"
        )

    adapter_sd = {}
    for name, (shape, gi) in param_info.items():
        frag = mappings[gi][DENOISER_PREFIX + name]
        numel = int(torch.tensor(shape).prod())
        assert numel == frag.numel, f"{name}: shape {tuple(shape)} != numel {frag.numel}"
        chunk = flats[gi][frag.start : frag.start + frag.numel].reshape(shape).clone()
        adapter_sd[name] = chunk

    n_lora = sum(1 for k in adapter_sd if ".lora_" in k)
    n_shape = sum(1 for k in adapter_sd if "proj_shape_condtion" in k)
    total = sum(v.numel() for v in adapter_sd.values())
    n_mapped = sum(len(m) for m in mappings)
    numel_mapped = sum(f.numel for m in mappings for f in m.values())
    print(
        f"Recovered {len(adapter_sd)} tensors "
        f"({n_lora} LoRA + {n_shape} proj_shape_condtion), {total/1e6:.1f}M params"
    )
    assert len(adapter_sd) == n_mapped, (
        f"recovered {len(adapter_sd)} of {n_mapped} trainable tensors"
    )
    assert total == numel_mapped, (
        f"recovered {total} of {numel_mapped} trainable elements"
    )
    print(
        f"  completeness: {len(adapter_sd)}/{n_mapped} tensors, "
        f"{total}/{numel_mapped} elements -- all trainable state captured"
    )

    os.makedirs(args.out, exist_ok=True)
    save_file(adapter_sd, pjoin(args.out, "adapter_model.safetensors"))
    with open(pjoin(args.out, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "rank": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "init_lora_weights": "gaussian",
                "target_modules": list(lora_config.target_modules),
                # Manifest of the trainable state, so the pipeline can confirm at load
                # time that it got the whole adapter and not a truncated copy.
                "num_tensors": len(adapter_sd),
                "num_elements": total,
                "source_checkpoint": os.path.abspath(args.ckpt),
            },
            f,
            indent=2,
        )
    print(f"Saved adapter checkpoint to {args.out}")


if __name__ == "__main__":
    main()
