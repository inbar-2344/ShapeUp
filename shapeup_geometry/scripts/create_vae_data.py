import warnings

warnings.filterwarnings("ignore")
import argparse

# Scripts live in shapeup_geometry/scripts/, so put the repo root on sys.path.
# data_preparation/ holds the sampling code and is put on sys.path too.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "data_preparation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shapeup_geometry.models.pipelines.pipeline import VAEDataCreationPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling this script, "
        "this argument is ignored and all available GPUs are always used.",
    )
    parser.add_argument(
        "--ids_list",
        nargs="+",
        default=None,
        help="surface ids to process (subdirectory names under <root_dir>/surfaces). "
        "If omitted, every subdirectory of <root_dir>/surfaces is processed.",
    )

    args, extras = parser.parse_known_args()

    vae_pipeline = VAEDataCreationPipeline(config_path=args.config)
    vae_pipeline(ids_list=args.ids_list)
