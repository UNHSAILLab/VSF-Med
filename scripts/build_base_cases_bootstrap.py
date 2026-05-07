"""Bootstrap CLI for Phase 1 — runs against local MIMIC-VQA data instead of
canonical PhysioNet/CheXpert/GMAI-MMBench. Produces a real base_cases.csv
that the rest of the pipeline can consume immediately.

Usage:
    python scripts/build_base_cases_bootstrap.py
    python scripts/build_base_cases_bootstrap.py --target 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import (  # noqa: E402
    MimicVqaLoader,
    build_base_cases,
    register_loader,
)


DEFAULT_IMAGE_DIRS = [
    "/home/bsada1/datasets/MIMIC_JPG/thousandfiles",
    "/home/bsada1/mimic_cxr_hundred_vqa",
]
DEFAULT_VQA_ROOT = ROOT / "data"            # contains train/valid/test.json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", default=str(ROOT / "protocol/dataset_sampling.bootstrap.yaml"))
    ap.add_argument("--vqa-root", default=str(DEFAULT_VQA_ROOT),
                    help="Directory containing train.json / valid.json / test.json")
    ap.add_argument("--image-dir", action="append", default=None,
                    help="Image directory (flat <image_id>.jpg files). Repeat for multiple. "
                         f"Defaults: {DEFAULT_IMAGE_DIRS}")
    ap.add_argument("--max-per-image", type=int, default=3,
                    help="Cap Q/A pairs per image to avoid one image dominating. Default 3.")
    ap.add_argument("--target", type=int, default=None,
                    help="Override target_count from the YAML.")
    ap.add_argument("--output", default=str(ROOT / "data/processed/base_cases.csv"))
    ap.add_argument("--pilot-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    image_dirs = [Path(d) for d in (args.image_dir or DEFAULT_IMAGE_DIRS)]
    print(f"Image dirs: {image_dirs}")
    print(f"VQA root:   {args.vqa_root}")

    register_loader("mimic_cxr_vqa", MimicVqaLoader(
        image_dirs=image_dirs,
        max_per_image=args.max_per_image,
    ))

    # Optionally override the YAML target_count
    sampling_path = Path(args.sampling)
    if args.target is not None:
        import yaml, tempfile
        spec = yaml.safe_load(sampling_path.read_text())
        for ds in spec["datasets"]:
            if ds["name"] == "mimic_cxr_vqa":
                ds["target_count"] = args.target
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(spec, tmp)
        tmp.close()
        sampling_path = Path(tmp.name)

    n = build_base_cases(
        sampling_yaml=sampling_path,
        dataset_roots={"mimic_cxr_vqa": Path(args.vqa_root)},
        output_csv=Path(args.output),
        pilot_size=args.pilot_size,
        full_size=10_000,        # large; effective cap is target_count
        seed=args.seed,
    )
    print(f"\nWrote {n} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
