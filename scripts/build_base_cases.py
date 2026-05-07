"""CLI driver for Phase 1 base_cases.csv generation.

Usage:
    python scripts/build_base_cases.py \\
        --sampling protocol/dataset_sampling.yaml \\
        --mimic-root /path/to/mimic-cxr-jpg/2.1.0 \\
        --chexpert-root /path/to/CheXpert-v1.0 \\
        --gmai-root /path/to/GMAI-MMBench \\
        --output data/processed/base_cases.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import build_base_cases  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", required=True,
                    help="Path to protocol/dataset_sampling.yaml")
    ap.add_argument("--mimic-root", required=False,
                    help="MIMIC-CXR-JPG root (containing metadata CSVs and files/)")
    ap.add_argument("--chexpert-root", required=False,
                    help="CheXpert root (containing train.csv / valid.csv)")
    ap.add_argument("--gmai-root", required=False,
                    help="GMAI-MMBench root (containing gmai_mmbench.jsonl)")
    ap.add_argument("--output", default="data/processed/base_cases.csv")
    ap.add_argument("--pilot-size", type=int, default=200)
    ap.add_argument("--full-size", type=int, default=3800)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    dataset_roots = {}
    if args.mimic_root:
        dataset_roots["mimic_cxr_jpg"] = Path(args.mimic_root)
    if args.chexpert_root:
        dataset_roots["chexpert"] = Path(args.chexpert_root)
    if args.gmai_root:
        dataset_roots["gmai_mmbench"] = Path(args.gmai_root)

    if not dataset_roots:
        print("ERROR: provide at least one --*-root flag.", file=sys.stderr)
        return 2

    n = build_base_cases(
        sampling_yaml=Path(args.sampling),
        dataset_roots=dataset_roots,
        output_csv=Path(args.output),
        pilot_size=args.pilot_size,
        full_size=args.full_size,
        seed=args.seed,
    )
    print(f"Wrote {n} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
