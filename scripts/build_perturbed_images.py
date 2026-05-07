"""CLI driver for Phase 2 visual perturbation generation.

Usage:
    python scripts/build_perturbed_images.py \\
        --base-cases data/processed/base_cases.csv \\
        --output-dir data/perturbed \\
        --metadata data/processed/perturbed_metadata.csv

By default generates all 8 methods. Use --methods to subset.
Quality control: rows in metadata CSV report SSIM/PSNR per (case, method).
The Go/No-Go 2 audit consumes this metadata to verify diagnostic readability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perturbations import build_perturbed_images, PERTURBATION_FILENAME_PREFIX  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cases", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata", default=None,
                    help="Optional CSV path for SSIM/PSNR metadata.")
    ap.add_argument("--methods", nargs="*", default=None,
                    choices=list(PERTURBATION_FILENAME_PREFIX.keys()))
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-generate files even if output already exists.")
    args = ap.parse_args()

    n = build_perturbed_images(
        base_cases_csv=Path(args.base_cases),
        perturbed_dir=Path(args.output_dir),
        methods=args.methods,
        metadata_csv=Path(args.metadata) if args.metadata else None,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
    )
    print(f"Wrote {n} perturbed images to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
