"""Push the VSF-Med v2 bundle to HuggingFace as a gated dataset.

Reads from ./hf_dataset/ (built by scripts/build_hf_dataset.py) and uploads
to ``saillab/vsfmed-v2`` as a private+gated dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "hf_dataset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="saillab/vsfmed-v2",
                    help="HF dataset repo id (org/name)")
    ap.add_argument("--private", action="store_true", default=True,
                    help="Create as private (gated) repo")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be uploaded; do not push")
    args = ap.parse_args()

    if not BUNDLE.exists():
        print(f"ERROR: bundle not found at {BUNDLE}; run build_hf_dataset.py first",
              file=sys.stderr)
        return 2

    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"Bundle contents ({len(files)} files, {total/1024:.1f} KB):")
    for p in files:
        rel = p.relative_to(BUNDLE)
        print(f"  {rel}  ({p.stat().st_size} bytes)")

    if args.dry_run:
        print("\n--dry-run set; not uploading.")
        return 0

    api = HfApi()

    # Create or get repo (gated/private)
    try:
        info = api.repo_info(args.repo_id, repo_type="dataset")
        print(f"\nRepo {args.repo_id} exists; updating.")
    except Exception:
        print(f"\nCreating repo {args.repo_id} (private, dataset)...")
        create_repo(args.repo_id, repo_type="dataset", private=args.private,
                    exist_ok=True)

    # Upload folder
    print(f"\nUploading {BUNDLE} to {args.repo_id}...")
    commit_url = upload_folder(
        folder_path=str(BUNDLE),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="VSF-Med v2 framework release (gated)",
    )
    print(f"\nUploaded.")
    print(f"  Commit URL: {commit_url}")
    print(f"  Dataset page: https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nNext: visit the dataset settings to enable gated access if not already.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
