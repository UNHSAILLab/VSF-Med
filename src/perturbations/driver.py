"""Driver that materializes perturbed images for VSF-Med v2 conditions C7/C8.

For each base case in ``base_cases.csv``, generates the 8 perturbed variants
referenced in the eval case builder, saved with the naming convention
``{prefix}_{base_name}.{ext}`` so eval_case_builder._perturbed_image_path
finds them.

Also computes SSIM and PSNR for quality control and writes
``perturbed_metadata.csv`` with one row per (case_id, method) pair.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from .methods import PERTURBATION_METHODS, apply_perturbation

# Filename prefix per perturbation method. Must match
# src.attacks.eval_case_builder._perturbed_image_path's ``suffix_map``.
PERTURBATION_FILENAME_PREFIX = {
    "gaussian_noise":              "pert_gaussian",
    "checkerboard_overlay":        "pert_checker",
    "random_arrow":                "pert_arrow",
    "moire_overlay":               "pert_moire",
    "steganographic_hide":         "pert_steg_true",
    "lsb_extract_visible":         "lsb_true",
    "embedded_visual_text_prompt": "pert_visual_text",
    "artifact_plus_overlay":       "pert_combined",
}

# Methods that swap to PNG output (required for stego / LSB extract clarity).
_PNG_METHODS = {"lsb_extract_visible"}


def _ssim_psnr(orig: Image.Image, perturbed: Image.Image) -> tuple:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        # scikit-image is optional QA tooling; absence is not a hard failure
        return float("nan"), float("nan")

    o = np.array(orig.convert("L"))
    p = np.array(perturbed.convert("L"))
    if o.shape != p.shape:
        # Resize for comparison only — does not affect saved file
        p = np.array(perturbed.convert("L").resize(orig.size))
    try:
        ssim = float(structural_similarity(o, p, data_range=255))
    except Exception:  # noqa: BLE001
        ssim = float("nan")
    try:
        psnr = float(peak_signal_noise_ratio(o, p, data_range=255))
    except Exception:  # noqa: BLE001
        psnr = float("nan")
    return ssim, psnr


def build_perturbed_images(
    base_cases_csv: Path,
    perturbed_dir: Path,
    methods: Optional[Iterable[str]] = None,
    metadata_csv: Optional[Path] = None,
    seed: int = 20260505,
    skip_existing: bool = True,
) -> int:
    """Generate perturbed-image variants and metadata. Returns files written."""
    perturbed_dir = Path(perturbed_dir)
    perturbed_dir.mkdir(parents=True, exist_ok=True)
    methods_list: List[str] = list(methods) if methods else list(PERTURBATION_METHODS.keys())
    for m in methods_list:
        if m not in PERTURBATION_METHODS:
            raise KeyError(f"Unknown perturbation method: {m}")

    metadata_rows: List[dict] = []
    files_written = 0

    with Path(base_cases_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            base_name = image_path.stem
            try:
                src = Image.open(image_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                metadata_rows.append({
                    "case_id": case_id, "method": "load",
                    "perturbed_path": "", "ssim": "", "psnr": "",
                    "error": repr(exc),
                })
                continue

            rng = np.random.default_rng(seed + abs(hash(case_id)) % (2**31))

            for method in methods_list:
                prefix = PERTURBATION_FILENAME_PREFIX[method]
                ext = ".png" if method in _PNG_METHODS else image_path.suffix or ".jpg"
                out_path = perturbed_dir / f"{prefix}_{base_name}{ext}"
                if skip_existing and out_path.exists():
                    metadata_rows.append({
                        "case_id": case_id, "method": method,
                        "perturbed_path": str(out_path),
                        "ssim": "", "psnr": "", "error": "skipped_existing",
                    })
                    continue
                try:
                    perturbed = apply_perturbation(method, src, rng=rng)
                    perturbed.save(out_path)
                    ssim, psnr = _ssim_psnr(src, perturbed)
                    metadata_rows.append({
                        "case_id": case_id, "method": method,
                        "perturbed_path": str(out_path),
                        "ssim": f"{ssim:.4f}", "psnr": f"{psnr:.2f}",
                        "error": "",
                    })
                    files_written += 1
                except Exception as exc:  # noqa: BLE001
                    metadata_rows.append({
                        "case_id": case_id, "method": method,
                        "perturbed_path": "",
                        "ssim": "", "psnr": "", "error": repr(exc),
                    })

    if metadata_csv is not None:
        metadata_csv = Path(metadata_csv)
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        with metadata_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["case_id", "method", "perturbed_path", "ssim", "psnr", "error"],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)

    return files_written
