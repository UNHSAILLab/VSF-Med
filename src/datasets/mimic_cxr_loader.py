"""MIMIC-CXR-JPG v2.1.0 loader.

Expects the standard PhysioNet directory layout:

    <root>/
      mimic-cxr-2.0.0-metadata.csv     # studies & DICOM ids
      mimic-cxr-2.0.0-chexpert.csv     # CheXpert labels per study
      mimic-cxr-2.0.0-split.csv        # official splits
      files/p10/p10000032/s50414267/<dicom_id>.jpg
      reports/p10/p10000032/s50414267.txt    # optional (gated)

Demographics are joined from MIMIC-IV core if available; otherwise omitted.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

from .base import CaseRecord, DatasetLoader

PATHOLOGY_COLUMNS = (
    "No Finding",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Consolidation",
    "Pneumothorax",
    "Atelectasis",
    "Support Devices",
)

PATHOLOGY_LABEL_MAP = {
    "No Finding": "no_finding",
    "Cardiomegaly": "cardiomegaly",
    "Edema": "edema",
    "Pleural Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia_or_consolidation",
    "Consolidation": "pneumonia_or_consolidation",
    "Pneumothorax": "pneumothorax",
    "Atelectasis": "atelectasis",
    "Support Devices": "support_devices",
}


class MimicCxrLoader(DatasetLoader):
    name = "mimic_cxr_jpg"

    def __init__(self, include_reports: bool = True) -> None:
        self.include_reports = include_reports

    def iter_cases(self, root: Path) -> Iterator[CaseRecord]:
        root = Path(root)
        meta = _read_csv(root / "mimic-cxr-2.0.0-metadata.csv")
        labels = _read_keyed_csv(
            root / "mimic-cxr-2.0.0-chexpert.csv",
            key_cols=("subject_id", "study_id"),
        )

        for row in meta:
            dicom_id = row["dicom_id"]
            subject_id = row["subject_id"]
            study_id = row["study_id"]
            view = row.get("ViewPosition") or None

            img_rel = (
                f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
            )
            img_path = root / img_rel
            study_labels = labels.get((subject_id, study_id), {})
            primary, is_normal = _summarize_chexpert(study_labels)

            report_text: Optional[str] = None
            if self.include_reports:
                report_path = root / f"reports/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt"
                if report_path.exists():
                    try:
                        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        report_text = None

            yield CaseRecord(
                source_id=f"{subject_id}-{study_id}-{dicom_id}",
                image_path=img_path,
                modality="cxr",
                anatomy="chest",
                view_position=view,
                labels=_normalize_labels(study_labels),
                report_text=report_text,
                demographic_metadata={"subject_id": subject_id},  # join MIMIC-IV upstream if needed
                license_or_access_notes="PhysioNet credentialed; do not redistribute.",
                primary_pathology=primary,
                is_normal=is_normal,
            )


def _read_csv(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"MIMIC-CXR metadata not found at {path}. Confirm root and PhysioNet access."
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_keyed_csv(path: Path, key_cols: tuple) -> Dict[tuple, dict]:
    out: Dict[tuple, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row[c] for c in key_cols)
            out[key] = row
    return out


def _normalize_labels(study_labels: dict) -> dict:
    """Map CheXpert raw "1.0"/"0.0"/"-1.0"/""/"NaN" to 0/1/-1."""
    out = {}
    for k, v in study_labels.items():
        if k in ("subject_id", "study_id"):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == 1.0:
            out[PATHOLOGY_LABEL_MAP.get(k, k.lower().replace(" ", "_"))] = 1
        elif f == 0.0:
            out[PATHOLOGY_LABEL_MAP.get(k, k.lower().replace(" ", "_"))] = 0
        elif f == -1.0:
            out[PATHOLOGY_LABEL_MAP.get(k, k.lower().replace(" ", "_"))] = -1
    return out


def _summarize_chexpert(labels: dict) -> tuple:
    """Pick a single primary pathology label and the normal/abnormal flag."""
    if not labels:
        return None, None
    if labels.get("No Finding") == "1.0":
        return "no_finding", True
    for src, dst in PATHOLOGY_LABEL_MAP.items():
        if dst == "no_finding":
            continue
        v = labels.get(src)
        if v == "1.0":
            return dst, False
    return None, False
