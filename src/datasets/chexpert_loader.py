"""CheXpert v1.0 loader.

Expects:

    <root>/
      train.csv  (or valid.csv / test.csv)
      train/patientNNNNN/studyN/view1_frontal.jpg
      ...

CheXpert CSV uses one row per image with 14 label columns. Same label
vocabulary as MIMIC-CXR's CheXpert columns, so we reuse the mapping.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, List

from .base import CaseRecord, DatasetLoader
from .mimic_cxr_loader import PATHOLOGY_LABEL_MAP


class CheXpertLoader(DatasetLoader):
    name = "chexpert"

    def __init__(self, splits: List[str] = ("train", "valid")) -> None:
        self.splits = tuple(splits)

    def iter_cases(self, root: Path) -> Iterator[CaseRecord]:
        root = Path(root)
        for split in self.splits:
            csv_path = root / f"{split}.csv"
            if not csv_path.exists():
                continue
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rel_path = row["Path"]   # e.g. "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"
                    img_path = root / rel_path
                    labels = _normalize(row)
                    primary, is_normal = _summarize(labels)
                    yield CaseRecord(
                        source_id=rel_path,
                        image_path=img_path,
                        modality="cxr",
                        anatomy="chest",
                        view_position=row.get("Frontal/Lateral"),
                        labels=labels,
                        report_text=None,
                        demographic_metadata={
                            "sex": row.get("Sex"),
                            "age": row.get("Age"),
                        },
                        license_or_access_notes="Stanford CheXpert; non-commercial research.",
                        primary_pathology=primary,
                        is_normal=is_normal,
                    )


def _normalize(row: dict) -> dict:
    out = {}
    for src, dst in PATHOLOGY_LABEL_MAP.items():
        v = row.get(src)
        if v in (None, ""):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if f == 1.0:
            out[dst] = 1
        elif f == 0.0:
            out[dst] = 0
        elif f == -1.0:
            out[dst] = -1
    return out


def _summarize(labels: dict) -> tuple:
    if labels.get("no_finding") == 1:
        return "no_finding", True
    for k, v in labels.items():
        if k != "no_finding" and v == 1:
            return k, False
    return None, None
