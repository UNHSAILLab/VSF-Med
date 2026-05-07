"""GMAI-MMBench loader.

GMAI-MMBench is a multi-modality medical VQA benchmark. Native format is a
JSONL file with one VQA item per row. Expected layout:

    <root>/
      gmai_mmbench.jsonl
      images/<modality>/<filename>

JSONL fields used (ignore unknown):
  case_id | id        — unique identifier
  modality            — e.g. "ct", "mri", "ophth", ...
  anatomy             — e.g. "chest", "fundus"
  question            — VQA prompt
  answer              — ground truth answer
  options             — optional multiple-choice (preserved into labels)
  image / image_path  — relative path under images/
  task_type           — dataset-native task category
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import CaseRecord, DatasetLoader


class GmaiMmbenchLoader(DatasetLoader):
    name = "gmai_mmbench"

    def __init__(self, jsonl_filename: str = "gmai_mmbench.jsonl") -> None:
        self.jsonl_filename = jsonl_filename

    def iter_cases(self, root: Path) -> Iterator[CaseRecord]:
        root = Path(root)
        jsonl_path = root / self.jsonl_filename
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"GMAI-MMBench JSONL not found at {jsonl_path}."
            )
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                source_id = str(rec.get("case_id") or rec.get("id"))
                rel_image = rec.get("image_path") or rec.get("image")
                img_path = (root / rel_image) if rel_image else None
                yield CaseRecord(
                    source_id=source_id,
                    image_path=img_path or Path(""),
                    modality=rec.get("modality", "unknown"),
                    anatomy=rec.get("anatomy", "unknown"),
                    view_position=None,
                    labels={"answer": rec.get("answer"),
                            "options": rec.get("options")},
                    report_text=None,
                    demographic_metadata={},
                    license_or_access_notes="GMAI-MMBench per dataset terms.",
                    primary_pathology=rec.get("task_type"),
                    is_normal=None,
                    extra={
                        "question": rec.get("question"),
                        "task_type": rec.get("task_type"),
                    },
                )
