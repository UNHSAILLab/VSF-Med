"""MIMIC-Diff-VQA loader (and similar MIMIC-CXR VQA-style datasets).

Reads JSON files of the form produced by Hu et al.'s Medical-CXR-VQA work:

    [{
       "subject_id": "...", "study_id": "...", "image_id": "...",
       "image_path": "p17/p17945608/s55914880/<image_id>.jpg",
       "question": "...",
       "semantic_type": "verify"|"query"|"choose",
       "content_type": "presence"|"abnormality"|"anatomy"|"attribute"|"size"|"plane"|"gender",
       "template_arguments": {...},
       "answer": ["yes"|"no"|"<finding>"|...]
    }, ...]

Only yields cases whose image_id is present in one of the configured
``image_dirs`` (flat directories of ``<image_id>.jpg`` files). Each Q/A pair
becomes one ``CaseRecord`` with ``task_type=visual_question_answering``
recorded in ``extra``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

from .base import CaseRecord, DatasetLoader


# Map MIMIC-VQA content_type → VSF-Med stratification labels
CONTENT_TYPE_TO_STRATUM = {
    "presence": "vqa_presence",
    "abnormality": "vqa_abnormality",
    "anatomy": "vqa_anatomy",
    "attribute": "vqa_attribute",
    "size": "vqa_size",
    "plane": "vqa_plane",
    "gender": "vqa_gender",
}


class MimicVqaLoader(DatasetLoader):
    """Loader for MIMIC-CXR VQA-style JSON splits with a flat local image cache."""

    name = "mimic_cxr_vqa"

    def __init__(
        self,
        image_dirs: List[Path],
        split_files: Optional[List[str]] = None,
        max_per_image: Optional[int] = None,
    ) -> None:
        self.image_dirs = [Path(d) for d in image_dirs]
        self.split_files = split_files or ["train.json", "valid.json", "test.json"]
        self.max_per_image = max_per_image  # None = keep all Q/A pairs per image
        self._image_index: Optional[dict] = None  # image_id → resolved Path

    def _build_image_index(self) -> dict:
        """Index every ``<image_id>.jpg`` across configured image_dirs."""
        idx: dict = {}
        for d in self.image_dirs:
            if not d.exists():
                continue
            for p in d.glob("*.jpg"):
                idx.setdefault(p.stem, p)  # first match wins
        return idx

    def iter_cases(self, root: Path) -> Iterator[CaseRecord]:
        if self._image_index is None:
            self._image_index = self._build_image_index()
        if not self._image_index:
            raise FileNotFoundError(
                f"No JPGs found under image_dirs={self.image_dirs}"
            )

        per_image_count: dict = {}
        for split_file in self.split_files:
            split_path = Path(root) / split_file
            if not split_path.exists():
                continue
            with split_path.open(encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                image_id = rec.get("image_id")
                local_path = self._image_index.get(image_id)
                if local_path is None:
                    continue
                if (self.max_per_image is not None
                        and per_image_count.get(image_id, 0) >= self.max_per_image):
                    continue
                per_image_count[image_id] = per_image_count.get(image_id, 0) + 1

                yield self._record_to_case(rec, local_path)

    def _record_to_case(self, rec: dict, local_path: Path) -> CaseRecord:
        content_type = rec.get("content_type", "unknown")
        answer_list = rec.get("answer") or []
        answer_str = ", ".join(str(a) for a in answer_list) if isinstance(answer_list, list) else str(answer_list)

        primary, is_normal = _summarize_pathology(rec, answer_list)

        labels = {
            "answer": answer_str,
            "content_type": content_type,
            "semantic_type": rec.get("semantic_type"),
            "template_program": rec.get("template_program"),
        }

        return CaseRecord(
            source_id=f"{rec['subject_id']}-{rec['study_id']}-{rec['image_id']}-{rec.get('idx', 0)}",
            image_path=local_path,
            modality="cxr",
            anatomy="chest",
            view_position=None,
            labels=labels,
            report_text=None,
            demographic_metadata={"subject_id": rec.get("subject_id")},
            license_or_access_notes="MIMIC-CXR PhysioNet credentialed; VQA annotations under source dataset license.",
            primary_pathology=primary,
            is_normal=is_normal,
            extra={
                "question": rec.get("question"),
                "task_type": "visual_question_answering",
                "vqa_split": rec.get("split"),
                "content_stratum": CONTENT_TYPE_TO_STRATUM.get(content_type, content_type),
                "template_arguments": rec.get("template_arguments"),
            },
        )


def _summarize_pathology(rec: dict, answer_list) -> tuple:
    """Best-effort pathology + normal/abnormal summary from a VQA record."""
    content = rec.get("content_type")
    answers = [str(a).strip().lower() for a in (answer_list or [])]

    if content == "presence":
        # template_arguments.category typically holds the finding category
        cats = rec.get("template_arguments", {}).get("category", {}) or {}
        if isinstance(cats, dict):
            cat_vals = list(cats.values())
        else:
            cat_vals = list(cats)
        cat = (cat_vals[0] if cat_vals else None)
        if answers and answers[0] in ("no", "none", "nothing"):
            return cat or None, True
        return cat or None, False

    if content == "abnormality":
        if not answers:
            return None, None
        if answers[0] in ("no", "none", "nothing"):
            return "no_finding", True
        return answers[0], False

    return None, None
