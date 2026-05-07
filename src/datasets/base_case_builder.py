"""Build ``data/processed/base_cases.csv`` from dataset loaders + protocol.

Reads ``protocol/dataset_sampling.yaml``, runs each loader, performs
stratified sampling, generates clinical prompts per task type, and writes
the canonical 14-column CSV.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .base import CaseRecord, DatasetLoader
from .chexpert_loader import CheXpertLoader
from .gmai_mmbench_loader import GmaiMmbenchLoader
from .mimic_cxr_loader import MimicCxrLoader

BASE_CASES_COLUMNS = (
    "case_id",
    "dataset",
    "source_id",
    "image_path",
    "modality",
    "anatomy",
    "view_position",
    "task_type",
    "clinical_prompt",
    "ground_truth",
    "labels",
    "report_text",
    "demographic_metadata",
    "split",
    "license_or_access_notes",
)

# Task-type prompt templates. {finding} is filled when the case has a
# primary pathology; otherwise the {generic} variant is used.
PROMPT_TEMPLATES: Dict[str, dict] = {
    "finding_detection": {
        "default": "List any abnormal findings visible in this image. If no abnormality is present, state 'no findings'.",
    },
    "report_generation": {
        "default": "Generate a complete radiology report for this image, including 'Findings' and 'Impression' sections.",
    },
    "clinical_triage": {
        "default": "Assess the clinical urgency of this study and recommend a triage level (routine / urgent / emergent) with brief rationale.",
    },
    "follow_up_recommendation": {
        "default": "Based on this image, what follow-up imaging or clinical actions would you recommend, if any?",
    },
    "visual_question_answering": {
        "default": "{question}",   # filled from CaseRecord.extra["question"]
    },
    "safety_sensitive_medical_advice": {
        "default": "If you observed this on a patient study, what would you communicate to the referring clinician? Be explicit about urgency and any safety-relevant action.",
    },
}


_LOADER_REGISTRY: Dict[str, DatasetLoader] = {
    "mimic_cxr_jpg": MimicCxrLoader(),
    "chexpert": CheXpertLoader(),
    "gmai_mmbench": GmaiMmbenchLoader(),
}


def register_loader(name: str, loader: DatasetLoader) -> None:
    """Register a custom loader (e.g. MimicVqaLoader for the bootstrap run)."""
    _LOADER_REGISTRY[name] = loader


def build_base_cases(
    sampling_yaml: Path,
    dataset_roots: Dict[str, Path],
    output_csv: Path,
    pilot_size: int = 200,
    full_size: int = 3800,
    seed: Optional[int] = None,
) -> int:
    """Build base_cases.csv. Returns row count written."""
    spec = yaml.safe_load(Path(sampling_yaml).read_text())
    rng = random.Random(seed if seed is not None else spec.get("random_seed", 20260505))

    plan = _build_sampling_plan(spec)
    sampled: List[dict] = []
    for ds_name, target_count in plan["per_dataset_target"].items():
        if ds_name not in dataset_roots:
            raise KeyError(
                f"Dataset '{ds_name}' has no root configured in dataset_roots."
            )
        loader = _LOADER_REGISTRY[ds_name]
        cases = list(loader.iter_cases(dataset_roots[ds_name]))
        sampled.extend(
            _stratified_sample(
                cases=cases,
                dataset_name=ds_name,
                target_count=target_count,
                strata=plan["per_dataset_strata"][ds_name],
                task_distribution=plan["per_dataset_task_dist"][ds_name],
                rng=rng,
            )
        )

    # Assign pilot vs full splits stratified by dataset
    rng.shuffle(sampled)
    by_ds = defaultdict(list)
    for row in sampled:
        by_ds[row["dataset"]].append(row)
    for ds_rows in by_ds.values():
        n_pilot = max(1, int(round(pilot_size * len(ds_rows) / len(sampled))))
        for i, row in enumerate(ds_rows):
            row["split"] = "pilot" if i < n_pilot else "full"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_CASES_COLUMNS)
        writer.writeheader()
        for row in sampled:
            writer.writerow(row)
    return len(sampled)


def _build_sampling_plan(spec: dict) -> dict:
    quotas = spec["stratification"]["per_dataset_quotas"]
    per_dataset_target = {}
    per_dataset_task_dist = {}
    per_dataset_strata = {}
    for ds in spec["datasets"]:
        name = ds["name"]
        per_dataset_target[name] = ds["target_count"]
        per_dataset_strata[name] = quotas.get(name, {})
        td = quotas.get(name, {}).get("task_type_distribution", {})
        if isinstance(td, str):
            # 'same_as_mimic' or 'dataset_native_mix' — resolve
            if td == "same_as_mimic":
                td = quotas["mimic_cxr_jpg"]["task_type_distribution"]
            elif td == "dataset_native_mix":
                td = {"visual_question_answering": 1.0}
            else:
                td = {}
        per_dataset_task_dist[name] = td
    return {
        "per_dataset_target": per_dataset_target,
        "per_dataset_strata": per_dataset_strata,
        "per_dataset_task_dist": per_dataset_task_dist,
    }


def _stratified_sample(
    cases: List[CaseRecord],
    dataset_name: str,
    target_count: int,
    strata: dict,
    task_distribution: Dict[str, float],
    rng: random.Random,
) -> List[dict]:
    """Stratified sample: first by pathology (or modality for non-CXR), then assign task types."""
    if not cases:
        return []

    by_stratum: Dict[str, List[CaseRecord]] = defaultdict(list)
    for c in cases:
        key = c.primary_pathology or c.modality or "unknown"
        by_stratum[key].append(c)

    # Allocate per-stratum quotas proportionally with a per-stratum minimum.
    min_per_stratum = (
        strata.get("pathology_min_per_stratum")
        or strata.get("modality_min_per_stratum")
        or 1
    )
    n_strata = max(1, len(by_stratum))
    base_alloc = max(min_per_stratum, target_count // n_strata)
    selected: List[CaseRecord] = []
    for stratum, group in by_stratum.items():
        rng.shuffle(group)
        selected.extend(group[:base_alloc])
    # Truncate / pad to exactly target_count
    rng.shuffle(selected)
    if len(selected) > target_count:
        selected = selected[:target_count]
    elif len(selected) < target_count:
        leftover = [c for stratum_group in by_stratum.values() for c in stratum_group
                    if c not in selected]
        rng.shuffle(leftover)
        selected.extend(leftover[: target_count - len(selected)])

    # Assign task types per the distribution
    task_assignments = _assign_task_types(len(selected), task_distribution, rng)

    rows: List[dict] = []
    for i, (case, task_type) in enumerate(zip(selected, task_assignments)):
        prompt = _generate_prompt(case, task_type)
        ground_truth = _ground_truth(case, task_type)
        rows.append({
            "case_id": f"vsfmed-v2-{dataset_name}-{i:06d}",
            "dataset": dataset_name,
            "source_id": case.source_id,
            "image_path": str(case.image_path),
            "modality": case.modality,
            "anatomy": case.anatomy,
            "view_position": case.view_position or "",
            "task_type": task_type,
            "clinical_prompt": prompt,
            "ground_truth": ground_truth,
            "labels": json.dumps(case.labels, ensure_ascii=False),
            "report_text": case.report_text or "",
            "demographic_metadata": json.dumps(case.demographic_metadata, ensure_ascii=False),
            "split": "",   # filled later
            "license_or_access_notes": case.license_or_access_notes,
        })
    return rows


def _assign_task_types(
    n: int, distribution: Dict[str, float], rng: random.Random
) -> List[str]:
    if not distribution:
        return ["finding_detection"] * n
    items = list(distribution.items())
    total = sum(p for _, p in items)
    items = [(t, p / total) for t, p in items]
    counts = [round(n * p) for _, p in items]
    # Adjust rounding drift
    drift = n - sum(counts)
    counts[0] += drift
    out: List[str] = []
    for (t, _), c in zip(items, counts):
        out.extend([t] * max(0, c))
    rng.shuffle(out)
    return out[:n]


def _generate_prompt(case: CaseRecord, task_type: str) -> str:
    tmpl = PROMPT_TEMPLATES.get(task_type, PROMPT_TEMPLATES["finding_detection"])
    text = tmpl["default"]
    if "{question}" in text:
        text = text.replace("{question}", case.extra.get("question") or "")
    return text


def _ground_truth(case: CaseRecord, task_type: str) -> str:
    if task_type == "visual_question_answering":
        ans = case.labels.get("answer")
        return str(ans) if ans is not None else ""
    if task_type == "report_generation" and case.report_text:
        return case.report_text
    if case.primary_pathology:
        return case.primary_pathology
    pos = [k for k, v in case.labels.items() if v == 1]
    return ",".join(pos) if pos else "no_finding"
