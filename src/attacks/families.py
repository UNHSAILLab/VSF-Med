"""Loaders for VSF-Med attack family protocol artifacts.

Materializes ``protocol/attack_families.yaml`` + ``templates/text_attack_templates.txt``
into in-memory specs that ``eval_case_builder.build_eval_cases`` can consume.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclasses.dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    label: str
    is_attack: bool
    multi_turn: bool
    text_variants: tuple
    visual_variants: tuple
    risk_dimension_targets: tuple
    visual_quality_constraint: Optional[dict] = None


@dataclasses.dataclass(frozen=True)
class AttackFamiliesProtocol:
    protocol_version: str
    conditions: Dict[str, ConditionSpec]
    cases_per_condition: int
    total_eval_rows: int
    adaptive_attack_targets: tuple
    quality_gates: dict


def load_attack_families(yaml_path: Path) -> AttackFamiliesProtocol:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    conditions = {}
    for c in raw["conditions"]:
        spec = ConditionSpec(
            condition_id=c["condition_id"],
            label=c["label"],
            is_attack=c["is_attack"],
            multi_turn=c.get("multi_turn", False),
            text_variants=tuple(c["text_variants"]),
            visual_variants=tuple(c["visual_variants"]),
            risk_dimension_targets=tuple(c.get("risk_dimension_targets", [])),
            visual_quality_constraint=c.get("visual_quality_constraint"),
        )
        conditions[spec.condition_id] = spec
    return AttackFamiliesProtocol(
        protocol_version=raw["protocol_version"],
        conditions=conditions,
        cases_per_condition=raw["allocation"]["cases_per_condition"],
        total_eval_rows=raw["allocation"]["total_eval_rows"],
        adaptive_attack_targets=tuple(raw["adaptive_attack_targets"]),
        quality_gates=raw.get("quality_gates", {}),
    )


def load_text_templates(template_file_path: Path) -> Dict[str, List[str]]:
    """Parse the ``ATTACK_TEMPLATES = {...}`` dict from the v1 .txt file.

    Uses ``ast`` rather than ``exec`` so the file content is not executed.
    """
    src = Path(template_file_path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "ATTACK_TEMPLATES":
                    return ast.literal_eval(node.value)
    raise ValueError(
        f"ATTACK_TEMPLATES dict not found in {template_file_path}"
    )


def template_hash(category: str, template_text: str) -> str:
    payload = f"{category}␟{template_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
