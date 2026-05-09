"""Load the pilot data into vsfmed_v2.* tables on Neon.

Idempotent: every insert uses ON CONFLICT DO UPDATE so re-running is safe.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

ROOT = Path(__file__).resolve().parents[1]

# Map model_id → tier classification (must match what we put in DB)
TIER = {
    "chexone_4b_instruct":  "specialist",
    "chexone_4b_reasoning": "specialist",
    "medgemma_4b_it":       "specialist",
    "medgemma_27b_it":      "specialist",
    "target_claude-haiku-4-5-20251001": "frontier",
    "target_gpt-5.4-mini-2026-03-17":   "frontier",
    "target_gemini-3-flash-preview":    "frontier",
}


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _read_jsonl(p: Path) -> Iterator[dict]:
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _upsert(conn, table: sa.Table, rows: list, conflict_cols: list,
            update_cols: list = None, batch_size: int = 1000) -> int:
    if not rows:
        return 0
    n = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        stmt = pg_insert(table).values(batch)
        if update_cols:
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_={c: stmt.excluded[c] for c in update_cols},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        conn.execute(stmt)
        n += len(batch)
    return n


def load_base_cases(conn, table: sa.Table) -> int:
    csv_path = ROOT / "data/processed/base_cases.csv"
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "case_id": r["case_id"],
                "dataset": r["dataset"],
                "source_id": r["source_id"] or None,
                "image_path": r["image_path"],
                "modality": r["modality"] or None,
                "anatomy": r["anatomy"] or None,
                "view_position": r["view_position"] or None,
                "task_type": r["task_type"] or None,
                "clinical_prompt": r["clinical_prompt"] or None,
                "ground_truth": r["ground_truth"] or None,
                "labels": json.loads(r["labels"]) if r["labels"] else None,
                "report_text": r["report_text"] or None,
                "demographic_metadata": json.loads(r["demographic_metadata"]) if r["demographic_metadata"] else None,
                "split": r["split"] or None,
                "license_or_access_notes": r["license_or_access_notes"] or None,
            })
    update_cols = [c for c in rows[0].keys() if c != "case_id"] if rows else []
    return _upsert(conn, table, rows, ["case_id"], update_cols)


def load_eval_cases(conn, table: sa.Table) -> int:
    jsonl = ROOT / "data/processed/eval_cases.jsonl"
    rows = []
    for d in _read_jsonl(jsonl):
        rows.append({
            "case_id": d["case_id"],
            "condition_id": d["condition_id"],
            "attack_family": d.get("attack_family"),
            "attack_variant": d.get("attack_variant"),
            "prompt": d["prompt"],
            "image_path": d["image_path"],
            "perturbed_image_path": d.get("perturbed_image_path"),
            "expected_safe_behavior": d.get("expected_safe_behavior"),
            "risk_dimension_targets": d.get("risk_dimension_targets"),
            "template_id": d.get("template_id"),
            "template_hash": d.get("template_hash"),
        })
    update_cols = [c for c in rows[0].keys() if c not in ("case_id","condition_id")] if rows else []
    return _upsert(conn, table, rows, ["case_id","condition_id"], update_cols)


def load_responses(conn, table: sa.Table) -> int:
    glob = (ROOT / "data/results").glob("pilot.*.jsonl")
    rows = []
    seen = set()
    for p in glob:
        for d in _read_jsonl(p):
            mid = d["model_id"]
            rid = f"{mid}|{d['case_id']}|{d['condition_id']}"
            if rid in seen:
                continue
            seen.add(rid)
            rows.append({
                "response_id": rid,
                "case_id": d["case_id"],
                "condition_id": d["condition_id"],
                "model_id": mid,
                "model_provider": d.get("model_provider"),
                "model_snapshot": d.get("model_snapshot"),
                "model_family": TIER.get(mid, d.get("model_family")),
                "response_text": d.get("response_text"),
                "error_status": d.get("error_status"),
                "latency_seconds": d.get("latency_seconds"),
                "input_token_count": d.get("input_token_count"),
                "output_token_count": d.get("output_token_count"),
                "timestamp": d.get("timestamp"),
            })
    update_cols = [c for c in rows[0].keys() if c != "response_id"] if rows else []
    return _upsert(conn, table, rows, ["response_id"], update_cols, batch_size=500)


def load_judge_scores(conn, table: sa.Table) -> int:
    paths = sorted((ROOT / "data/results").glob("pilot_judge_scores.*.jsonl"))
    paths = [p for p in paths
             if "combined" not in p.name and ".reliability" not in p.name]
    rows = []
    for p in paths:
        for d in _read_jsonl(p):
            rid = d.get("response_id") or f'{d["target_model_id"]}|{d["case_id"]}|{d["condition_id"]}'
            rows.append({
                "response_id": rid,
                "case_id": d["case_id"],
                "condition_id": d["condition_id"],
                "target_model_id": d["target_model_id"],
                "judge_model_id": d["judge_model_id"],
                "judge_provider": d.get("judge_provider"),
                "judge_snapshot": d.get("judge_snapshot"),
                "score_prompt_injection": d.get("score_prompt_injection"),
                "score_jailbreak": d.get("score_jailbreak"),
                "score_confidentiality": d.get("score_confidentiality"),
                "score_misinformation": d.get("score_misinformation"),
                "score_dos_or_refusal": d.get("score_dos_or_refusal"),
                "score_persistence": d.get("score_persistence"),
                "score_safety_bypass": d.get("score_safety_bypass"),
                "score_clinical_decision_impact": d.get("score_clinical_decision_impact"),
                "score_diagnostic_correctness": d.get("score_diagnostic_correctness"),
                "score_appropriate_refusal": d.get("score_appropriate_refusal"),
                "vsf_total": d.get("vsf_total"),
                "severity_classification": d.get("severity_classification"),
                "short_rationale": d.get("short_rationale"),
                "judge_timestamp": d.get("judge_timestamp"),
                "judge_latency_seconds": d.get("judge_latency_seconds"),
                "error_status": d.get("error_status"),
            })
    update_cols = [c for c in rows[0].keys()
                   if c not in ("response_id", "judge_model_id")] if rows else []
    return _upsert(conn, table, rows,
                   ["response_id","judge_model_id"], update_cols, batch_size=500)


def main() -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2

    engine = sa.create_engine(url)
    md = sa.MetaData(schema="vsfmed_v2")
    md.reflect(bind=engine, only=["base_cases","eval_cases","model_responses","judge_scores"])
    base_t  = md.tables["vsfmed_v2.base_cases"]
    eval_t  = md.tables["vsfmed_v2.eval_cases"]
    resp_t  = md.tables["vsfmed_v2.model_responses"]
    judge_t = md.tables["vsfmed_v2.judge_scores"]

    with engine.begin() as conn:
        n_base = load_base_cases(conn, base_t)
        print(f"  base_cases: {n_base} upserted")
        n_eval = load_eval_cases(conn, eval_t)
        print(f"  eval_cases: {n_eval} upserted")
        n_resp = load_responses(conn, resp_t)
        print(f"  model_responses: {n_resp} upserted")
        n_jud = load_judge_scores(conn, judge_t)
        print(f"  judge_scores: {n_jud} upserted")

    # Verify counts
    with engine.connect() as c:
        for t in ("base_cases","eval_cases","model_responses","judge_scores"):
            n = c.execute(sa.text(f"SELECT COUNT(*) FROM vsfmed_v2.{t}")).scalar()
            print(f"  vsfmed_v2.{t}: {n:,} rows total in db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
