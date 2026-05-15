"""Ingest a clinician-exported JSON back into vsfmed_v2.annotations.

Usage:
    python scripts/import_clinician_json.py path/to/vsfmed_annotations__alice__2026-05-09.json

The JSON must have the shape produced by the static annotation app:
{
    "annotator": {"name": "...", "role": "...", "started_at": "..."},
    "annotations": [{...}, ...],
    "exported_at": "..."
}

Idempotent: running twice just upserts the latest revision per (sample_id, annotator_id).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def upsert_annotator(conn, annotator: dict) -> int:
    name = (annotator.get("name") or "").strip()
    role = (annotator.get("role") or "").strip().lower()      # CHECK constraint is lowercase
    if not name or role not in ("radiologist", "clinician", "adjudicator"):
        raise ValueError(f"Invalid annotator name/role: {name!r} / {role!r}")
    existing = conn.execute(sa.text(
        "SELECT annotator_id FROM vsfmed_v2.annotators WHERE name=:n AND role=:r"
    ), {"n": name, "r": role}).scalar()
    if existing:
        return int(existing)
    new_id = conn.execute(sa.text(
        "INSERT INTO vsfmed_v2.annotators (name, role) VALUES (:n, :r) RETURNING annotator_id"
    ), {"n": name, "r": role}).scalar()
    print(f"  registered new annotator: {name} ({role}) → id={new_id}")
    return int(new_id)


def main(argv: list) -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2

    if len(argv) < 2:
        print(f"Usage: {argv[0]} <path-to-clinician-json>", file=sys.stderr)
        return 2

    payload_path = Path(argv[1])
    payload = json.loads(payload_path.read_text())
    annotator = payload.get("annotator", {})
    annotations = payload.get("annotations", []) or []
    print(f"Loading {len(annotations)} annotations from {payload_path.name}")

    engine = sa.create_engine(url)
    md = sa.MetaData(schema="vsfmed_v2")
    md.reflect(bind=engine, only=["annotations"])
    annotations_t = md.tables["vsfmed_v2.annotations"]

    inserted = 0
    skipped = 0
    with engine.begin() as conn:
        annotator_id = upsert_annotator(conn, annotator)

        existing = conn.execute(sa.text(
            "SELECT sample_id, MAX(revision) FROM vsfmed_v2.annotations "
            "WHERE annotator_id=:aid GROUP BY sample_id"
        ), {"aid": annotator_id}).all()
        max_rev = {sid: rev for sid, rev in existing}

        rows = []
        for a in annotations:
            sid = int(a.get("sample_id"))
            rev = (max_rev.get(sid, 0) or 0) + 1
            harm = a.get("clinical_harm_0_4")
            if harm is None:
                skipped += 1
                continue
            rows.append({
                "sample_id": sid,
                "annotator_id": annotator_id,
                "revision": rev,
                "clinical_harm_0_4": int(harm),
                "diagnostic_correctness_0_4": int(a["diagnostic_correctness_0_4"]),
                "unsafe_recommendation": bool(a["unsafe_recommendation"]),
                "missed_critical_finding": bool(a["missed_critical_finding"]),
                "over_refusal": bool(a["over_refusal"]),
                "attack_changed_meaning": (None if a.get("attack_changed_meaning") is None
                                           else bool(a["attack_changed_meaning"])),
                "confidence_in_label_0_4": int(a["confidence_in_label_0_4"]),
                "free_text_notes": a.get("free_text_notes") or None,
                "flagged_for_second_look": bool(a.get("flagged_for_second_look", False)),
                "submitted_at": a.get("submitted_at"),
            })

        for i in range(0, len(rows), 200):
            batch = rows[i:i+200]
            stmt = pg_insert(annotations_t).values(batch).on_conflict_do_nothing(
                index_elements=["sample_id", "annotator_id", "revision"]
            )
            conn.execute(stmt)
            inserted += len(batch)

    print(f"Upserted: {inserted}   Skipped (no harm rating): {skipped}")
    with engine.connect() as conn:
        n = conn.execute(sa.text(
            "SELECT COUNT(*) FROM vsfmed_v2.annotations WHERE annotator_id=:aid"
        ), {"aid": annotator_id}).scalar()
        print(f"vsfmed_v2.annotations now has {n} rows for annotator_id={annotator_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
