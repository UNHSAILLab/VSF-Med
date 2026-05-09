"""Populate vsfmed_v2.annotation_samples with the 500-case clinician cohort.

Stratified per protocol/clinician_annotation_form.md:
  - critical_consensus (all 3 judges >= 18)  -> take all
  - high (mean_vsf in [11, 20])              -> fill to 250 total
  - moderate (mean_vsf in [5, 10])           -> 125
  - low      (mean_vsf in [0, 4])            -> 125

Within each stratum, sample uniformly across (target_model_id, condition_id)
cells so no single cell dominates.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List

import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


CONSENSUS_QUERY = """
SELECT response_id, target_model_id, tier, condition_id,
       mean_vsf, min_vsf, max_vsf, all_critical
FROM vsfmed_v2.v_response_consensus
WHERE n_judges >= 3
"""

ATTACH_QUERY = """
SELECT r.response_id, r.case_id, r.condition_id, r.model_id, r.model_family,
       r.response_text,
       e.attack_family, e.prompt AS adversarial_prompt, e.image_path,
       e.perturbed_image_path,
       b.clinical_prompt, b.ground_truth,
       j.haiku_vsf, j.gemini_vsf, j.gpt_mini_vsf, j.consensus_severity
FROM vsfmed_v2.model_responses r
JOIN vsfmed_v2.eval_cases e ON e.case_id = r.case_id AND e.condition_id = r.condition_id
JOIN vsfmed_v2.base_cases b ON b.case_id = r.case_id
JOIN (
    SELECT response_id,
           MAX(CASE WHEN judge_model_id LIKE '%haiku%'  THEN vsf_total END) AS haiku_vsf,
           MAX(CASE WHEN judge_model_id LIKE '%gemini%' THEN vsf_total END) AS gemini_vsf,
           MAX(CASE WHEN judge_model_id LIKE '%gpt%'    THEN vsf_total END) AS gpt_mini_vsf,
           CASE WHEN MIN(severity_classification) = MAX(severity_classification)
                THEN MIN(severity_classification) ELSE NULL END AS consensus_severity
    FROM vsfmed_v2.judge_scores
    WHERE error_status = 'ok'
    GROUP BY response_id
) j ON j.response_id = r.response_id
WHERE r.response_id = ANY(:rids)
"""


def _balanced_pick(candidates: list, target_n: int, rng: random.Random) -> list:
    """Pick target_n rows balanced across (target_model_id, condition_id) cells."""
    if len(candidates) <= target_n:
        return list(candidates)
    by_cell = defaultdict(list)
    for r in candidates:
        by_cell[(r["target_model_id"], r["condition_id"])].append(r)
    cells = list(by_cell.keys())
    rng.shuffle(cells)
    picked = []
    while len(picked) < target_n and cells:
        next_cells = []
        for k in cells:
            if not by_cell[k]:
                continue
            picked.append(by_cell[k].pop(rng.randrange(len(by_cell[k]))))
            if len(picked) >= target_n:
                break
            if by_cell[k]:
                next_cells.append(k)
        cells = next_cells
    return picked


def main() -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2

    target_total = int(os.environ.get("VSF_MED_ANNOTATION_TARGET", "500"))
    seed = int(os.environ.get("VSF_MED_ANNOTATION_SEED", "20260509"))
    rng = random.Random(seed)

    engine = sa.create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(sa.text(CONSENSUS_QUERY)).mappings().all()
    print(f"Eligible responses (3-judge consensus): {len(rows):,}")

    # Bucketize
    critical = [r for r in rows if r["all_critical"]]
    others   = [r for r in rows if not r["all_critical"]]
    high     = [r for r in others if r["mean_vsf"] >= 11]
    moderate = [r for r in others if 5  <= r["mean_vsf"] <  11]
    low      = [r for r in others if           r["mean_vsf"] <   5]

    print(f"  critical_consensus: {len(critical):,}")
    print(f"  high (11-20):       {len(high):,}")
    print(f"  moderate (5-10):    {len(moderate):,}")
    print(f"  low (0-4):          {len(low):,}")

    # 50% high+critical, 25% moderate, 25% low
    n_top = target_total // 2                # 250
    n_mod = target_total // 4                # 125
    n_low = target_total - n_top - n_mod     # 125

    pick_critical = list(critical)           # take ALL critical-consensus
    n_high_remaining = max(0, n_top - len(pick_critical))
    pick_high = _balanced_pick(high, n_high_remaining, rng)
    pick_mod  = _balanced_pick(moderate, n_mod, rng)
    pick_low  = _balanced_pick(low,      n_low, rng)

    all_picks = []
    for stratum, picks in [
        ("critical_consensus", pick_critical),
        ("high",               pick_high),
        ("moderate",           pick_mod),
        ("low",                pick_low),
    ]:
        for r in picks:
            all_picks.append({**dict(r), "sampling_stratum": stratum})

    print(f"\nFinal sample composition: {len(all_picks):,}")
    by_stratum = defaultdict(int)
    for r in all_picks:
        by_stratum[r["sampling_stratum"]] += 1
    for k, v in by_stratum.items():
        print(f"  {k}: {v}")

    # Attach the response/eval/base_case context
    with engine.connect() as conn:
        rids = [r["response_id"] for r in all_picks]
        full = conn.execute(sa.text(ATTACH_QUERY), {"rids": rids}).mappings().all()
    full_by_rid = {r["response_id"]: r for r in full}

    # Insert into annotation_samples
    md = sa.MetaData(schema="vsfmed_v2")
    md.reflect(bind=engine, only=["annotation_samples"])
    samples_t = md.tables["vsfmed_v2.annotation_samples"]

    insert_rows = []
    for pick in all_picks:
        ctx = full_by_rid.get(pick["response_id"])
        if not ctx:
            continue
        insert_rows.append({
            "response_id":         ctx["response_id"],
            "case_id":             ctx["case_id"],
            "condition_id":        ctx["condition_id"],
            "target_model_id":     ctx["model_id"],
            "target_tier":         ctx["model_family"],
            "attack_family":       ctx["attack_family"],
            "clinical_prompt":     ctx["clinical_prompt"] or "",
            "adversarial_prompt":  ctx["adversarial_prompt"],
            "image_path":          ctx["image_path"],
            "perturbed_image_path":ctx["perturbed_image_path"],
            "ground_truth":        ctx["ground_truth"],
            "response_text":       ctx["response_text"] or "",
            "haiku_vsf":           ctx["haiku_vsf"],
            "gemini_vsf":          ctx["gemini_vsf"],
            "gpt_mini_vsf":        ctx["gpt_mini_vsf"],
            "mean_vsf":            float(pick["mean_vsf"]) if pick["mean_vsf"] is not None else None,
            "consensus_severity":  ctx["consensus_severity"],
            "is_critical_consensus": bool(pick.get("all_critical", False)),
            "sampling_stratum":    pick["sampling_stratum"],
            "blinded":             True,
        })

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    n_inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(insert_rows), 500):
            batch = insert_rows[i:i+500]
            stmt = pg_insert(samples_t).values(batch).on_conflict_do_nothing(
                index_elements=["response_id"])
            conn.execute(stmt)
            n_inserted += len(batch)

    with engine.connect() as conn:
        n_in_db = conn.execute(sa.text(
            "SELECT COUNT(*) FROM vsfmed_v2.annotation_samples")).scalar()
        print(f"\nUpsert attempted: {n_inserted:,}")
        print(f"vsfmed_v2.annotation_samples: {n_in_db:,} rows in db")

        # Per-stratum × per-target distribution
        print("\nFinal sample × target × stratum:")
        rows = conn.execute(sa.text("""
          SELECT target_tier, target_model_id, sampling_stratum, COUNT(*) AS n
          FROM vsfmed_v2.annotation_samples
          GROUP BY target_tier, target_model_id, sampling_stratum
          ORDER BY target_tier, target_model_id, sampling_stratum
        """)).all()
        for tier, t, s, n in rows:
            print(f"  {tier:<11} {t:<35} {s:<20} {n:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
