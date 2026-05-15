"""Generate the two v2 figures referenced by the paper.

  fig1_per_target_severity.pdf  — stacked-bar severity-tier distribution per target
  fig2_calibration.pdf          — VSF severity tier → clinician harm-rate calibration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "VSF_Med_IEEE/images/v2"
OUT.mkdir(parents=True, exist_ok=True)


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


TARGET_LABEL = {
    "target_claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "target_gpt-5.4-mini-2026-03-17":   "GPT-5.4-mini",
    "target_gemini-3-flash-preview":    "Gemini 3 Flash",
    "chexone_4b_instruct":              "CheXOne (instruct)",
    "chexone_4b_reasoning":             "CheXOne (reasoning)",
    "medgemma_4b_it":                   "MedGemma 4B",
    "medgemma_27b_it":                  "MedGemma 27B",
}
TIER_OF = {
    "target_claude-haiku-4-5-20251001": "Frontier",
    "target_gpt-5.4-mini-2026-03-17":   "Frontier",
    "target_gemini-3-flash-preview":    "Frontier",
    "chexone_4b_instruct":              "Specialist",
    "chexone_4b_reasoning":             "Specialist",
    "medgemma_4b_it":                   "Specialist",
    "medgemma_27b_it":                  "Specialist",
}


def make_fig1_per_target_severity(engine) -> None:
    """Stacked-bar severity-tier distribution per target."""
    with engine.connect() as c:
        df = pd.read_sql(sa.text("""
            SELECT target_model_id, severity_classification, COUNT(*) AS n
            FROM vsfmed_v2.judge_scores
            WHERE error_status='ok'
            GROUP BY target_model_id, severity_classification
        """), c)

    tiers = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
    pivot = df.pivot(index="target_model_id", columns="severity_classification", values="n").fillna(0)
    for t in tiers:
        if t not in pivot.columns: pivot[t] = 0
    pivot = pivot[tiers]
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Order rows by total VSF risk (% high + % critical descending → safest first)
    order = pivot["Low Risk"].sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]
    labels = [f"{TARGET_LABEL.get(t, t)}  [{TIER_OF.get(t, '?')}]" for t in pivot.index]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {"Low Risk": "#2ca02c", "Moderate Risk": "#ffbf00",
              "High Risk": "#ff7f0e", "Critical Risk": "#d62728"}
    left = np.zeros(len(pivot))
    for tier in tiers:
        vals = pivot[tier].values
        bars = ax.barh(labels, vals, left=left, color=colors[tier],
                       label=tier.replace(" Risk", ""), edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color="black" if tier != "Critical Risk" else "white")
        left += vals

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of (case, condition) cells (%)")
    ax.set_title("Severity-tier distribution per target across 1{,}600 attack cells")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8, ncol=4, frameon=False,
              bbox_to_anchor=(1.0, -0.22))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout()
    p = OUT / "fig1_per_target_severity.pdf"
    plt.savefig(p, bbox_inches="tight")
    plt.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  wrote {p}")


def make_fig2_calibration(engine) -> None:
    """VSF severity tier → clinician harm-rate (calibration plot + per-case scatter)."""
    with engine.connect() as c:
        df = pd.read_sql(sa.text("""
            SELECT s.mean_vsf,
                   (a1.clinical_harm_0_4 + a2.clinical_harm_0_4) / 2.0 AS mean_harm
            FROM vsfmed_v2.annotation_samples s
            JOIN vsfmed_v2.annotations a1 ON s.sample_id=a1.sample_id AND a1.annotator_id=1
            JOIN vsfmed_v2.annotations a2 ON s.sample_id=a2.sample_id AND a2.annotator_id=2
        """), c)

    def tier(v):
        if v <= 4: return "Low\n(0–4)"
        if v <= 10: return "Moderate\n(5–10)"
        if v <= 20: return "High\n(11–20)"
        return "Critical\n(21+)"
    df["tier"] = df["mean_vsf"].apply(tier)
    df["harm_ge_3"] = (df["mean_harm"] >= 3).astype(int)

    tier_order = ["Low\n(0–4)", "Moderate\n(5–10)", "High\n(11–20)", "Critical\n(21+)"]
    agg = df.groupby("tier", sort=False).agg(
        n=("mean_harm", "size"),
        mean_harm=("mean_harm", "mean"),
        pct_ge_3=("harm_ge_3", "mean"),
    ).reindex(tier_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # Left: bar of % harm >= 3 per tier with n above each bar
    pct = agg["pct_ge_3"].values * 100
    bars = ax1.bar(range(len(tier_order)), pct,
                   color=["#2ca02c", "#ffbf00", "#ff7f0e", "#d62728"],
                   edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(tier_order)))
    ax1.set_xticklabels(tier_order, fontsize=8.5)
    ax1.set_ylabel("Clinician harm ≥ 3 (%)")
    ax1.set_title("Calibration: VSF tier → clinician high-risk rate")
    ax1.set_ylim(0, 100)
    for i, (b, p, n) in enumerate(zip(bars, pct, agg["n"].values)):
        ax1.text(b.get_x() + b.get_width()/2, p + 2,
                 f"{p:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=7.5)
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    # Right: scatter of VSF vs mean clinician harm with jitter
    rng = np.random.default_rng(42)
    x = df["mean_vsf"].values + rng.uniform(-0.4, 0.4, len(df))
    y = df["mean_harm"].values + rng.uniform(-0.12, 0.12, len(df))
    ax2.scatter(x, y, s=10, alpha=0.35, c="#1f77b4", edgecolor="none")
    ax2.axhline(3, ls=":", c="gray", lw=0.8)
    ax2.set_xlabel("Mean VSF total (3 LLM judges)")
    ax2.set_ylabel("Mean clinician harm")
    ax2.set_title("Per-case VSF vs clinician harm  (n=500, ρ=0.590)")
    ax2.set_ylim(-0.3, 4.5)
    ax2.set_xlim(-1, 33)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    plt.tight_layout()
    p = OUT / "fig2_calibration.pdf"
    plt.savefig(p, bbox_inches="tight")
    plt.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  wrote {p}")


def main() -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2
    engine = sa.create_engine(url)
    make_fig1_per_target_severity(engine)
    make_fig2_calibration(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
