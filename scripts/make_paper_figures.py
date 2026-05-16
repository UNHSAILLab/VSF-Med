"""Generate paper figures with a Medtronic-inspired palette.

  fig1_per_target_severity.pdf  — stacked-bar severity-tier distribution per target
  fig2_lme_interaction.pdf      — LME coefficients: specialist disadvantage per condition
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "VSF_Med_IEEE/images/v2"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Medtronic-inspired palette ----------
MED_NAVY = "#001F5B"     # Medtronic navy
MED_BLUE = "#0066B3"     # Primary blue
MED_CYAN = "#00A5E0"     # Bright cyan-blue
MED_GREEN = "#00A859"    # Safe-tier green
MED_GOLD = "#FFC72C"     # Moderate-tier gold
MED_ORANGE = "#F47B30"   # High-tier orange
MED_RED = "#C8102E"      # Critical-tier red
MED_GRAY = "#7A7A7A"
MED_LIGHTGRAY = "#D9D9D9"

TIER_COLORS = {
    "Low Risk": MED_GREEN,
    "Moderate Risk": MED_GOLD,
    "High Risk": MED_ORANGE,
    "Critical Risk": MED_RED,
}

# Global style: clean, lots of whitespace, navy text
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MED_NAVY,
    "axes.labelcolor": MED_NAVY,
    "axes.titlecolor": MED_NAVY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MED_NAVY,
    "ytick.color": MED_NAVY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


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
    """Per-target severity tier distribution (mean-of-judges response cells).

    Label-readability fix: bars are extra-tall, percent labels go on top of each
    segment with adaptive contrast (white on dark, navy on light), and a value
    column sits to the right of each bar so small slices still get a number.
    """
    with engine.connect() as c:
        df = pd.read_sql(sa.text("""
            SELECT j.case_id, j.condition_id, j.target_model_id,
                   r.model_family AS tier,
                   AVG(j.vsf_total)::REAL AS mean_vsf
            FROM vsfmed_v2.judge_scores j
            JOIN vsfmed_v2.model_responses r USING (response_id)
            WHERE j.error_status='ok' AND r.error_status='ok'
            GROUP BY j.case_id, j.condition_id, j.target_model_id, r.model_family
        """), c)

    def to_tier(v):
        if v < 5: return "Low Risk"
        if v < 11: return "Moderate Risk"
        if v < 21: return "High Risk"
        return "Critical Risk"
    df["severity"] = df["mean_vsf"].apply(to_tier)

    tiers = ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
    pivot = df.groupby(["target_model_id","severity"]).size().unstack(fill_value=0)
    for t in tiers:
        if t not in pivot.columns: pivot[t] = 0
    pivot = pivot[tiers]
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Sort: safest first (highest Low Risk share)
    order = pivot["Low Risk"].sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]
    labels = [f"{TARGET_LABEL[t]}\n[{TIER_OF[t]}]" for t in pivot.index]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.18)

    left = np.zeros(len(pivot))
    for tier in tiers:
        vals = pivot[tier].values
        ax.barh(labels, vals, left=left, height=0.68,
                color=TIER_COLORS[tier], edgecolor="white", linewidth=0.6,
                label=tier.replace(" Risk", ""))
        for i, v in enumerate(vals):
            if v >= 3.5:
                # Choose label color for contrast against the segment fill
                bg = TIER_COLORS[tier]
                white_bg = bg in (MED_RED, MED_NAVY)   # very dark backgrounds
                txt_color = "white" if white_bg else MED_NAVY
                ax.text(left[i] + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold",
                        zorder=5)
            elif 0.05 <= v < 3.5:
                # Too small to label inside; annotate outside the right edge of the row
                pass
        left += vals

    # Total-row annotations on the right edge (replaces the small-slice gap)
    crit = pivot["Critical Risk"].values
    for i, c in enumerate(crit):
        ax.text(101.5, i, f"{c:.1f}% crit",
                ha="left", va="center", fontsize=7.5, color=MED_RED,
                fontweight="bold" if c >= 1.0 else "normal")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Share of response cells (%)", color=MED_NAVY)
    ax.set_title("Severity-tier distribution by target (n=1{,}600 response cells per target)",
                 pad=12)
    ax.invert_yaxis()

    # Legend below the chart
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                    fontsize=9, ncol=4, frameon=False,
                    handlelength=1.5, handletextpad=0.6, columnspacing=1.8)
    for text in leg.get_texts():
        text.set_color(MED_NAVY)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=MED_LIGHTGRAY, lw=0.5)
    ax.tick_params(left=False)

    p = OUT / "fig1_per_target_severity.pdf"
    plt.savefig(p, bbox_inches="tight")
    plt.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close()
    print(f"  wrote {p}")


def make_fig2_lme_interaction() -> None:
    """LME interaction coefficients: specialist disadvantage per attack condition.

    Reads from analysis/lofo_and_mixed_effects.json (or refits if missing).
    Shows tier[specialist] x condition interactions with 95% CI error bars.
    """
    # Load the precomputed LME results
    src = ROOT / "analysis/lofo_and_mixed_effects.json"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}; run scripts/analysis_lofo_and_mixed_effects.py first")
    data = json.loads(src.read_text())
    fixed = data["mixed_effects"]["fixed_effects"]

    # Pull only the tier:condition_id interaction rows
    rows = []
    for fe in fixed:
        term = fe["term"]
        if "tier[T.specialist]:condition_id[T." in term:
            # Extract condition id, e.g. "C4_persistence"
            cond = term.split("condition_id[T.")[1].rstrip("]")
            rows.append({
                "condition": cond,
                "coef": fe["coef"],
                "ci_low": fe["ci_low"],
                "ci_high": fe["ci_high"],
                "pval": fe["pval"],
            })
    df = pd.DataFrame(rows)

    # Pretty labels and ordering
    pretty = {
        "C2_prompt_injection":   "C2 Prompt injection",
        "C3_jailbreak":          "C3 Jailbreak",
        "C4_persistence":        "C4 Persistence",
        "C5_misinformation":     "C5 Misinformation",
        "C6_confidentiality":    "C6 Confidentiality",
        "C7_visual_artifact":    "C7 Visual artifact",
        "C8_combined":           "C8 Combined image+text",
    }
    order = ["C4_persistence","C5_misinformation","C3_jailbreak",
             "C6_confidentiality","C8_combined","C2_prompt_injection","C7_visual_artifact"]
    df["sort_key"] = df["condition"].map({c: i for i, c in enumerate(order)})
    df = df.sort_values("sort_key")
    df["label"] = df["condition"].map(pretty)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.87, bottom=0.16)

    y_pos = np.arange(len(df))
    colors = [MED_BLUE if p < 0.05 else MED_GRAY for p in df["pval"]]
    err_low = (df["coef"] - df["ci_low"]).values
    err_high = (df["ci_high"] - df["coef"]).values

    bars = ax.barh(y_pos, df["coef"].values, height=0.62,
                   xerr=[err_low, err_high],
                   color=colors, edgecolor="white", linewidth=0.6,
                   error_kw=dict(ecolor=MED_NAVY, lw=1.0, capsize=3))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, color=MED_NAVY)

    # Vertical zero line
    ax.axvline(0, color=MED_NAVY, lw=0.8, ls="-")

    # Annotate coefficient values to the right of each bar
    for i, (coef, p, hi) in enumerate(zip(df["coef"].values, df["pval"].values, df["ci_high"].values)):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(hi + 0.15, i, f"{coef:+.2f}  {sig}",
                va="center", fontsize=8.5, color=MED_NAVY)

    ax.set_xlabel("Specialist disadvantage in VSF total (vs.\\ frontier, at the same condition)",
                  color=MED_NAVY)
    ax.set_title("Specialist - Frontier interaction by attack condition\n"
                 "linear mixed-effects, $n=11{,}199$ cells, 200 case groups",
                 pad=10)
    ax.set_xlim(-1.5, df["ci_high"].max() + 1.2)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=MED_LIGHTGRAY, lw=0.5)
    ax.tick_params(left=False)

    # Significance legend (top right)
    leg_items = [
        plt.Rectangle((0,0),1,1, color=MED_BLUE,  label="$p < 0.05$"),
        plt.Rectangle((0,0),1,1, color=MED_GRAY,  label="not significant"),
    ]
    leg = ax.legend(handles=leg_items, loc="lower right", fontsize=8, frameon=False)
    for t in leg.get_texts(): t.set_color(MED_NAVY)

    p = OUT / "fig2_lme_interaction.pdf"
    plt.savefig(p, bbox_inches="tight")
    plt.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
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
    make_fig2_lme_interaction()
    return 0


if __name__ == "__main__":
    sys.exit(main())
