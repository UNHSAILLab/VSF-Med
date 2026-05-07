"""Inter-judge reliability metrics for VSF-Med v2.

Computes:
  - Krippendorff α per scoring dimension and across all dimensions
  - Spearman ρ between every judge pair
  - Score-distribution entropy per judge (guards against collapse-to-zero)
  - Per-judge mean ± SD per dimension

Operates on data/results/{...}_judge_scores.jsonl.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCORE_DIMS = (
    "score_prompt_injection",
    "score_jailbreak",
    "score_confidentiality",
    "score_misinformation",
    "score_dos_or_refusal",
    "score_persistence",
    "score_safety_bypass",
    "score_clinical_decision_impact",
    "score_diagnostic_correctness",
    "score_appropriate_refusal",
)


@dataclasses.dataclass
class ReliabilityReport:
    n_responses: int
    judges: List[str]
    krippendorff_alpha_per_dim: Dict[str, float]
    krippendorff_alpha_pooled: float
    spearman_rho_pairs: Dict[Tuple[str, str], Dict[str, float]]   # (j1, j2) -> dim -> rho
    score_distribution_entropy: Dict[str, Dict[str, float]]       # judge -> dim -> H
    per_judge_mean_sd: Dict[str, Dict[str, Tuple[float, float]]]  # judge -> dim -> (mean, sd)

    def to_dict(self) -> dict:
        return {
            "n_responses": self.n_responses,
            "judges": self.judges,
            "krippendorff_alpha_per_dim": self.krippendorff_alpha_per_dim,
            "krippendorff_alpha_pooled": self.krippendorff_alpha_pooled,
            "spearman_rho_pairs": {
                f"{a}|{b}": v for (a, b), v in self.spearman_rho_pairs.items()
            },
            "score_distribution_entropy": self.score_distribution_entropy,
            "per_judge_mean_sd": self.per_judge_mean_sd,
        }


def _load_scores(path: Path) -> List[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("error_status") != "ok":
                continue
            rows.append(d)
    return rows


def _index_by_response(rows: List[dict]) -> Dict[str, Dict[str, dict]]:
    """response_id -> judge_model_id -> row."""
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["response_id"]][r["judge_model_id"]] = r
    return out


# ---------- Krippendorff's α (interval, with missing data) ----------

def _krippendorff_alpha(matrix: List[List[Optional[float]]]) -> float:
    """matrix[i][j] = score from judge i on item j (None for missing).

    Implements interval Krippendorff with pairwise distances. Returns nan
    when there is not enough data to compute observed disagreement.
    """
    n_judges = len(matrix)
    n_items = len(matrix[0]) if matrix else 0
    if n_items < 2 or n_judges < 2:
        return float("nan")

    # Per-item ratings list
    ratings_per_item: List[List[float]] = []
    for j in range(n_items):
        col = [matrix[i][j] for i in range(n_judges) if matrix[i][j] is not None]
        if len(col) >= 2:
            ratings_per_item.append([float(v) for v in col])

    if not ratings_per_item:
        return float("nan")

    # Observed disagreement (pairwise within item)
    Do_num = Do_den = 0.0
    all_values: List[float] = []
    for col in ratings_per_item:
        m = len(col)
        all_values.extend(col)
        for a in range(m):
            for b in range(a + 1, m):
                Do_num += (col[a] - col[b]) ** 2
        Do_den += m * (m - 1) / 2

    if Do_den == 0:
        return float("nan")
    Do = Do_num / Do_den

    # Expected disagreement (across all pairs, ignoring item)
    n = len(all_values)
    if n < 2:
        return float("nan")
    De_num = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            De_num += (all_values[a] - all_values[b]) ** 2
    De = De_num / (n * (n - 1) / 2)

    if De == 0:
        return float("nan")
    return 1.0 - Do / De


# ---------- Spearman ρ ----------

def _rank(xs: List[float]) -> List[float]:
    """Average-rank ranking (handles ties)."""
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _spearman_rho(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


# ---------- distribution entropy ----------

def _entropy_norm(values: Iterable[int], k_bins: int = 5) -> float:
    """Shannon entropy normalized to [0, 1] across k_bins (the 0..4 ordinal levels)."""
    counts = [0] * k_bins
    n = 0
    for v in values:
        idx = max(0, min(k_bins - 1, int(v)))
        counts[idx] += 1
        n += 1
    if n == 0:
        return float("nan")
    H = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / n
        H -= p * math.log2(p)
    return H / math.log2(k_bins)


# ---------- public API ----------

def compute_reliability(scores_jsonl: Path) -> ReliabilityReport:
    rows = _load_scores(scores_jsonl)
    by_resp = _index_by_response(rows)
    judges = sorted({r["judge_model_id"] for r in rows})
    response_ids = sorted(by_resp.keys())

    # Build judge × response matrix per dimension
    alpha_per_dim: Dict[str, float] = {}
    pooled_matrix: List[List[Optional[float]]] = [
        [None] * (len(response_ids) * len(SCORE_DIMS)) for _ in judges
    ]

    for di, dim in enumerate(SCORE_DIMS):
        matrix = [[None] * len(response_ids) for _ in judges]
        for ri, rid in enumerate(response_ids):
            for ji, j in enumerate(judges):
                row = by_resp[rid].get(j)
                if row is not None:
                    matrix[ji][ri] = float(row[dim])
                    pooled_matrix[ji][di * len(response_ids) + ri] = float(row[dim])
        alpha_per_dim[dim] = _krippendorff_alpha(matrix)

    pooled_alpha = _krippendorff_alpha(pooled_matrix)

    # Spearman ρ per judge pair, per dimension
    rho_pairs: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ai in range(len(judges)):
        for bi in range(ai + 1, len(judges)):
            ja, jb = judges[ai], judges[bi]
            d_rho: Dict[str, float] = {}
            for dim in SCORE_DIMS:
                xs, ys = [], []
                for rid in response_ids:
                    ra, rb = by_resp[rid].get(ja), by_resp[rid].get(jb)
                    if ra and rb:
                        xs.append(ra[dim])
                        ys.append(rb[dim])
                d_rho[dim] = _spearman_rho(xs, ys)
            rho_pairs[(ja, jb)] = d_rho

    # Distribution entropy and mean/sd per judge per dim
    entropy: Dict[str, Dict[str, float]] = {j: {} for j in judges}
    mean_sd: Dict[str, Dict[str, Tuple[float, float]]] = {j: {} for j in judges}
    for j in judges:
        rows_j = [r for r in rows if r["judge_model_id"] == j]
        for dim in SCORE_DIMS:
            vals = [r[dim] for r in rows_j]
            entropy[j][dim] = _entropy_norm(vals)
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                mean_sd[j][dim] = (mean, math.sqrt(var))
            else:
                mean_sd[j][dim] = (float("nan"), float("nan"))

    return ReliabilityReport(
        n_responses=len(response_ids),
        judges=judges,
        krippendorff_alpha_per_dim=alpha_per_dim,
        krippendorff_alpha_pooled=pooled_alpha,
        spearman_rho_pairs=rho_pairs,
        score_distribution_entropy=entropy,
        per_judge_mean_sd=mean_sd,
    )
