"""Build the VSF-Med v2 HuggingFace dataset bundle (gated, framework-only).

Bundle contents:
  README.md                  dataset card with gated-access metadata
  DATA_USE_AGREEMENT.md      what credentialed users accept
  protocol/                  the v2 protocol freeze (6 YAML/MD artifacts)
  analysis/                  aggregate results pulled from Neon (6 CSVs)
  schema/                    Postgres DDL for the data + analysis tables

NOT included (PhysioNet credentialed, never published):
  - MIMIC-CXR images
  - MIMIC-Diff-VQA question/answer text
  - Per-response model outputs
  - Per-case clinician annotations
  - Case IDs that resolve back to PhysioNet patient IDs

This is a framework + aggregate-results release, not a raw-data release.
Reproducers must obtain PhysioNet credentialing and follow the protocol.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "hf_dataset"


def _load_env() -> None:
    p = ROOT / ".env"
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


README = """\
---
license: cc-by-nc-4.0
language:
  - en
tags:
  - medical
  - vision-language
  - adversarial-robustness
  - safety
  - benchmark
  - radiology
size_categories:
  - n<1K
pretty_name: VSF-Med v2 (Framework + Aggregate Results)
viewer: false
extra_gated_prompt: >-
  Access to this dataset requires PhysioNet credentialing for MIMIC-CXR
  (https://physionet.org/content/mimic-cxr-jpg/) and acceptance of the
  Data Use Agreement included in this repository. The dataset itself
  contains only framework artifacts and aggregate results; reproduction
  requires obtaining MIMIC-CXR and MIMIC-Diff-VQA separately.
extra_gated_fields:
  PhysioNet credentialed:
    type: checkbox
  Institution:
    type: text
  Research purpose:
    type: text
  Agree to data use terms:
    type: checkbox
---

# VSF-Med v2 — Vulnerability Scoring Framework for Medical Vision-Language Models

This dataset contains the **framework and aggregate analysis results**
for VSF-Med v2, a clinician-validated adversarial-evaluation framework
for medical vision-language models (VLMs). It does **not** contain MIMIC-CXR
images or MIMIC-Diff-VQA question/answer pairs; those must be obtained
separately through PhysioNet credentialing.

## What is here

```
protocol/                  v2 protocol freeze artifacts
├── attack_families.yaml         8-condition attack taxonomy (C1 through C8)
├── scoring_rubric.yaml          8 vulnerability + 2 utility dimensions, 0-4 ordinal
├── model_list.yaml              7 target models + 3 LLM judges (snapshots)
├── dataset_sampling.yaml        sampling spec across MIMIC-CXR / CheXpert / GMAI-MMBench
├── statistical_analysis_plan.md pre-registered SAP
└── clinician_annotation_form.md 8-field clinician rubric + 2-rater protocol

analysis/                  aggregate results from the pilot
├── analysis_metrics.csv         scalar headline metrics (kappa, alpha, rho, AUROC)
├── judge_dim_alpha.csv          per-dimension Krippendorff alpha (3-judge and LOFO)
├── judge_pair_rho.csv           pairwise Spearman rho between the 3 LLM judges
├── per_target_severity.csv      per-target severity tier distribution
├── lme_fixed_effects.csv        mixed-effects fixed-effect coefficients
└── clinician_calibration.csv    VSF tier vs clinician harm calibration

schema/                    Postgres DDL
├── vsfmed_v2_schema.sql              base tables + clinician annotation pipeline
└── vsfmed_v2_analysis_schema.sql     derived analysis tables
```

## What is NOT here (and why)

The following components are PhysioNet credentialed and remain gated outside
of HuggingFace. Reproducers must obtain them through their original sources:

| Component | Source | Access |
|---|---|---|
| MIMIC-CXR images | https://physionet.org/content/mimic-cxr-jpg/ | Credentialed |
| MIMIC-Diff-VQA splits | Hu et al., KDD 2023 | https://github.com/Holipori/MIMIC-Diff-VQA |
| Per-response model outputs | This pilot | Credentialed access on request |
| Per-case clinician harm ratings | This pilot | Credentialed access on request |
| Attack template repertoire | This work | Restricted (dual-use risk) |

## How to reproduce

1. Obtain PhysioNet credentialing for MIMIC-CXR-JPG.
2. Download the MIMIC-Diff-VQA splits from the source repository.
3. Clone the VSF-Med codebase: https://github.com/UNHSAILLab/VSF-Med
4. Follow the protocol in `protocol/` to build base cases, materialize attack
   conditions, run the 7 target models, score with the 3 LLM judges, and
   collect the 500-case dual clinician annotation.
5. The aggregate results in `analysis/` are what your reproduction should match.

## Headline numbers (pilot, n = 200 base cases x 8 conditions x 7 targets)

| Metric | Value | Preferred bar |
|---|---:|---:|
| Weighted Cohen's kappa (linear, clinician harm) | 0.645 | >= 0.55 |
| Spearman rho(mean VSF, mean clinician harm) | 0.590 | >= 0.45 |
| AUROC (VSF predicts clinician harm >= 3) | 0.836 | >= 0.70 |
| Pooled 3-judge Krippendorff alpha | 0.586 | >= 0.60 |

## Citation

A workshop paper describing this pilot is in submission to ICMLA 2026.
See the GitHub repository for the latest preprint and BibTeX entry.

## License

Framework artifacts are released under CC BY-NC 4.0. The MIMIC-CXR images
and MIMIC-Diff-VQA annotations referenced (but not redistributed) by this
dataset are subject to their original PhysioNet and source licenses.

## Contact

Open an issue at https://github.com/UNHSAILLab/VSF-Med for questions,
or use the gated-access form on this dataset's HuggingFace page.
"""


DUA = """\
# Data Use Agreement — VSF-Med v2 Framework Release

By accessing this dataset you confirm that:

1. **You hold PhysioNet credentialing** for MIMIC-CXR or are working with a
   collaborator who does. The framework references MIMIC-CXR derivatives;
   you may not infer or attempt to recover identifiers from any aggregate
   metric in this release.

2. **You will not redistribute** any element of this release that lets a
   third party reconstruct MIMIC-CXR-derived content, including but not
   limited to:
   - Per-case prompts that quote MIMIC-Diff-VQA question text
   - Image hashes, file paths, or case identifiers that resolve to
     PhysioNet patient identifiers
   - Per-response model outputs from our pilot

3. **You will treat the attack template repertoire** (not in this release,
   available separately) as dual-use research and follow responsible
   disclosure if you discover new attack vectors.

4. **You will cite** the VSF-Med v2 paper and the underlying MIMIC-CXR and
   MIMIC-Diff-VQA papers in any publication that uses this framework.

5. **You will not use** the framework to harm patients or to evade
   regulated medical device approval processes.

Violations may result in revoked HuggingFace access and notification to
the credentialing institution.
"""


def write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def export_table_csv(conn, table: str, out: Path):
    df = pd.read_sql(sa.text(f"SELECT * FROM vsfmed_v2.{table} ORDER BY 1"), conn)
    # Drop server-side bookkeeping columns
    for c in ("inserted_at",):
        if c in df.columns: df = df.drop(columns=[c])
    df.to_csv(out, index=False)
    print(f"    wrote {out.name}  ({len(df)} rows)")


def main() -> int:
    _load_env()
    url = os.environ.get("VSF_MED_DATABASE_URL")
    if not url:
        print("ERROR: VSF_MED_DATABASE_URL not set", file=sys.stderr)
        return 2

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()

    # 1. Readme + DUA at top level
    write_text(BUNDLE / "README.md", README)
    write_text(BUNDLE / "DATA_USE_AGREEMENT.md", DUA)

    # 2. Protocol
    p_dir = BUNDLE / "protocol"; p_dir.mkdir()
    for fn in ["attack_families.yaml","scoring_rubric.yaml","model_list.yaml",
               "dataset_sampling.yaml","statistical_analysis_plan.md",
               "clinician_annotation_form.md"]:
        src = ROOT / "protocol" / fn
        if src.exists():
            shutil.copyfile(src, p_dir / fn)
    print(f"  protocol/: {sum(1 for _ in p_dir.iterdir())} files")

    # 3. Schema
    s_dir = BUNDLE / "schema"; s_dir.mkdir()
    for fn in ["vsfmed_v2_schema.sql","vsfmed_v2_analysis_schema.sql"]:
        src = ROOT / "src/database" / fn
        if src.exists():
            shutil.copyfile(src, s_dir / fn)
    print(f"  schema/: {sum(1 for _ in s_dir.iterdir())} files")

    # 4. Analysis CSVs from Neon (authoritative source)
    a_dir = BUNDLE / "analysis"; a_dir.mkdir()
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        for tbl in ("analysis_metrics","judge_dim_alpha","judge_pair_rho",
                    "per_target_severity","lme_fixed_effects","clinician_calibration"):
            export_table_csv(conn, tbl, a_dir / f"{tbl}.csv")

    # Final size report
    total = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    n_files = sum(1 for p in BUNDLE.rglob("*") if p.is_file())
    print(f"\nBundle: {n_files} files, {total/1024:.1f} KB at {BUNDLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
