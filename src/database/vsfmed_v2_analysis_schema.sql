-- Analysis result tables for VSF-Med v2.
-- These are derived from the raw judge_scores + annotations + clinician data,
-- and are stored here so the paper's headline numbers are reproducible from
-- the database without re-running the full Python analysis.

SET search_path TO vsfmed_v2, public;

-- ---------------- generic key-value scalar metrics ----------------
-- e.g. pooled Krippendorff alpha, AUROC, weighted Cohen kappa.
CREATE TABLE IF NOT EXISTS vsfmed_v2.analysis_metrics (
    metric_id        SERIAL PRIMARY KEY,
    metric_name      TEXT NOT NULL,          -- e.g. "krippendorff_alpha_pooled"
    metric_scope     TEXT NOT NULL,          -- "full", "lofo", "two_judge", "clinician"
    metric_value     DOUBLE PRECISION,
    metric_lower     DOUBLE PRECISION,       -- 95% CI lower (optional)
    metric_upper     DOUBLE PRECISION,       -- 95% CI upper (optional)
    n_observations   INT,
    description      TEXT,
    inserted_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (metric_name, metric_scope)
);
CREATE INDEX IF NOT EXISTS idx_metric_name  ON vsfmed_v2.analysis_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metric_scope ON vsfmed_v2.analysis_metrics(metric_scope);

-- ---------------- per-dimension Krippendorff alpha ----------------
CREATE TABLE IF NOT EXISTS vsfmed_v2.judge_dim_alpha (
    dim_id          SERIAL PRIMARY KEY,
    dimension       TEXT NOT NULL,           -- e.g. "score_prompt_injection"
    judge_scope     TEXT NOT NULL,           -- "3_judge_full" | "lofo"
    n_judges        SMALLINT NOT NULL,
    krippendorff_alpha DOUBLE PRECISION,
    classification  TEXT,                    -- "preferred" | "acceptable" | "below_floor"
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (dimension, judge_scope)
);

-- ---------------- pairwise Spearman rho between judges ----------------
CREATE TABLE IF NOT EXISTS vsfmed_v2.judge_pair_rho (
    pair_id         SERIAL PRIMARY KEY,
    judge_a         TEXT NOT NULL,
    judge_b         TEXT NOT NULL,
    target_scope    TEXT NOT NULL DEFAULT 'vsf_total',  -- which score we correlated
    spearman_rho    DOUBLE PRECISION,
    n_pairs         INT,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (judge_a, judge_b, target_scope)
);

-- ---------------- per-target severity tier table (paper Table I) ----------------
CREATE TABLE IF NOT EXISTS vsfmed_v2.per_target_severity (
    row_id          SERIAL PRIMARY KEY,
    target_model_id TEXT NOT NULL,
    tier            TEXT NOT NULL,           -- specialist | frontier | open_generalist
    aggregation     TEXT NOT NULL,           -- "response_cell_mean" | "per_judge"
    n               INT NOT NULL,
    low_pct         DOUBLE PRECISION,
    moderate_pct    DOUBLE PRECISION,
    high_pct        DOUBLE PRECISION,
    critical_pct    DOUBLE PRECISION,
    mean_vsf        DOUBLE PRECISION,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (target_model_id, aggregation)
);

-- ---------------- linear mixed-effects fixed effects (paper Fig 2) ----------------
CREATE TABLE IF NOT EXISTS vsfmed_v2.lme_fixed_effects (
    fe_id           SERIAL PRIMARY KEY,
    model_formula   TEXT NOT NULL,
    term            TEXT NOT NULL,
    coef            DOUBLE PRECISION,
    se              DOUBLE PRECISION,
    ci_low          DOUBLE PRECISION,
    ci_high         DOUBLE PRECISION,
    pval            DOUBLE PRECISION,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_formula, term)
);

-- ---------------- VSF tier vs clinician calibration (paper Table V) ----------------
CREATE TABLE IF NOT EXISTS vsfmed_v2.clinician_calibration (
    cal_id          SERIAL PRIMARY KEY,
    vsf_tier        TEXT NOT NULL,           -- "Low" | "Moderate" | "High" | "Critical"
    vsf_lower       DOUBLE PRECISION,        -- inclusive lower bound on mean VSF
    vsf_upper       DOUBLE PRECISION,        -- exclusive upper bound (NULL = +inf)
    n_samples       INT NOT NULL,
    mean_harm       DOUBLE PRECISION,
    pct_harm_ge_3   DOUBLE PRECISION,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (vsf_tier)
);
