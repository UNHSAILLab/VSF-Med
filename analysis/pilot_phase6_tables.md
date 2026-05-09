# VSF-Med v2 Pilot — Phase 6 Tables (full 7-target lineup)

Pilot cohort: 200 cases × 8 conditions × **7 target models** = 11,200 responses

Judges: 3 (Claude Haiku 4.5, Gemini 2.5 Flash, GPT-4o-mini)

Total LLM-judge scores: 33,599


## Table 1 — Per-judge calibration

| Judge | n | mean VSF | SD | %Critical |
|---|---:|---:|---:|---:|
| claude-haiku-4-5-20251001 | 11,200 | 4.24 | 4.99 | 1.1% |
| gemini-2.5-flash | 11,199 | 2.73 | 4.40 | 1.1% |
| gpt-4o-mini | 11,200 | 5.00 | 4.72 | 0.5% |

## Table 2 — Inter-judge reliability (n=11,200)

| Dimension | Krippendorff α (3-judge) | Status |
|---|---:|---|
| score_prompt_injection | 0.500 | ✗ below floor |
| score_jailbreak | 0.401 | ✗ below floor |
| score_confidentiality | 0.760 | ✓ preferred |
| score_misinformation | 0.618 | ✓ preferred |
| score_dos_or_refusal | 0.537 | ~ acceptable |
| score_persistence | 0.560 | ~ acceptable |
| score_safety_bypass | 0.347 | ✗ below floor |
| score_clinical_decision_impact | 0.476 | ✗ below floor |
| score_diagnostic_correctness | 0.568 | ~ acceptable |
| score_appropriate_refusal | 0.491 | ✗ below floor |
| **POOLED** | **0.586** | acceptable |

## Table 3 — Target × condition mean VSF (averaged across 3 judges)

| Target | Tier | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | Overall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CheXOne (instruct) | specialist | 1.3 | 2.9 | 5.0 | 6.0 | 5.8 | 3.0 | 1.6 | 1.9 | **3.43** |
| CheXOne (reasoning) | specialist | 1.7 | 2.7 | 5.2 | 6.7 | 6.4 | 2.0 | 2.1 | 2.5 | **3.64** |
| MedGemma 27B | specialist | 2.2 | 5.1 | 7.9 | 12.3 | 8.9 | 6.5 | 2.7 | 3.3 | **6.12** |
| MedGemma 4B | specialist | 2.3 | 3.7 | 8.1 | 7.9 | 8.4 | 6.1 | 2.6 | 3.0 | **5.25** |
| Claude Haiku 4.5 | frontier | 2.7 | 1.0 | 4.5 | 3.6 | 4.1 | 1.7 | 2.7 | 2.3 | **2.83** |
| GPT-5.4-mini | frontier | 1.2 | 4.4 | 4.1 | 3.5 | 4.2 | 1.6 | 1.6 | 2.0 | **2.82** |
| Gemini 3 Flash | frontier | 1.7 | 4.2 | 5.7 | 3.6 | 6.6 | 4.4 | 1.9 | 2.5 | **3.84** |
| **TIER AVG: specialist** | specialist | 1.9 | 3.6 | 6.5 | 8.2 | 7.4 | 4.4 | 2.2 | 2.7 | **4.61** |
| **TIER AVG: frontier** | frontier | 1.9 | 3.2 | 4.8 | 3.6 | 4.9 | 2.6 | 2.1 | 2.3 | **3.16** |

## Table 4 — Per-target severity tier distribution

| Target | Tier | n | %Low | %Mod | %High | %Critical |
|---|---|---:|---:|---:|---:|---:|
| CheXOne (instruct) | specialist | 4,800 | 70.3% | 21.4% | 7.9% | **0.5%** |
| CheXOne (reasoning) | specialist | 4,800 | 68.5% | 22.4% | 8.9% | **0.2%** |
| MedGemma 27B | specialist | 4,800 | 50.6% | 29.1% | 16.8% | **3.5%** |
| MedGemma 4B | specialist | 4,800 | 56.3% | 27.3% | 15.1% | **1.4%** |
| Claude Haiku 4.5 | frontier | 4,800 | 75.9% | 18.5% | 5.6% | **0.0%** |
| GPT-5.4-mini | frontier | 4,799 | 75.7% | 18.7% | 5.4% | **0.1%** |
| Gemini 3 Flash | frontier | 4,800 | 66.4% | 24.4% | 8.7% | **0.4%** |

## Table 5 — High-confidence critical responses (all 3 judges ≥ 18)

**61 responses** flagged critical-risk by 3-judge consensus. Distribution:

| Target | Tier | Condition | Count |
|---|---|---|---:|
| MedGemma 27B | specialist | C4_persistence | 51 |
| Gemini 3 Flash | frontier | C4_persistence | 5 |
| MedGemma 4B | specialist | C4_persistence | 3 |
| CheXOne (reasoning) | specialist | C4_persistence | 1 |
| CheXOne (reasoning) | specialist | C3_jailbreak | 1 |

## Table 6 — Pairwise Spearman ρ on vsf_total

| Judge A | Judge B | ρ | n |
|---|---|---:|---:|
| claude-haiku-4-5-20251001 | gemini-2.5-flash | 0.728 | 11,199 |
| claude-haiku-4-5-20251001 | gpt-4o-mini | 0.646 | 11,200 |
| gemini-2.5-flash | gpt-4o-mini | 0.567 | 11,199 |