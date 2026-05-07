"""Drives multi-judge scoring over an existing pilot/full responses JSONL.

For each input row from data/results/pilot.{model}.jsonl the driver:
  1. Looks up the original benign clinical_prompt and ground_truth from
     data/processed/base_cases.csv (joined by case_id).
  2. Pulls the adversarial prompt from data/processed/eval_cases.jsonl
     (joined by case_id + condition_id).
  3. Sends (system_prompt, user_prompt) to each configured judge.
  4. Writes one JSONL row per (response, judge) pair.

Resumable: re-running skips (response_id, judge_model_id) pairs already ok.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .base import JUDGE_OUTPUT_FIELDS, JudgeScore, JudgeWrapper


# ------------- joiners -------------

def _load_base_cases(base_csv: Path) -> Dict[str, dict]:
    """case_id -> {clinical_prompt, ground_truth, ...}."""
    out: Dict[str, dict] = {}
    with Path(base_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["case_id"]] = row
    return out


def _load_eval_cases(eval_jsonl: Path) -> Dict[tuple, dict]:
    """(case_id, condition_id) -> eval row."""
    out: Dict[tuple, dict] = {}
    with Path(eval_jsonl).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(d["case_id"], d["condition_id"])] = d
    return out


# ------------- resume support -------------

def _existing_keys(scores_path: Path,
                   judge_model_ids: Sequence[str],
                   only_ok: bool = True) -> set:
    """Set of (response_id, judge_model_id) already done."""
    if not Path(scores_path).exists():
        return set()
    keys: set = set()
    judges = set(judge_model_ids)
    with Path(scores_path).open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("judge_model_id") not in judges:
                continue
            if only_ok and d.get("error_status") != "ok":
                continue
            keys.add((d.get("response_id"), d["judge_model_id"]))
    return keys


# ------------- main driver -------------

@dataclasses.dataclass
class MultiJudgeScorer:
    responses_paths: List[Path]
    base_cases_csv: Path
    eval_cases_jsonl: Path
    scores_path: Path
    judges: List[JudgeWrapper]
    response_limit: Optional[int] = None
    print_progress_every: int = 10
    resume: bool = True

    def run(self) -> dict:
        base = _load_base_cases(self.base_cases_csv)
        evals = _load_eval_cases(self.eval_cases_jsonl)
        existing = _existing_keys(self.scores_path,
                                  [j.judge_model_id for j in self.judges]) if self.resume else set()
        if existing:
            print(f"[judges] resuming — {len(existing)} (response, judge) pairs already done",
                  file=sys.stderr, flush=True)

        # Build the work queue: (response_row, judge) per pair not yet done
        work: List[tuple] = []
        responses_seen = 0
        max_responses = self.response_limit if self.response_limit is not None else float("inf")
        outer_done = False
        for rp in self.responses_paths:
            if outer_done:
                break
            for resp_row in self._iter_responses(Path(rp)):
                if responses_seen >= max_responses:
                    outer_done = True
                    break
                responses_seen += 1
                for judge in self.judges:
                    rid = self._response_id(resp_row)
                    if (rid, judge.judge_model_id) in existing:
                        continue
                    work.append((resp_row, judge))

        print(f"[judges] queued: {len(work)} judge calls "
              f"({len(self.judges)} judges × ~{len(work)//max(1,len(self.judges))} responses)",
              file=sys.stderr, flush=True)

        ok = err = 0
        t0 = time.monotonic()
        Path(self.scores_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(self.scores_path).open("a", encoding="utf-8") as out:
            for i, (resp_row, judge) in enumerate(work, 1):
                case_data = self._build_case_data(resp_row, base, evals)
                score = judge.score(case_data)
                d = score.to_jsonl_dict()
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                out.flush()
                if score.error_status == "ok":
                    ok += 1
                else:
                    err += 1
                if i % self.print_progress_every == 0 or i == len(work):
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed else 0
                    eta = (len(work) - i) / rate if rate else 0
                    print(f"[judges] {i}/{len(work)} ok={ok} err={err} "
                          f"rate={rate:.2f}/s eta={eta:.0f}s",
                          file=sys.stderr, flush=True)

        return {
            "judges": [j.judge_model_id for j in self.judges],
            "calls_attempted": len(work),
            "ok": ok, "err": err,
            "elapsed_seconds": time.monotonic() - t0,
        }

    @staticmethod
    def _response_id(row: dict) -> str:
        return f"{row['model_id']}|{row['case_id']}|{row['condition_id']}"

    def _iter_responses(self, path: Path):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _build_case_data(self, resp: dict, base: dict, evals: dict) -> dict:
        b = base.get(resp["case_id"], {})
        ev = evals.get((resp["case_id"], resp["condition_id"]), {})
        return {
            "response_id": self._response_id(resp),
            "case_id": resp["case_id"],
            "condition_id": resp["condition_id"],
            "attack_family": resp.get("attack_family"),
            "target_model_id": resp["model_id"],
            "benign_prompt": b.get("clinical_prompt", ""),
            "clinical_prompt": b.get("clinical_prompt", ""),
            "prompt": ev.get("prompt", ""),
            "ground_truth": b.get("ground_truth", ""),
            "response_text": resp.get("response_text", ""),
        }


def score_responses(
    responses_paths: List[Path],
    base_cases_csv: Path,
    eval_cases_jsonl: Path,
    scores_path: Path,
    judges: List[JudgeWrapper],
    response_limit: Optional[int] = None,
) -> dict:
    return MultiJudgeScorer(
        responses_paths=responses_paths,
        base_cases_csv=base_cases_csv,
        eval_cases_jsonl=eval_cases_jsonl,
        scores_path=scores_path,
        judges=judges,
        response_limit=response_limit,
    ).run()
