"""Batch runner: model × eval_cases.jsonl → responses.jsonl.

Resumable and idempotent. The runner reads the existing responses file (if
any) and skips (case_id, condition_id, model_id) triples that already have
an ``ok`` response, so killed runs can be resumed by re-invoking with the
same arguments.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from models.wrappers import ModelRequest, ModelWrapper, build_wrapper
from models.wrappers.base import ERROR_OK, hash_image, hash_prompt


@dataclasses.dataclass(frozen=True)
class EvalCaseRow:
    case_id: str
    condition_id: str
    attack_family: str
    attack_variant: str
    prompt: str
    image_path: str
    perturbed_image_path: Optional[str]
    expected_safe_behavior: str
    template_id: str
    template_hash: str

    @property
    def effective_image(self) -> str:
        """Visual variants (C7/C8) point to a perturbed image."""
        return self.perturbed_image_path or self.image_path

    @classmethod
    def from_jsonl_dict(cls, d: dict) -> "EvalCaseRow":
        return cls(
            case_id=d["case_id"],
            condition_id=d["condition_id"],
            attack_family=d["attack_family"],
            attack_variant=d["attack_variant"],
            prompt=d["prompt"],
            image_path=d["image_path"],
            perturbed_image_path=d.get("perturbed_image_path"),
            expected_safe_behavior=d.get("expected_safe_behavior", ""),
            template_id=d.get("template_id", ""),
            template_hash=d.get("template_hash", ""),
        )


def load_eval_cases(path: Path,
                    case_limit: Optional[int] = None,
                    conditions: Optional[List[str]] = None,
                    case_ids: Optional[set] = None) -> Iterator[EvalCaseRow]:
    """Stream eval cases from a JSONL file with optional filters."""
    seen_cases = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if conditions and d["condition_id"] not in conditions:
                continue
            if case_ids is not None and d["case_id"] not in case_ids:
                continue
            if case_limit is not None and len(seen_cases) >= case_limit:
                if d["case_id"] not in seen_cases:
                    continue
            seen_cases.add(d["case_id"])
            yield EvalCaseRow.from_jsonl_dict(d)


def _existing_keys(responses_path: Path, model_id: str,
                   only_ok: bool = True) -> set:
    """Read responses file and return the set of (case_id, condition_id) for
    which this model already has an ``ok`` response (for idempotent resume).
    """
    if not responses_path.exists():
        return set()
    keys: set = set()
    with responses_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("model_id") != model_id:
                continue
            if only_ok and d.get("error_status") != ERROR_OK:
                continue
            keys.add((d["case_id"], d["condition_id"]))
    return keys


@dataclasses.dataclass
class BatchRunner:
    eval_cases_path: Path
    responses_path: Path
    case_limit: Optional[int] = None
    conditions: Optional[List[str]] = None
    case_ids: Optional[set] = None
    flush_every: int = 1
    print_progress_every: int = 10
    resume: bool = True

    def run(self, wrapper: ModelWrapper) -> dict:
        self.responses_path.parent.mkdir(parents=True, exist_ok=True)

        existing = _existing_keys(self.responses_path, wrapper.model_id) if self.resume else set()
        if existing:
            print(f"[{wrapper.model_id}] resuming — {len(existing)} responses already complete",
                  file=sys.stderr, flush=True)

        rows = list(load_eval_cases(
            self.eval_cases_path,
            case_limit=self.case_limit,
            conditions=self.conditions,
            case_ids=self.case_ids,
        ))
        to_run = [r for r in rows if (r.case_id, r.condition_id) not in existing]
        print(f"[{wrapper.model_id}] target_rows={len(rows)} to_run={len(to_run)}",
              file=sys.stderr, flush=True)

        ok = err = 0
        t0 = time.monotonic()
        with self.responses_path.open("a", encoding="utf-8") as out:
            for i, row in enumerate(to_run, 1):
                req = ModelRequest(
                    case_id=row.case_id,
                    condition_id=row.condition_id,
                    attack_family=row.attack_family,
                    dataset="mimic_cxr_vqa",
                    prompt=row.prompt,
                    image_path=row.effective_image,
                )
                resp = wrapper.generate(req)
                d = resp.to_jsonl_dict()
                # Add eval-case provenance
                d["attack_variant"] = row.attack_variant
                d["template_id"] = row.template_id
                d["template_hash"] = row.template_hash
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                if i % self.flush_every == 0:
                    out.flush()

                if resp.error_status == ERROR_OK:
                    ok += 1
                else:
                    err += 1
                if i % self.print_progress_every == 0 or i == len(to_run):
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed else 0
                    eta = (len(to_run) - i) / rate if rate else 0
                    print(f"[{wrapper.model_id}] {i}/{len(to_run)} ok={ok} err={err} "
                          f"rate={rate:.2f}/s eta={eta:.0f}s",
                          file=sys.stderr, flush=True)

        elapsed = time.monotonic() - t0
        return {
            "model_id": wrapper.model_id,
            "rows_attempted": len(to_run),
            "ok": ok,
            "err": err,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_row": elapsed / max(1, len(to_run)),
        }


def run_model_on_cases(
    model_id: str,
    eval_cases_path: Path,
    responses_path: Path,
    case_limit: Optional[int] = None,
    conditions: Optional[List[str]] = None,
) -> dict:
    """Convenience entrypoint."""
    wrapper = build_wrapper(model_id)
    runner = BatchRunner(
        eval_cases_path=eval_cases_path,
        responses_path=responses_path,
        case_limit=case_limit,
        conditions=conditions,
    )
    return runner.run(wrapper)
