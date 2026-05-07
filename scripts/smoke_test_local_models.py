"""Phase 3 Go/No-Go 3 driver — smoke-test every local wrapper on real GPU.

Run on a machine with the right HF auth + GPU memory for each model.
Verifies (a) the wrapper loads, (b) generation succeeds on a small benign
prompt, and (c) the wrapper is *image-aware* (responses differ when the
image differs — catches silent image-drop, the primary Phase 3 risk).

Usage:
    python scripts/smoke_test_local_models.py
    python scripts/smoke_test_local_models.py --models chexone_4b_instruct medgemma_4b_it
    python scripts/smoke_test_local_models.py --image-a path/a.jpg --image-b path/b.jpg
    python scripts/smoke_test_local_models.py --output smoke_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

# Make ``src`` importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.wrappers import build_wrapper, list_registered  # noqa: E402

DEFAULT_PROMPT = "Briefly describe the most prominent feature visible in this image."


def make_default_test_images(td: Path) -> tuple:
    """Generate two visibly-different small RGB images for image-awareness test."""
    from PIL import Image

    a = td / "smoke_a.jpg"
    b = td / "smoke_b.jpg"
    Image.new("RGB", (224, 224), color=(20, 20, 20)).save(a)         # dark
    Image.new("RGB", (224, 224), color=(230, 230, 230)).save(b)      # light
    return str(a), str(b)


def run_one(model_id: str, image_a: str, image_b: str, prompt: str) -> dict:
    print(f"\n=== {model_id} ===", flush=True)
    result = {"model_id": model_id, "load_ok": False, "generate_ok": False,
              "image_aware": None, "error": None}
    try:
        w = build_wrapper(model_id)
        print(f"  hf_repo: {w.hf_repo}  family: {w.model_family}", flush=True)
        smoke = w.smoke_test(prompt, image_a, image_b)
        result["load_ok"] = True
        result["generate_ok"] = smoke["error_a"] == "ok" and smoke["error_b"] == "ok"
        result["image_aware"] = smoke["image_actually_consumed"]
        result.update({
            "response_a_len": smoke["response_a_len"],
            "response_b_len": smoke["response_b_len"],
            "error_a": smoke["error_a"],
            "error_b": smoke["error_b"],
        })
        for k in ("generate_ok", "image_aware"):
            print(f"  {k}: {result[k]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
        traceback.print_exc()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model_ids to test. Defaults to all registered locals.")
    ap.add_argument("--image-a", type=str, default=None)
    ap.add_argument("--image-b", type=str, default=None)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--output", type=str, default=None,
                    help="Optional JSONL output for results.")
    args = ap.parse_args()

    targets = args.models or list_registered()
    print(f"Smoke-testing {len(targets)} models: {targets}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if args.image_a and args.image_b:
            ia, ib = args.image_a, args.image_b
        else:
            ia, ib = make_default_test_images(td)

        results = [run_one(m, ia, ib, args.prompt) for m in targets]

    print("\n=== Summary ===")
    print(f"{'model_id':<28} {'load':<6} {'gen':<6} {'image_aware':<12} error")
    print("-" * 80)
    failures = 0
    for r in results:
        load = "OK" if r["load_ok"] else "FAIL"
        gen = "OK" if r["generate_ok"] else ("--" if not r["load_ok"] else "FAIL")
        aware = str(r["image_aware"]) if r["image_aware"] is not None else "--"
        err = (r["error"] or "")[:40]
        print(f"{r['model_id']:<28} {load:<6} {gen:<6} {aware:<12} {err}")
        if not (r["load_ok"] and r["generate_ok"] and r["image_aware"]):
            failures += 1

    if args.output:
        Path(args.output).write_text("\n".join(json.dumps(r) for r in results) + "\n")
        print(f"\nResults written to {args.output}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
