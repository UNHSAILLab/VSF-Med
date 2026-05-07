"""Uniform wrapper contract for VSF-Med v2 evaluation targets.

Every concrete model wrapper subclasses :class:`ModelWrapper` and emits a
:class:`ModelResponse` with the fields enumerated in
``protocol/model_list.yaml#logging_required_fields``. The ``smoke_test``
helper exists specifically to catch the silent-image-drop class of bugs that
the EXPERIMENT_TO_DO plan flags as a primary risk for Phase 3.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional

WRAPPER_API_VERSION = "1.0"

# Canonical column order for data/results/full_responses.jsonl
RESPONSE_JSONL_FIELDS = (
    "case_id",
    "condition_id",
    "model_id",
    "model_provider",
    "model_snapshot",
    "model_family",
    "wrapper_version",
    "wrapper_api_version",
    "dataset",
    "attack_family",
    "prompt_hash",
    "image_hash",
    "response_text",
    "error_status",
    "latency_seconds",
    "input_token_count",
    "output_token_count",
    "input_image_format",
    "temperature",
    "top_p",
    "max_output_tokens",
    "timestamp",
)


@dataclasses.dataclass(frozen=True)
class DecodingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    seed: Optional[int] = 20260505


@dataclasses.dataclass(frozen=True)
class ModelRequest:
    case_id: str
    condition_id: str
    attack_family: str
    dataset: str
    prompt: str
    image_path: Optional[str] = None
    multi_turn_history: Optional[tuple] = None  # tuple-of-dicts for hashability
    decoding: DecodingParams = dataclasses.field(default_factory=DecodingParams)


ERROR_OK = "ok"
ERROR_RATE_LIMITED = "rate_limited"
ERROR_IMAGE_REJECTED = "image_rejected"
ERROR_REFUSAL = "refusal"
ERROR_TIMEOUT = "timeout"
ERROR_OTHER = "other"


@dataclasses.dataclass
class ModelResponse:
    case_id: str
    condition_id: str
    model_id: str
    model_provider: str
    model_snapshot: str
    model_family: str
    wrapper_version: str
    response_text: str
    error_status: str
    latency_seconds: float
    timestamp: str
    dataset: Optional[str] = None
    attack_family: Optional[str] = None
    prompt_hash: Optional[str] = None
    image_hash: Optional[str] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    input_image_format: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    wrapper_api_version: str = WRAPPER_API_VERSION
    raw_response: Optional[dict] = None

    def to_jsonl_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("raw_response", None)
        return {k: d.get(k) for k in RESPONSE_JSONL_FIELDS}


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def hash_image(image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ModelWrapper(abc.ABC):
    """Abstract base for every evaluation target.

    Subclasses set the four identity attributes in ``__init__`` and implement
    ``_call``. The base class handles timing, hashing, error normalization,
    and JSONL packaging.
    """

    model_id: str
    model_provider: str
    model_snapshot: str
    model_family: str  # frontier | medical_specialist | open_generalist
    wrapper_version: str = "0.1.0"
    input_image_format: str = "png_or_jpeg_b64"

    @abc.abstractmethod
    def _call(self, request: ModelRequest) -> dict:
        """Provider-specific call. Returns a raw dict with at minimum:

        ``text`` (str), ``input_tokens`` (int|None), ``output_tokens`` (int|None),
        ``error_status`` (str, one of ERROR_*).
        """
        raise NotImplementedError

    def generate(self, request: ModelRequest) -> ModelResponse:
        start = time.monotonic()
        ts = now_iso()
        try:
            raw = self._call(request)
            err = raw.get("error_status", ERROR_OK)
            text = raw.get("text", "")
            in_tok = raw.get("input_tokens")
            out_tok = raw.get("output_tokens")
        except Exception as exc:  # noqa: BLE001 — we deliberately normalize all failures
            raw = {"exception": repr(exc)}
            err = ERROR_OTHER
            text = ""
            in_tok = None
            out_tok = None
        latency = time.monotonic() - start

        return ModelResponse(
            case_id=request.case_id,
            condition_id=request.condition_id,
            model_id=self.model_id,
            model_provider=self.model_provider,
            model_snapshot=self.model_snapshot,
            model_family=self.model_family,
            wrapper_version=self.wrapper_version,
            response_text=text,
            error_status=err,
            latency_seconds=latency,
            timestamp=ts,
            dataset=request.dataset,
            attack_family=request.attack_family,
            prompt_hash=hash_prompt(request.prompt),
            image_hash=hash_image(request.image_path),
            input_token_count=in_tok,
            output_token_count=out_tok,
            input_image_format=self.input_image_format,
            temperature=request.decoding.temperature,
            top_p=request.decoding.top_p,
            max_output_tokens=request.decoding.max_output_tokens,
            raw_response=raw,
        )

    def smoke_test(
        self,
        prompt: str,
        image_path_a: str,
        image_path_b: str,
        case_id: str = "smoke",
    ) -> dict:
        """Verify the wrapper actually consumes the image.

        Calls the model twice on the same prompt with two different images.
        If responses are byte-identical, the wrapper is silently dropping
        the image. Plan flags this as the primary Phase 3 failure mode.
        """
        req_a = ModelRequest(
            case_id=case_id, condition_id="C1_benign", attack_family="benign",
            dataset="smoke", prompt=prompt, image_path=image_path_a,
        )
        req_b = dataclasses.replace(req_a, image_path=image_path_b)
        resp_a = self.generate(req_a)
        resp_b = self.generate(req_b)
        identical = resp_a.response_text.strip() == resp_b.response_text.strip()
        return {
            "model_id": self.model_id,
            "responses_identical": identical,
            "image_actually_consumed": not identical,
            "response_a_len": len(resp_a.response_text),
            "response_b_len": len(resp_b.response_text),
            "error_a": resp_a.error_status,
            "error_b": resp_b.error_status,
        }


def write_responses_jsonl(responses: Iterable[ModelResponse], output_path: str) -> int:
    """Append responses to a JSONL file. Returns count written."""
    n = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r.to_jsonl_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
