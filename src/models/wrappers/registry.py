"""Factory mapping ``model_id`` (as in protocol/model_list.yaml) → wrapper.

Use ``build_wrapper("medgemma_27b_it")`` rather than instantiating wrappers
by hand so the runner can iterate over the lineup without knowing class
locations.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import ModelWrapper

# Registered model_id → constructor.
# Every constructor must take no required arguments (defaults baked in).
_REGISTRY: Dict[str, Callable[[], ModelWrapper]] = {}


def register(model_id: str, constructor: Callable[[], ModelWrapper]) -> None:
    if model_id in _REGISTRY:
        raise ValueError(f"model_id already registered: {model_id}")
    _REGISTRY[model_id] = constructor


def build_wrapper(model_id: str) -> ModelWrapper:
    if model_id not in _REGISTRY:
        raise KeyError(
            f"Unknown model_id: {model_id}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_id]()


def list_registered() -> list:
    return sorted(_REGISTRY.keys())


# ---------- Default registrations (open-weight / local models) ----------
def _register_locals() -> None:
    from .chexone import CheXOneWrapper
    from .llama_vision import Llama4ScoutWrapper
    from .llava_med import LlavaMedWrapper
    from .medgemma import MedGemmaWrapper
    from .qwen_vl import Qwen36VLWrapper

    register("chexone_4b_reasoning", lambda: CheXOneWrapper(mode="reasoning"))
    register("chexone_4b_instruct",  lambda: CheXOneWrapper(mode="instruct"))
    register("medgemma_4b_it",       lambda: MedGemmaWrapper(size="4b"))
    register("medgemma_27b_it",      lambda: MedGemmaWrapper(size="27b"))
    register("llava_med_v1_5",       lambda: LlavaMedWrapper())
    register("qwen3_6_27b",          lambda: Qwen36VLWrapper())
    register("llama_4_scout",        lambda: Llama4ScoutWrapper())


_register_locals()
