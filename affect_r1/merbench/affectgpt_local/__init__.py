"""Local replacements for AffectGPT modules required by MERBench evaluation."""

from __future__ import annotations

from .datasets import build_dataset_map
from .llm_utils import (
    extract_openset_batchcalling,
    func_read_batch_calling_model,
    openset_to_sentiment_batchcalling,
)
from .wheel_metrics import hitrate_metric_calculation, wheel_metric_calculation


def build_local_modules():
    """Return a dict mirroring the original AffectGPT module contract."""
    dataset_map = build_dataset_map()
    return {
        "dataset_map": dataset_map,
        "extract_openset": extract_openset_batchcalling,
        "openset_to_sentiment": openset_to_sentiment_batchcalling,
        "wheel_metric": wheel_metric_calculation,
        "hitrate_metric": hitrate_metric_calculation,
        "load_llm": func_read_batch_calling_model,
    }

