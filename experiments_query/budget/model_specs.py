"""
Approximate architecture specs (non-embedding params M, hidden size D, layers L)
for the OpenRouter model pool, for use with the FLOPs cost model in flops.py.

NOTE: params are APPROXIMATE non-embedding parameter counts (total params minus
the token-embedding matrix, which is negligible for large models). Hidden/layers
are the published architectures. Edit these if you need exact numbers — the cost
functions in flops.py are independent of these values.
"""

from __future__ import annotations

from typing import Dict, Sequence

from experiments_query.budget.flops import ModelSpec

# name -> ModelSpec(params M, hidden D, layers L)
MODEL_SPECS: Dict[str, ModelSpec] = {
    "meta-llama/llama-3.2-1b-instruct":  ModelSpec("meta-llama/llama-3.2-1b-instruct",  params=1.0e9,  hidden=2048, layers=16),
    "meta-llama/llama-3.2-3b-instruct":  ModelSpec("meta-llama/llama-3.2-3b-instruct",  params=2.8e9,  hidden=3072, layers=28),
    "google/gemma-3-4b-it":              ModelSpec("google/gemma-3-4b-it",              params=3.4e9,  hidden=2560, layers=34),
    "qwen/qwen-2.5-7b-instruct":         ModelSpec("qwen/qwen-2.5-7b-instruct",         params=7.0e9,  hidden=3584, layers=28),
    "meta-llama/llama-3.1-8b-instruct":  ModelSpec("meta-llama/llama-3.1-8b-instruct",  params=7.5e9,  hidden=4096, layers=32),
    "google/gemma-3-12b-it":             ModelSpec("google/gemma-3-12b-it",             params=11.0e9, hidden=3840, layers=48),
    "google/gemma-3-27b-it":             ModelSpec("google/gemma-3-27b-it",             params=25.0e9, hidden=4608, layers=62),
    "meta-llama/llama-3.1-70b-instruct": ModelSpec("meta-llama/llama-3.1-70b-instruct", params=69.0e9, hidden=8192, layers=80),
    "qwen/qwen-2.5-72b-instruct":        ModelSpec("qwen/qwen-2.5-72b-instruct",        params=71.0e9, hidden=8192, layers=80),
    "qwen/qwen3-8b":                     ModelSpec("qwen/qwen3-8b",                     params=7.5e9,  hidden=4096, layers=36),
    "qwen/qwen3-14b":                    ModelSpec("qwen/qwen3-14b",                    params=14.0e9, hidden=5120, layers=40),
    "qwen/qwen3-32b":                    ModelSpec("qwen/qwen3-32b",                    params=31.0e9, hidden=5120, layers=64),
}


def get_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name]


def smallest_model(specs: Sequence[ModelSpec] = None) -> ModelSpec:
    """The smallest model in the pool (by parameter count) — the budget unit."""
    pool = list(specs) if specs is not None else list(MODEL_SPECS.values())
    return min(pool, key=lambda m: m.params)
