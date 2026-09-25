"""
Memory bank for query-level TTS experience (read-only at inference; built offline).

Two tiers:
  * GENERAL experience  -- fixed policy priors that won across every sweep
    (query-independent), in `general.json`.
  * QUERY-SPECIFIC experience -- soft, retrieved-by-similarity preferences for the
    knobs that are accuracy-decisive with no dominant value, in
    `query_experience.jsonl` (one row per past (query, budget) run).

Retrieval is HYBRID + BUDGET-AWARE: a new query is embedded (sentence-transformers)
and tagged (LLM self-generated + self-verified); candidates are filtered/weighted by
tag compatibility + budget proximity, then ranked by embedding cosine similarity.
The top-k neighbours' winning choices are aggregated into a soft preference used to
override ONLY the query-specific fields of a Tier-1-seeded Config.

This module is read-only.  Building the bank lives in build_memory_bank.py.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------- #
# Tier 1: general experience (fixed knobs that won in every sweep)
# ---------------------------------------------------------------------------- #
GENERAL_DEFAULTS: Dict[str, Any] = {
    "global": {
        # winners that were robust across all queries / sweeps
        "stop_policy": "composite",
        "width_depth_policy": "depth_biased",     # depth-leaning; always_width never preferred
        "route_uncertainty_hi": 0.6,              # width trigger high (0.5 never per-query best)
        "aggregation_policy": "plurality",         # beats the old llm_fusion default, cheaper
        "temp_policy": "fixed",                    # closed-loop temp is a no-op for weak models
        "equivalence_policy": "auto",
        "prune_policy": "none",
        "max_iters": 6,
        "max_nodes": 24,
    },
    # per-domain nudges (merged over global)
    "per_domain": {
        "math": {"aggregation_policy": "plurality"},
        "mmlu": {"aggregation_policy": "plurality"},   # llm_fusion mildly better but plurality is cheap & close
    },
}

# knobs the bank is allowed to set per query (everything else stays a fixed prior)
QUERY_SPECIFIC_FIELDS = ("model", "init_width", "leaf_select_policy",
                         "aggregation_policy", "budget_policy")

DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


def difficulty_bucket(score: float) -> str:
    return DIFFICULTY_BUCKETS[min(2, int(score * 3))]


# ---------------------------------------------------------------------------- #
# Query embedding (sentence-transformers, lazy + cached)
# ---------------------------------------------------------------------------- #
class QueryEmbedder:
    """Wraps all-MiniLM-L6-v2; returns a unit-norm vector. Degrades to None if the
    package is unavailable (retrieval then falls back to tag/difficulty matching)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._ok: Optional[bool] = None

    def _ensure(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._ok = True
        except Exception as exc:                      # noqa: BLE001
            print(f"[memory_bank] embedder unavailable ({str(exc)[:80]}); "
                  f"falling back to tag/difficulty retrieval")
            self._ok = False
        return self._ok

    @lru_cache(maxsize=4096)
    def encode(self, text: str) -> Optional[Tuple[float, ...]]:
        if not self._ensure():
            return None
        v = self._model.encode(text or "", normalize_embeddings=True)
        return tuple(float(x) for x in v)


def cosine(a: Optional[Tuple[float, ...]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))                   # both are unit-norm


# ---------------------------------------------------------------------------- #
# Query tagging (LLM self-generated + self-verified)
# ---------------------------------------------------------------------------- #
_TAG_PROMPT = (
    "Analyse the following problem and describe it with tags. Reply with ONLY a JSON "
    "object with keys: difficulty (one of easy|medium|hard), problem_type (a short "
    "lowercase label, e.g. algebra, geometry, factual_recall, coding), "
    "needs_computation (true/false). No prose.\n\nProblem:\n{q}"
)
_VERIFY_PROMPT = (
    "Problem:\n{q}\n\nProposed tags: {tags}\n\nAre these tags correct? If any field is "
    "wrong, output the corrected JSON object (same keys); if all correct, output the "
    "same JSON. Reply with ONLY the JSON object."
)


def _parse_tags(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    return d


class QueryTagger:
    """One tagging call + one self-verification call, with a difficulty/domain fallback."""

    def __init__(self, model_name: str, llm_getter) -> None:
        self.model_name = model_name
        self._llm = llm_getter(model_name)

    async def tag(self, task_text: str, domain: str, difficulty_score: float) -> Dict[str, Any]:
        from swarm.llm.format import Message
        fallback = {"domain": domain,
                    "difficulty": difficulty_bucket(difficulty_score),
                    "problem_type": "unknown", "needs_computation": domain in ("math", "gsm8k"),
                    "_verified": False}
        try:
            raw = await self._agen([Message(role="user", content=_TAG_PROMPT.format(q=task_text[:1500]))])
            tags = _parse_tags(raw)
            if tags is None:
                return fallback
            raw2 = await self._agen([Message(role="user",
                                             content=_VERIFY_PROMPT.format(q=task_text[:1500], tags=json.dumps(tags)))])
            verified = _parse_tags(raw2)
            out = verified or tags
            out["domain"] = domain
            out.setdefault("difficulty", difficulty_bucket(difficulty_score))
            out["_verified"] = verified is not None
            return out
        except Exception as exc:                      # noqa: BLE001
            print(f"[memory_bank] tagging failed ({str(exc)[:80]}); using fallback")
            return fallback

    async def _agen(self, messages) -> str:
        out = await self._llm.agen(messages, max_tokens=120, temperature=0.0)
        return out[0] if isinstance(out, tuple) else str(out)


# ---------------------------------------------------------------------------- #
# The bank
# ---------------------------------------------------------------------------- #
class MemoryBank:
    def __init__(self, general: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
        self.general = general
        self.rows = rows

    # -- loading ------------------------------------------------------------- #
    @classmethod
    def load(cls, bank_dir: str) -> "MemoryBank":
        gpath = os.path.join(bank_dir, "general.json")
        rpath = os.path.join(bank_dir, "query_experience.jsonl")
        general = json.load(open(gpath)) if os.path.exists(gpath) else GENERAL_DEFAULTS
        rows = []
        if os.path.exists(rpath):
            rows = [json.loads(l) for l in open(rpath, encoding="utf-8") if l.strip()]
        return cls(general, rows)

    # -- Tier 1 -------------------------------------------------------------- #
    def general_for(self, domain: str) -> Dict[str, Any]:
        merged = dict(self.general.get("global", {}))
        merged.update(self.general.get("per_domain", {}).get(domain, {}))
        return merged

    # -- Tier 2 retrieval (hybrid, budget-aware) ----------------------------- #
    def retrieve(self, domain: str, tags: Dict[str, Any],
                 embedding: Optional[Tuple[float, ...]], budget: float,
                 k: int = 8) -> List[Dict[str, Any]]:
        scored = []
        for r in self.rows:
            if r.get("domain") != domain:                       # tag filter: same domain
                continue
            tag_bonus = 0.0
            if r.get("tags", {}).get("difficulty") == tags.get("difficulty"):
                tag_bonus += 0.15
            if r.get("tags", {}).get("problem_type") == tags.get("problem_type"):
                tag_bonus += 0.10
            # budget proximity in [0,1]
            bl = float(r.get("budget_level", budget) or budget)
            bud = 1.0 - min(1.0, abs(bl - budget) / max(budget, 1e-6))
            sim = cosine(embedding, r.get("embedding"))
            r_score = sim + tag_bonus + 0.15 * bud
            scored.append((r_score, r))
        scored.sort(key=lambda t: -t[0])
        return [r for _, r in scored[:k]]

    def soft_prefs(self, neighbours: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate neighbours' winning choices into an accuracy-weighted soft-argmax
        per query-specific field (+ the distribution, for transparency)."""
        prefs: Dict[str, Any] = {}
        for field in QUERY_SPECIFIC_FIELDS:
            weight: Dict[Any, float] = {}
            for r in neighbours:
                val = r.get("chosen", {}).get(field)
                if val is None:
                    continue
                w = 0.5 + float(r.get("acc", 0.0))            # prefer high-accuracy neighbours
                weight[val] = weight.get(val, 0.0) + w
            if weight:
                total = sum(weight.values())
                best = max(weight, key=weight.get)
                prefs[field] = {"value": best,
                                "dist": {str(k): round(v / total, 3) for k, v in weight.items()},
                                "n": len(neighbours)}
        return prefs

    # -- public: produce Config overrides for a query ------------------------ #
    def config_overrides(self, domain: str, tags: Dict[str, Any],
                         embedding: Optional[Tuple[float, ...]], budget: float,
                         k: int = 8) -> Dict[str, Any]:
        overrides = dict(self.general_for(domain))               # Tier 1 general
        prefs = self.soft_prefs(self.retrieve(domain, tags, embedding, budget, k))
        for field, p in prefs.items():                           # Tier 2 overrides
            if field == "model":
                overrides["backbone"] = p["value"]
            elif field == "budget_policy":
                overrides["budget_policy"] = p["value"]
            else:
                overrides[field] = p["value"]
        overrides["_source"] = {"tier2_fields": list(prefs), "tags": tags}
        return overrides


# ---------------------------------------------------------------------------- #
# Applying overrides to a Config
# ---------------------------------------------------------------------------- #
_CONFIG_KEYS = None


def apply_overrides(cfg, overrides: Dict[str, Any]):
    """Return a copy of `cfg` (an experiment_adaptive_tts.Config) with the memory
    overrides applied. Only keys that are real Config fields are used."""
    global _CONFIG_KEYS
    if _CONFIG_KEYS is None:
        from dataclasses import fields
        _CONFIG_KEYS = {f.name for f in fields(cfg)}
    kw = {k: v for k, v in overrides.items()
          if k in _CONFIG_KEYS and not k.startswith("_")}
    return replace(cfg, **kw) if kw else cfg
