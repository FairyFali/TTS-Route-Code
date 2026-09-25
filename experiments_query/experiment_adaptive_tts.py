"""
Adaptive, query-specific Test-Time Scaling (TTS) graph construction.

Given a query, a fixed model backbone, and a MAX compute budget, this script
*grows* a TTS graph one node at a time.  It starts from a configurable number of
independent samples (initial width) and then, at each iteration, inspects the
current leaf outputs and decides to

    (a) EXPAND IN WIDTH  -- add K new independent parallel samples, or
    (b) EXPAND IN DEPTH  -- pick one/more promising leaves (no successors yet)
                            and generate refining child nodes from them, or
    (c) STOP.

The decision uses label-free signals over the current leaves -- quality (self
consistency), diversity, uncertainty, agreement, and marginal utility per unit
cost.  Every knob (initial-width policy, width/depth policy, width-increment
policy, leaf-selection policy, aggregation policy, stopping policy, thresholds,
models, budget, temperatures, seeds, output paths) is configurable.

Unlike experiment_dynamic_tts.py (which runs whole *fixed archetypes* at growing
sizes), here the graph is built INCREMENTALLY at the node level: existing nodes'
outputs are computed once and frozen, and new nodes are attached to the live
graph.  This is what makes the construction genuinely adaptive.

The adaptive graph is compared against fixed PARALLEL / SEQUENTIAL / WIDE-TREE /
DEEP-TREE baselines under the same budget, and per-query trajectories plus a
dataset-level report (and plots) are written out.  Finally, reusable query-level
"memory" (preferred initial width, width-vs-depth tendency, promising-leaf
characteristics, effective stopping conditions per domain) is exported so a
future run can warm-start construction for similar queries.

Cost currency
-------------
1 unit = one direct inference of LLaMA-3.2-1B on THIS query (query-adaptive
lengths), identical to the other experiments_query scripts.  Node cost uses the
measured prompt/completion tokens through the FLOPs model in experiments_query.budget.

Usage
-----
    python experiments_query/experiment_adaptive_tts.py \
        --domains math --limit 8 --max-budget 60 \
        --backbone meta-llama/llama-3.2-1b-instruct \
        --width-depth-policy uncertainty_routed --stop-policy composite

    python experiments_query/experiment_adaptive_tts.py --analyze-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from swarm.llm import LLMRegistry
from swarm.llm.format import Message
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry

from experiments_query.budget import MODEL_SPECS, Task, node_cost, unit_cost
from experiments_query.run_tts import _design_num_nodes
from experiments_query.experiment_tts_preference import (
    load_datasets, query_task, score, UNIT_MODEL, SPLITS, DOMAIN_OUTPUT,
)

try:
    from scipy import stats as _scipy_stats
except Exception:                                    # optional; t-test degrades gracefully
    _scipy_stats = None

logger = logging.getLogger("adaptive_tts")


# =========================================================================== #
# Configuration
# =========================================================================== #
@dataclass
class Config:
    # --- models / domains / budget ------------------------------------------ #
    backbone: str = "meta-llama/llama-3.2-1b-instruct"
    aggregator: Optional[str] = None                 # None -> same as backbone
    domains: Tuple[str, ...] = ("math",)
    limit: int = 8
    max_budget: float = 60.0                          # in 1B-inference units
    max_iters: int = 8                                # hard cap on grow iterations
    max_nodes: int = 24                               # hard cap on total agent nodes

    # --- generation --------------------------------------------------------- #
    max_tokens: int = 1024
    width_temp: float = 0.7                           # independent samples -> diverse
    depth_temp: float = 0.5                           # refinement -> more focused
    agg_temp: float = 0.3

    # --- policies (names resolved against the registries below) ------------- #
    init_width_policy: str = "difficulty"             # fixed|budget_frac|difficulty
    init_width: int = 3                               # k0 for the "fixed" policy
    init_width_cap: int = 6
    width_depth_policy: str = "uncertainty_routed"    # uncertainty_routed|diversity_gated|alternate|always_width|always_depth
    width_count_policy: str = "uncertainty_scaled"    # fixed|uncertainty_scaled|budget_fill
    width_increment: int = 2
    leaf_select_policy: str = "composite"             # best_quality|most_uncertain|most_novel|composite
    depth_leaves_per_iter: int = 1                     # base leaves per depth step (fixed policy)
    expansion_width_policy: str = "fixed"             # fixed|budget_frac|uncertainty_scaled (leaves per depth step)
    expansion_frac: float = 0.25                       # budget_frac: leaves ~ frac * affordable nodes (proportion of budget)
    aggregation_policy: str = "llm_fusion"            # llm_fusion|plurality|best_consistency
    stop_policy: str = "composite"                    # composite|budget_only|agreement|uncertainty|diversity_sat|marginal

    # --- decision thresholds ------------------------------------------------ #
    agreement_stop: float = 0.85                      # stop if plurality share >= this
    uncertainty_stop: float = 0.15                    # stop if (1-agreement) <= this
    diversity_sat_delta: float = 0.05                 # stop if diversity barely changed
    min_marginal_per_cost: float = 0.01              # stop if agreement-gain/cost below this
    practical_min_improvement: float = 0.03           # "meaningful" improvement floor
    patience: int = 1                                 # allow this many stagnant iters first
    route_uncertainty_hi: float = 0.5                 # >= -> prefer WIDTH (explore)
    route_uncertainty_lo: float = 0.25                # <= -> prefer STOP-ish / DEPTH consolidate

    # --- leaf-selection composite weights ----------------------------------- #
    w_quality: float = 0.4
    w_uncertainty: float = 0.3
    w_novelty: float = 0.3

    # --- query-adaptive extensions (knobs 1-4) ------------------------------ #
    # 1. per-query budget allocation
    budget_policy: str = "fixed"            # fixed|difficulty|probe
    budget_min: float = 20.0
    budget_max: float = 100.0
    # 2. closed-loop temperature (+ optional n-sampling)
    temp_policy: str = "fixed"              # fixed|adaptive
    width_temp_hi: float = 1.0              # width temp when diversity < tau_div
    tau_div: float = 0.5
    width_n: int = 1                        # num_comps per width call (cheap width)
    # 3. query-adaptive aggregation
    agg_switch_agreement: float = 0.6       # >= -> cheap plurality; else llm_fusion
    strong_aggregator: Optional[str] = None
    agg_model_switch_uncertainty: float = 0.5
    # 4. answer-equivalence + leaf pruning
    equivalence_policy: str = "auto"        # auto|exact|numeric|code
    prune_policy: str = "none"              # none|beam
    beam_k: int = 4
    prune_min_agreement: float = 0.6        # only prune when agreement >= this (protect minority)

    # --- run control -------------------------------------------------------- #
    seed: int = 0
    concurrency: int = 8
    # Per-model concurrency caps, applied ON TOP of `concurrency`. Some pool models have a
    # much tighter per-model rate limit than the account's overall limit -- gemma-3-4b in
    # particular returns RateLimitError under load, and before the fix in NodeRunner.run
    # those failures were cached as empty zero-cost generations, silently corrupting results.
    # Throttling just that model is far cheaper than throttling the whole run.
    model_concurrency: Dict[str, int] = field(
        default_factory=lambda: {"google/gemma-3-4b-it": 4})
    outdir: str = "experiments_query/results/adaptive"
    cache_path: str = "experiments_query/results/adaptive/node_cache.json"
    run_baselines: bool = True
    make_plots: bool = True
    plot_queries: int = 6
    overwrite: bool = False
    # --- memory-bank warm-start (read-only) --------------------------------- #
    use_memory: bool = False
    memory_dir: str = "experiments_query/memory_bank"

    def resolved_aggregator(self) -> str:
        return self.aggregator or self.backbone


# =========================================================================== #
# Explicit incremental graph representation
# =========================================================================== #
@dataclass
class GNode:
    nid: int
    kind: str                       # "width" (independent) | "depth" (refine) | "agg"
    model: str
    depth: int
    parent: Optional[int]
    prompt: str
    output: str
    answer_key: str                 # label-free cluster key
    oracle_score: float             # uses the label (analysis only, NOT for decisions)
    prompt_tokens: int
    completion_tokens: int
    cost_units: float
    children: List[int] = field(default_factory=list)
    pruned: bool = False                    # beam-pruned leaf (excluded from leaves/aggregation)


class AdaptiveGraph:
    """A live, incrementally grown DAG of GNodes rooted at independent samples."""

    def __init__(self) -> None:
        self.nodes: Dict[int, GNode] = {}
        self._next = 0

    def add(self, node_kwargs: Dict[str, Any]) -> GNode:
        nid = self._next
        self._next += 1
        node = GNode(nid=nid, **node_kwargs)
        self.nodes[nid] = node
        if node.parent is not None:
            self.nodes[node.parent].children.append(nid)
        return node

    # -- views used by the policies ----------------------------------------- #
    def leaves(self) -> List[GNode]:
        """Answer-bearing leaves (no successors, not pruned), excluding the aggregator."""
        return [n for n in self.nodes.values()
                if n.kind != "agg" and not n.children and not n.pruned]

    def answer_nodes(self) -> List[GNode]:
        return [n for n in self.nodes.values() if n.kind != "agg"]

    def total_cost(self) -> float:
        return sum(n.cost_units for n in self.nodes.values())

    def width(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == "width")

    def max_depth(self) -> int:
        return max((n.depth for n in self.answer_nodes()), default=0)

    def structure(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.answer_nodes()),
            "width": self.width(),
            "max_depth": self.max_depth(),
            "edges": [[n.parent, n.nid] for n in self.nodes.values()
                      if n.parent is not None],
        }


# =========================================================================== #
# LLM node execution (cached, cost-aware, resumable)
# =========================================================================== #
_LLM_CACHE: Dict[str, Any] = {}


def _get_llm(model: str):
    if model not in _LLM_CACHE:
        _LLM_CACHE[model] = LLMRegistry.get(model)
    return _LLM_CACHE[model]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]


async def _agen(model: str, messages: List[Message], max_tokens: int,
                temperature: float, num_comps: int = 1):
    """Call the backbone; normalise the return.

    num_comps==1 -> (resp:str, p_tok, c_tok); num_comps>1 -> (resps:list[str], p_tok, c_tok)
    (p_tok/c_tok are group totals for n>1: prefill shared, decode summed)."""
    out = await _get_llm(model).agen(messages, max_tokens=max_tokens,
                                     temperature=temperature, num_comps=num_comps)
    if isinstance(out, tuple):
        resp = out[0]
        pt = out[1] if len(out) > 1 and out[1] is not None else 0
        ct = out[2] if len(out) > 2 and out[2] is not None else 0
        return resp, int(pt), int(ct)
    # bare list/str (no cost) -- shouldn't happen with the patched backend
    return out, 0, 0


class NodeRunner:
    """Runs (and caches) one LLM node; converts tokens -> normalised units."""

    def __init__(self, cfg: Config, cache: Dict[str, Any],
                 sem: asyncio.Semaphore) -> None:
        self.cfg = cfg
        self.cache = cache
        self.sem = sem
        self._new = 0
        self.failures = 0      # node calls that errored out; never cached, see run()
        self._model_caps = dict(getattr(cfg, "model_concurrency", None) or {})
        self._model_sems: Dict[str, asyncio.Semaphore] = {}

    def _model_sem(self, model: str) -> Optional[asyncio.Semaphore]:
        cap = self._model_caps.get(model)
        if not cap:
            return None
        sem = self._model_sems.get(model)
        if sem is None:
            sem = self._model_sems[model] = asyncio.Semaphore(cap)
        return sem

    def _kind_temp(self, kind: str) -> float:
        return {"width": self.cfg.width_temp, "depth": self.cfg.depth_temp,
                "agg": self.cfg.agg_temp}.get(kind, self.cfg.width_temp)

    async def run(self, qid: str, model: str, kind: str, depth: int,
                  sample_idx: int, messages: List[Message], task: Task,
                  unit_q: float, temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> Dict[str, Any]:
        temp = self._kind_temp(kind) if temperature is None else temperature
        mt = self.cfg.max_tokens if max_tokens is None else max_tokens
        mt_key = "" if max_tokens is None else f"|mt{max_tokens}"   # non-breaking cache key
        prompt_text = "\n".join(m.content for m in messages)
        key = f"{qid}|{model}|{kind}|d{depth}|s{sample_idx}|t{temp}{mt_key}|{_hash(prompt_text)}"
        if key in self.cache:
            return self.cache[key]
        failed = False

        async def _call():
            nonlocal failed
            async with self.sem:
                try:
                    return await _agen(model, messages, mt, temp)
                except Exception as exc:             # keep the trajectory alive
                    logger.warning("node call failed (%s): %s", key, str(exc)[:160])
                    failed = True
                    return "", 0, 0

        # Acquire the per-model slot first, so a throttled model queues on its own limit
        # instead of occupying a global slot while it waits.
        msem = self._model_sem(model)
        if msem is None:
            resp, pt, ct = await _call()
        else:
            async with msem:
                resp, pt, ct = await _call()
        spec = MODEL_SPECS.get(model)
        flops = node_cost(pt, ct, spec) if (spec and pt) else 0.0
        rec = {
            "output": resp, "prompt": prompt_text, "model": model,
            "prompt_tokens": pt, "completion_tokens": ct,
            "cost_units": (flops / unit_q) if unit_q else 0.0,
        }
        if failed:
            # NEVER cache a failed call. A rate-limit or transport error would otherwise be
            # frozen on disk as a legitimate empty generation costing 0, which downstream
            # scores as "wrong answer, free" -- silently corrupting every future run that
            # reuses this cache. Returning it uncached keeps the trajectory alive for this
            # attempt while letting a later run retry the node.
            self.failures += 1
            return rec
        self.cache[key] = rec
        self._new += 1
        if self._new % 25 == 0:
            _save_cache(self.cfg.cache_path, self.cache)
        return rec

    async def run_samples(self, qid: str, model: str, depth: int, sample_base: int,
                          messages: List[Message], task: Task, unit_q: float,
                          n: int, temperature: float,
                          max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """n independent width samples. n==1 -> one run(); n>1 -> a single num_comps=n
        call whose group cost is split evenly across the n cached sample records."""
        if n <= 1:
            return [await self.run(qid, model, "width", depth, sample_base,
                                   messages, task, unit_q, temperature, max_tokens)]
        mt = self.cfg.max_tokens if max_tokens is None else max_tokens
        mt_key = "" if max_tokens is None else f"|mt{max_tokens}"   # non-breaking cache key
        prompt_text = "\n".join(m.content for m in messages)
        keys = [f"{qid}|{model}|width|d{depth}|s{sample_base + i}|t{temperature}|n{n}{mt_key}|{_hash(prompt_text)}"
                for i in range(n)]
        if all(k in self.cache for k in keys):
            return [self.cache[k] for k in keys]
        async with self.sem:
            try:
                resps, pt, ct = await _agen(model, messages, mt,
                                            temperature, num_comps=n)
                if not isinstance(resps, list):
                    resps = [resps]
            except Exception as exc:
                logger.warning("multi-sample call failed (%s): %s", keys[0], str(exc)[:160])
                resps, pt, ct = [""] * n, 0, 0
        resps = (resps + [""] * n)[:n]
        spec = MODEL_SPECS.get(model)
        total_flops = node_cost(pt, ct, spec) if (spec and pt) else 0.0
        per_cost = (total_flops / unit_q / n) if (unit_q and n) else 0.0
        recs = []
        for i, k in enumerate(keys):
            rec = {"output": resps[i], "prompt": prompt_text, "model": model,
                   "prompt_tokens": pt // n, "completion_tokens": ct // n,
                   "cost_units": per_cost}
            self.cache[k] = rec
            recs.append(rec)
        self._new += 1
        if self._new % 25 == 0:
            _save_cache(self.cfg.cache_path, self.cache)
        return recs


# =========================================================================== #
# Prompt construction (reuses the domain prompt set, with safe fallbacks)
# =========================================================================== #
def _prompt_set(domain: str):
    return PromptSetRegistry.get(domain)


def _role_constraint(ps, domain: str, refine: bool) -> Tuple[str, str]:
    role = ps.get_role()
    constraint = ps.get_constraint()
    if domain == "math":
        constraint = ("Answer the question. Show your work, then output your "
                      "final answer as 'The answer is: \\boxed{...}'.")
    elif domain == "humaneval" and not refine:
        constraint = ("Write your solution as a single Python code block, e.g.\n"
                      "```python\n# code\n```")
    return role, constraint


def width_messages(ps, domain: str, task_text: str) -> List[Message]:
    role, constraint = _role_constraint(ps, domain, refine=False)
    prompt = ps.get_answer_prompt(question=task_text)
    return [Message(role="system", content=f"You are {role}. {constraint}"),
            Message(role="user", content=prompt)]


def depth_messages(ps, domain: str, task_text: str, parent_output: str) -> List[Message]:
    role, constraint = _role_constraint(ps, domain, refine=True)
    try:
        prompt = ps.get_answer_prompt_refine_last_answers(task_text, [parent_output])
    except Exception:
        prompt = (f"Question: {task_text}\n\nA prior attempt is below. Improve it "
                  f"and give the correct final answer.\n\n{parent_output}")
    return [Message(role="system", content=f"You are {role}. {constraint}"),
            Message(role="user", content=prompt)]


def fusion_messages(ps, domain: str, task_text: str,
                    leaf_outputs: List[str]) -> List[Message]:
    role, constraint = _role_constraint(ps, domain, refine=True)
    prompt = None
    for meth in ("get_select_best", "get_answer_prompt_refine_last_answers"):
        fn = getattr(ps, meth, None)
        if fn is not None:
            try:
                prompt = fn(task_text, leaf_outputs)
                break
            except Exception:
                prompt = None
    if prompt is None:
        joined = "\n\n".join(f"[Candidate {i + 1}]\n{o}"
                             for i, o in enumerate(leaf_outputs))
        prompt = (f"Question: {task_text}\n\nSeveral candidate solutions are "
                  f"below. Synthesise the single best final answer.\n\n{joined}")
    return [Message(role="system",
                    content=f"You are {role}. Aggregate carefully. {constraint}"),
            Message(role="user", content=prompt)]


# =========================================================================== #
# Answer keys (label-free clustering) and signals
# =========================================================================== #
def _normalize(s: str) -> str:
    return "".join(str(s).split()).strip("$ ").lower()


def _to_number(s: str) -> Optional[float]:
    """Parse a scalar answer to a float (handles fractions a/b and \\frac{a}{b}); None on failure."""
    import re
    t = str(s).strip().replace("\\!", "").replace("\\,", "").replace(",", "")
    t = t.replace("$", "").replace(" ", "")
    m = re.fullmatch(r"\\?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", t):
        a, b = t.split("/")
        try:
            return float(a) / float(b)
        except ZeroDivisionError:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _key_exact(domain, dataset, raw: str) -> str:
    ans = dataset.postprocess_answer(raw)
    return _normalize(ans) or "∅"


def _key_numeric(domain, dataset, raw: str) -> str:
    """Numeric equivalence: 1/2 == 0.5 == 0.500. Falls back to exact on parse failure."""
    try:
        ans = dataset.postprocess_answer(raw)
    except Exception:
        ans = raw
    num = _to_number(ans)
    if num is not None:
        return f"num:{round(num, 6):g}"
    return _normalize(ans) or "∅"


def _key_code(domain, dataset, raw: str) -> str:
    import re
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    body = m.group(1) if m else raw
    return "code:" + _hash(_normalize(body))


EQUIVALENCE = {"exact": _key_exact, "numeric": _key_numeric, "code": _key_code}
_AUTO_EQUIV = {"math": "numeric", "gsm8k": "numeric",
               "humaneval": "code", "livecodebench": "code"}


def answer_key(domain: str, dataset, raw: str, equiv: str = "auto") -> str:
    """Label-FREE cluster key for a leaf output (used for agreement/diversity).

    `equiv` selects the equivalence class: exact|numeric|code, or "auto" (per domain).
    """
    if not raw:
        return "∅"
    if equiv == "auto":
        equiv = _AUTO_EQUIV.get(domain, "exact")
    try:
        return EQUIVALENCE.get(equiv, _key_exact)(domain, dataset, raw)
    except Exception:
        return "∅"


def leaf_signals(keys: List[str]) -> Dict[str, float]:
    """Aggregate agreement / uncertainty / diversity / entropy over leaf keys."""
    n = len(keys)
    if n == 0:
        return {"n": 0, "agreement": 0.0, "uncertainty": 1.0,
                "diversity": 1.0, "entropy": 0.0, "plurality_key": "∅"}
    counts = Counter(keys)
    top_key, top = counts.most_common(1)[0]
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return {
        "n": n,
        "agreement": top / n,
        "uncertainty": 1.0 - top / n,
        "diversity": len(counts) / n,
        "entropy": entropy,
        "plurality_key": top_key,
    }


def per_leaf_scores(leaves: List[GNode]) -> Dict[int, Dict[str, float]]:
    """Per-leaf label-free descriptors used by the leaf-selection policies."""
    keys = [l.answer_key for l in leaves]
    counts = Counter(keys)
    n = len(leaves) or 1
    out: Dict[int, Dict[str, float]] = {}
    for l in leaves:
        consistency = counts[l.answer_key] / n            # plurality membership
        out[l.nid] = {
            "quality": consistency,                       # established-ness
            "novelty": 1.0 - consistency,                 # minority-ness
            "disagreement": 1.0 - consistency,
            "uncertainty": 1.0 - consistency,
            "length": len(l.output),
            # potential-to-improve: near-tie minority answers are worth refining
            "improvement_potential": (1.0 - consistency) * consistency * 4.0,
        }
    return out


# =========================================================================== #
# Policy registries
# =========================================================================== #
InitWidthFn = Callable[[Config, str, Task, float, random.Random], int]
INIT_WIDTH_POLICIES: Dict[str, InitWidthFn] = {}
WIDTH_DEPTH_POLICIES: Dict[str, Callable] = {}
WIDTH_COUNT_POLICIES: Dict[str, Callable] = {}
LEAF_SELECT_POLICIES: Dict[str, Callable] = {}
STOP_POLICIES: Dict[str, Callable] = {}
BUDGET_POLICIES: Dict[str, Callable] = {}


def _register(reg, name):
    def deco(fn):
        reg[name] = fn
        return fn
    return deco


# --- per-query budget-allocation policies ----------------------------------- #
# fn(cfg, domain, task, first_signals, orig_budget) -> effective per-query budget
def _difficulty_score(task: Task) -> float:
    """0..1 length-based difficulty proxy (longer prompt/target -> harder)."""
    return max(0.0, min(1.0, task.avg_input_len / 256.0 + task.avg_output_len / 1024.0))


@_register(BUDGET_POLICIES, "fixed")
def _bp_fixed(cfg, domain, task, first_sig, orig_budget):
    return orig_budget


@_register(BUDGET_POLICIES, "difficulty")
def _bp_difficulty(cfg, domain, task, first_sig, orig_budget):
    d = _difficulty_score(task)
    return cfg.budget_min + (cfg.budget_max - cfg.budget_min) * d


@_register(BUDGET_POLICIES, "probe")
def _bp_probe(cfg, domain, task, first_sig, orig_budget):
    # spend proportional to the initial-width samples' disagreement (a free probe):
    # confident (low-uncertainty / "moot") queries get budget_min, contested ones budget_max
    u = first_sig.get("uncertainty", 0.5)
    return cfg.budget_min + (cfg.budget_max - cfg.budget_min) * u


# --- initial-width policies ------------------------------------------------- #
def _per_node_units(cfg: Config, task: Task, unit_q: float) -> float:
    spec = MODEL_SPECS.get(cfg.backbone)
    return (unit_cost(spec, task) / unit_q) if (spec and unit_q) else 1.0


@_register(INIT_WIDTH_POLICIES, "fixed")
def _iw_fixed(cfg, domain, task, unit_q, rng):
    return max(1, min(cfg.init_width, cfg.init_width_cap))


@_register(INIT_WIDTH_POLICIES, "budget_frac")
def _iw_budget_frac(cfg, domain, task, unit_q, rng):
    per = _per_node_units(cfg, task, unit_q)
    k = int((0.4 * cfg.max_budget) / max(per, 1e-6))
    return max(1, min(k, cfg.init_width_cap))


@_register(INIT_WIDTH_POLICIES, "difficulty")
def _iw_difficulty(cfg, domain, task, unit_q, rng):
    # Longer / higher-output-budget queries start wider (more likely to be hard).
    hard = task.avg_input_len / 128.0 + task.avg_output_len / 256.0
    k = int(round(2 + hard))
    return max(2, min(k, cfg.init_width_cap))


# --- width-vs-depth policies ------------------------------------------------ #
# each returns ("width" | "depth" | "stop_hint", rationale:str)
@_register(WIDTH_DEPTH_POLICIES, "uncertainty_routed")
def _wd_uncertainty(cfg, sig, graph, hist, rng):
    u = sig["uncertainty"]
    if u >= cfg.route_uncertainty_hi:
        return "width", f"high uncertainty {u:.2f}>= {cfg.route_uncertainty_hi}: gather independent samples"
    if u <= cfg.route_uncertainty_lo:
        return "depth", f"low uncertainty {u:.2f}<= {cfg.route_uncertainty_lo}: consolidate leading answer via refinement"
    return "depth", f"moderate uncertainty {u:.2f}: refine promising leaves"


@_register(WIDTH_DEPTH_POLICIES, "diversity_gated")
def _wd_diversity(cfg, sig, graph, hist, rng):
    if sig["diversity"] >= 0.6 and sig["agreement"] < 0.6:
        return "width", f"diverse ({sig['diversity']:.2f}) & low agreement: widen to find consensus"
    return "depth", f"diversity {sig['diversity']:.2f} saturating: deepen best leaves"


@_register(WIDTH_DEPTH_POLICIES, "alternate")
def _wd_alternate(cfg, sig, graph, hist, rng):
    n_prev = sum(1 for h in hist if h.get("action") in ("width", "depth"))
    return ("width", "alternating: width") if n_prev % 2 == 0 else ("depth", "alternating: depth")


@_register(WIDTH_DEPTH_POLICIES, "always_width")
def _wd_width(cfg, sig, graph, hist, rng):
    return "width", "policy=always_width"


@_register(WIDTH_DEPTH_POLICIES, "always_depth")
def _wd_depth(cfg, sig, graph, hist, rng):
    return "depth", "policy=always_depth"


@_register(WIDTH_DEPTH_POLICIES, "depth_biased")
def _wd_depth_biased(cfg, sig, graph, hist, rng):
    """Refine by default (depth beats width for weak backbones); only widen when
    leaves are highly uncertain AND the graph is still narrow (bootstrap samples)."""
    if sig["uncertainty"] >= cfg.route_uncertainty_hi and graph.width() < 3:
        return "width", (f"high uncertainty {sig['uncertainty']:.2f} & narrow "
                         f"(w={graph.width()}): bootstrap a few independent samples")
    return "depth", f"depth-biased: refine promising leaf (uncertainty {sig['uncertainty']:.2f})"


# --- width-increment policies ----------------------------------------------- #
@_register(WIDTH_COUNT_POLICIES, "fixed")
def _wc_fixed(cfg, sig, remaining_units, per_node, rng):
    return max(1, cfg.width_increment)


@_register(WIDTH_COUNT_POLICIES, "uncertainty_scaled")
def _wc_uncertainty(cfg, sig, remaining_units, per_node, rng):
    # more uncertain -> add more parallel samples
    return max(1, int(round(cfg.width_increment * (0.5 + sig["uncertainty"]))))


@_register(WIDTH_COUNT_POLICIES, "budget_fill")
def _wc_budget_fill(cfg, sig, remaining_units, per_node, rng):
    k = int((0.3 * remaining_units) / max(per_node, 1e-6))
    return max(1, min(k, 2 * cfg.width_increment))


# --- leaf-selection policies (for depth expansion) -------------------------- #
@_register(LEAF_SELECT_POLICIES, "best_quality")
def _ls_best(cfg, leaves, pls, k, rng):
    return sorted(leaves, key=lambda l: -pls[l.nid]["quality"])[:k]


@_register(LEAF_SELECT_POLICIES, "most_uncertain")
def _ls_uncertain(cfg, leaves, pls, k, rng):
    return sorted(leaves, key=lambda l: -pls[l.nid]["uncertainty"])[:k]


@_register(LEAF_SELECT_POLICIES, "most_novel")
def _ls_novel(cfg, leaves, pls, k, rng):
    return sorted(leaves, key=lambda l: -pls[l.nid]["novelty"])[:k]


@_register(LEAF_SELECT_POLICIES, "composite")
def _ls_composite(cfg, leaves, pls, k, rng):
    def sc(l):
        s = pls[l.nid]
        return (cfg.w_quality * s["quality"]
                + cfg.w_uncertainty * s["uncertainty"]
                + cfg.w_novelty * s["improvement_potential"])
    return sorted(leaves, key=lambda l: -sc(l))[:k]


# --- stopping policies ------------------------------------------------------ #
# return (stop:bool, reason:str)
def _marginal_per_cost(hist) -> float:
    if len(hist) < 2:
        return float("inf")
    d_agree = hist[-1]["agreement"] - hist[-2]["agreement"]
    d_cost = max(hist[-1]["cum_cost"] - hist[-2]["cum_cost"], 1e-6)
    return d_agree / d_cost


@_register(STOP_POLICIES, "budget_only")
def _st_budget(cfg, sig, hist, graph, stagnation):
    if graph.total_cost() >= cfg.max_budget:
        return True, "max_budget"
    return False, ""


@_register(STOP_POLICIES, "agreement")
def _st_agreement(cfg, sig, hist, graph, stagnation):
    if sig["agreement"] >= cfg.agreement_stop:
        return True, "high_agreement"
    return _st_budget(cfg, sig, hist, graph, stagnation)


@_register(STOP_POLICIES, "uncertainty")
def _st_uncertainty(cfg, sig, hist, graph, stagnation):
    if sig["uncertainty"] <= cfg.uncertainty_stop:
        return True, "low_uncertainty"
    return _st_budget(cfg, sig, hist, graph, stagnation)


@_register(STOP_POLICIES, "diversity_sat")
def _st_diversity(cfg, sig, hist, graph, stagnation):
    if len(hist) >= 2 and abs(hist[-1]["diversity"] - hist[-2]["diversity"]) <= cfg.diversity_sat_delta \
            and sig["agreement"] >= 0.5:
        return True, "diversity_saturated"
    return _st_budget(cfg, sig, hist, graph, stagnation)


@_register(STOP_POLICIES, "marginal")
def _st_marginal(cfg, sig, hist, graph, stagnation):
    mpc = _marginal_per_cost(hist)
    if mpc < cfg.min_marginal_per_cost and stagnation > cfg.patience:
        return True, "low_marginal_gain_per_cost"
    return _st_budget(cfg, sig, hist, graph, stagnation)


@_register(STOP_POLICIES, "composite")
def _st_composite(cfg, sig, hist, graph, stagnation):
    if graph.total_cost() >= cfg.max_budget:
        return True, "max_budget"
    if graph.width() >= cfg.max_nodes or len(graph.answer_nodes()) >= cfg.max_nodes:
        return True, "max_nodes"
    if sig["agreement"] >= cfg.agreement_stop:
        return True, "high_agreement"
    if sig["uncertainty"] <= cfg.uncertainty_stop:
        return True, "low_uncertainty"
    if len(hist) >= 2:
        d_agree = hist[-1]["agreement"] - hist[-2]["agreement"]
        d_div = abs(hist[-1]["diversity"] - hist[-2]["diversity"])
        if d_agree < cfg.practical_min_improvement and d_div <= cfg.diversity_sat_delta:
            if stagnation >= cfg.patience:
                return True, "no_meaningful_improvement"
        if _marginal_per_cost(hist) < cfg.min_marginal_per_cost and stagnation >= cfg.patience:
            return True, "low_marginal_gain_per_cost"
    return False, ""


# =========================================================================== #
# Adaptive construction for one query
# =========================================================================== #
async def build_adaptive(cfg: Config, runner: NodeRunner, domain: str,
                         dataset, record, idx: int) -> Dict[str, Any]:
    qid = f"{domain}|{idx}"
    rng = random.Random(f"{cfg.seed}|{qid}")
    inputs = dataset.record_to_swarm_input(record)
    task_text = inputs["task"]
    task = query_task(domain, inputs)
    unit_q = unit_cost(MODEL_SPECS[UNIT_MODEL], task)
    ps = _prompt_set(domain)
    graph = AdaptiveGraph()
    hist: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, Any]] = []

    def oracle(raw: str) -> float:
        return score(domain, dataset, record, raw, inputs)

    def akey(raw: str) -> str:
        return answer_key(domain, dataset, raw, cfg.equivalence_policy)

    async def make_width_nodes(count: int, sample_base: int,
                               temperature: Optional[float] = None) -> None:
        msgs = width_messages(ps, domain, task_text)
        temp = (temperature if temperature is not None else cfg.width_temp)
        # batch into num_comps=width_n calls (cheap width); recs preserves order
        recs: List[Dict[str, Any]] = []
        base = sample_base
        while len(recs) < count:
            n = min(cfg.width_n, count - len(recs)) if cfg.width_n > 1 else 1
            batch = await runner.run_samples(qid, cfg.backbone, 0, base, msgs,
                                             task, unit_q, n, temp)
            recs.extend(batch)
            base += n
        for rec in recs[:count]:
            graph.add(dict(
                kind="width", model=cfg.backbone, depth=0, parent=None,
                prompt=rec["prompt"], output=rec["output"],
                answer_key=akey(rec["output"]),
                oracle_score=oracle(rec["output"]),
                prompt_tokens=rec["prompt_tokens"],
                completion_tokens=rec["completion_tokens"],
                cost_units=rec["cost_units"]))

    async def make_depth_nodes(parents: List[GNode]) -> None:
        jobs = []
        for p in parents:
            msgs = depth_messages(ps, domain, task_text, p.output)
            jobs.append((p, msgs))
        recs = await asyncio.gather(*[
            runner.run(qid, cfg.backbone, "depth", p.depth + 1, p.nid, m, task,
                       unit_q) for p, m in jobs])
        for (p, _), rec in zip(jobs, recs):
            graph.add(dict(
                kind="depth", model=cfg.backbone, depth=p.depth + 1, parent=p.nid,
                prompt=rec["prompt"], output=rec["output"],
                answer_key=akey(rec["output"]),
                oracle_score=oracle(rec["output"]),
                prompt_tokens=rec["prompt_tokens"],
                completion_tokens=rec["completion_tokens"],
                cost_units=rec["cost_units"]))

    def avg_node_cost() -> float:
        ans = graph.answer_nodes()
        return (graph.total_cost() / len(ans)) if ans else _per_node_units(cfg, task, unit_q)

    def affordable_count() -> int:
        """How many more agent nodes fit, reserving one aggregator call."""
        avg = avg_node_cost()
        budget_left = cfg.max_budget - avg - graph.total_cost()   # reserve ~1 agg node
        return max(0, int(budget_left / max(avg, 1e-6)))

    def maybe_prune(leaves: List[GNode]) -> List[int]:
        """Beam pruning (knob 4). Gated on high agreement to protect minority-correct
        answers (esp. MATH). Returns list of pruned nids."""
        if cfg.prune_policy != "beam" or len(leaves) <= cfg.beam_k:
            return []
        sig = leaf_signals([l.answer_key for l in leaves])
        if sig["agreement"] < cfg.prune_min_agreement:      # only prune when converging
            return []
        pls = per_leaf_scores(leaves)

        def sc(l):
            s = pls[l.nid]
            return (cfg.w_quality * s["quality"] + cfg.w_uncertainty * s["uncertainty"]
                    + cfg.w_novelty * s["improvement_potential"])
        ranked = sorted(leaves, key=lambda l: -sc(l))
        pruned = []
        for l in ranked[cfg.beam_k:]:
            l.pruned = True
            pruned.append(l.nid)
        return pruned

    # --- iteration 0: initial width (measure one, then fill within budget) --- #
    orig_budget = cfg.max_budget
    effective_budget = orig_budget
    try:
        k0 = INIT_WIDTH_POLICIES[cfg.init_width_policy](cfg, domain, task, unit_q, rng)
        await make_width_nodes(1, sample_base=0)                   # measure real cost first
        extra = min(k0 - 1, affordable_count(), cfg.max_nodes - graph.width())
        if extra > 0:
            await make_width_nodes(extra, sample_base=graph.width())

        # knob 1: set the per-query effective budget from a free probe (initial samples)
        first_sig = leaf_signals([l.answer_key for l in graph.leaves()])
        effective_budget = BUDGET_POLICIES[cfg.budget_policy](
            cfg, domain, task, first_sig, orig_budget)
        cfg.max_budget = effective_budget       # queries run sequentially; restored in finally

        stop_reason = "max_iters"
        stagnation = 0
        for it in range(cfg.max_iters):
            pruned_now = maybe_prune(graph.leaves())      # knob 4: beam pruning
            leaves = graph.leaves()
            keys = [l.answer_key for l in leaves]
            sig = leaf_signals(keys)
            best_oracle = max((n.oracle_score for n in graph.answer_nodes()), default=0.0)
            rec = {
                "iter": it, "action": None, "rationale": "",
                "cum_cost": graph.total_cost(),
                "agreement": sig["agreement"], "uncertainty": sig["uncertainty"],
                "diversity": sig["diversity"], "entropy": sig["entropy"],
                "n_leaves": sig["n"], "width": graph.width(),
                "max_depth": graph.max_depth(),
                "best_oracle_so_far": best_oracle,
                "plurality_key": sig["plurality_key"],
                "pruned": pruned_now,
            }
            hist.append(rec)

            # stagnation tracking (agreement not meaningfully improving)
            if it >= 1 and (hist[-1]["agreement"] - hist[-2]["agreement"]) < cfg.practical_min_improvement:
                stagnation += 1
            else:
                stagnation = 0

            stop, reason = STOP_POLICIES[cfg.stop_policy](cfg, sig, hist, graph, stagnation)
            if stop:
                rec["action"], rec["rationale"], stop_reason = "stop", reason, reason
                trajectory.append({**rec, **graph.structure()})
                break

            action, rationale = WIDTH_DEPTH_POLICIES[cfg.width_depth_policy](
                cfg, sig, graph, hist, rng)
            per_node = _per_node_units(cfg, task, unit_q)
            remaining = cfg.max_budget - graph.total_cost()
            budget_room = affordable_count()
            if budget_room <= 0:
                rec["action"], rec["rationale"], stop_reason = "stop", "max_budget", "max_budget"
                trajectory.append({**rec, **graph.structure()})
                break

            if action == "width" or (action == "depth" and not leaves):
                count = WIDTH_COUNT_POLICIES[cfg.width_count_policy](
                    cfg, sig, remaining, per_node, rng)
                count = max(1, min(count, cfg.max_nodes - graph.width(), budget_room))
                # knob 2: closed-loop temperature -- widen hotter when diversity collapses
                w_temp = cfg.width_temp
                if cfg.temp_policy == "adaptive" and sig["diversity"] < cfg.tau_div:
                    w_temp = cfg.width_temp_hi
                rec["action"] = "width"
                rec["rationale"] = f"{rationale}; +{count} samples @T={w_temp}"
                trajectory.append({**rec, **graph.structure(),
                                   "action_count": count, "width_temp": w_temp})
                await make_width_nodes(count, sample_base=graph.width(), temperature=w_temp)
            else:
                pls = per_leaf_scores(leaves)
                # expansion_width = leaves per depth step (query-specific; can be a
                # PROPORTION of the estimated budget per the research proposal).
                if cfg.expansion_width_policy == "budget_frac":
                    ew = int(round(cfg.expansion_frac * budget_room))
                elif cfg.expansion_width_policy == "uncertainty_scaled":
                    ew = int(round(cfg.depth_leaves_per_iter * (0.5 + sig["uncertainty"])))
                else:
                    ew = cfg.depth_leaves_per_iter
                k_leaves = max(1, min(ew, budget_room, len(leaves)))
                picks = LEAF_SELECT_POLICIES[cfg.leaf_select_policy](
                    cfg, leaves, pls, k_leaves, rng)
                rec["action"] = "depth"
                rec["rationale"] = (f"{rationale}; refine leaves "
                                    f"{[p.nid for p in picks]} "
                                    f"(sel={cfg.leaf_select_policy})")
                trajectory.append({**rec, **graph.structure(),
                                   "picked_leaves": [p.nid for p in picks]})
                await make_depth_nodes(picks)

        # --- aggregate final answer ---------------------------------------- #
        final_leaves = graph.leaves()
        final_answer, agg_extra = await aggregate(cfg, runner, ps, domain, dataset,
                                                  task_text, task, unit_q, qid,
                                                  graph, final_leaves)
        final_score = oracle(final_answer)
        sig_final = leaf_signals([l.answer_key for l in final_leaves])
        result = {
            "qid": qid, "domain": domain, "idx": idx,
            "task_preview": task_text[:160],
            "final_answer_preview": str(final_answer)[:400],
            "final_score": final_score,
            "used_budget": graph.total_cost(),
            "max_budget": orig_budget,
            "effective_budget": effective_budget,
            "budget_policy": cfg.budget_policy,
            "n_nodes": len(graph.answer_nodes()),
            "n_pruned": sum(1 for n in graph.nodes.values() if n.pruned),
            "width": graph.width(), "max_depth": graph.max_depth(),
            "n_iters": len(hist),
            "stop_reason": stop_reason,
            "final_agreement": sig_final["agreement"],
            "final_diversity": sig_final["diversity"],
            "best_leaf_oracle": max((n.oracle_score for n in graph.answer_nodes()),
                                    default=0.0),
            "aggregation": agg_extra,
            "structure": graph.structure(),
            "trajectory": trajectory,
            "nodes": [asdict(n) for n in graph.nodes.values()],
        }
    finally:
        cfg.max_budget = orig_budget            # restore shared cfg for the next query
    return result


async def aggregate(cfg, runner, ps, domain, dataset, task_text, task, unit_q,
                    qid, graph: AdaptiveGraph, leaves: List[GNode]):
    """Produce the final answer from the leaves per the aggregation policy."""
    if not leaves:
        return "", {"policy": cfg.aggregation_policy, "note": "no leaves"}

    policy = cfg.aggregation_policy
    agg_model = cfg.resolved_aggregator()
    extra_note: Dict[str, Any] = {}

    # knob 3: query-adaptive aggregation -- pick strategy + model from live signals
    if policy == "adaptive":
        sig = leaf_signals([l.answer_key for l in leaves])
        extra_note = {"agreement": round(sig["agreement"], 3),
                      "uncertainty": round(sig["uncertainty"], 3)}
        if sig["agreement"] >= cfg.agg_switch_agreement:
            policy = "plurality"          # majority vote is trustworthy -> cheap, no LLM
        else:
            policy = "llm_fusion"         # complementary answers -> synthesize
            if cfg.strong_aggregator and sig["uncertainty"] >= cfg.agg_model_switch_uncertainty:
                agg_model = cfg.strong_aggregator   # spend a bigger model on hard queries
        extra_note["chosen"] = policy
        extra_note["agg_model"] = agg_model

    if policy == "plurality":
        keys = [l.answer_key for l in leaves]
        plur = Counter(keys).most_common(1)[0][0]
        rep = next(l for l in leaves if l.answer_key == plur)
        return rep.output, {"policy": "plurality", "plurality_key": plur,
                            "cost_units": 0.0, **extra_note}

    if policy == "best_consistency":
        pls = per_leaf_scores(leaves)
        rep = max(leaves, key=lambda l: pls[l.nid]["quality"])
        return rep.output, {"policy": "best_consistency", "cost_units": 0.0, **extra_note}

    # llm_fusion (default): one aggregator call over all leaf outputs
    msgs = fusion_messages(ps, domain, task_text, [l.output for l in leaves])
    rec = await runner.run(qid, agg_model, "agg",
                           graph.max_depth() + 1, 0, msgs, task, unit_q)
    graph.add(dict(kind="agg", model=agg_model,
                   depth=graph.max_depth() + 1, parent=None,
                   prompt=rec["prompt"], output=rec["output"],
                   answer_key=answer_key(domain, dataset, rec["output"], cfg.equivalence_policy),
                   oracle_score=0.0, prompt_tokens=rec["prompt_tokens"],
                   completion_tokens=rec["completion_tokens"],
                   cost_units=rec["cost_units"]))
    return rec["output"], {"policy": "llm_fusion", "cost_units": rec["cost_units"],
                           **extra_note}


# =========================================================================== #
# Fixed baselines (same primitives, sized to the same budget)
# =========================================================================== #
async def build_baseline(cfg: Config, runner: NodeRunner, shape: str, domain: str,
                         dataset, record, idx: int,
                         avg_node_cost: Optional[float] = None) -> Dict[str, Any]:
    qid = f"{domain}|{idx}"
    inputs = dataset.record_to_swarm_input(record)
    task_text = inputs["task"]
    task = query_task(domain, inputs)
    unit_q = unit_cost(MODEL_SPECS[UNIT_MODEL], task)
    ps = _prompt_set(domain)
    graph = AdaptiveGraph()

    def oracle(raw: str) -> float:
        return score(domain, dataset, record, raw, inputs)

    # Size to the SAME budget in real cost. Prefer the measured per-node cost from
    # the adaptive run (accurate); fall back to the analytical FLOPs design.
    if avg_node_cost and avg_node_cost > 0:
        n = max(1, min(cfg.max_nodes,
                       int((cfg.max_budget - avg_node_cost) / avg_node_cost)))
    else:
        topo = {"parallel": "parallel", "sequential": "chain",
                "wide-tree": "star", "deep-tree": "tree"}[shape]
        n = _design_num_nodes(topo, [cfg.backbone], cfg.resolved_aggregator(),
                              cfg.max_budget, task, hard_cap=cfg.max_nodes) or 1

    async def w(sample_idx, kind="width", depth=0, parent=None, parent_out=None):
        if parent_out is None:
            msgs = width_messages(ps, domain, task_text)
        else:
            msgs = depth_messages(ps, domain, task_text, parent_out)
        rec = await runner.run(qid, cfg.backbone, kind, depth, sample_idx, msgs,
                               task, unit_q)
        return graph.add(dict(
            kind=kind, model=cfg.backbone, depth=depth, parent=parent,
            prompt=rec["prompt"], output=rec["output"],
            answer_key=answer_key(domain, dataset, rec["output"], cfg.equivalence_policy),
            oracle_score=oracle(rec["output"]),
            prompt_tokens=rec["prompt_tokens"],
            completion_tokens=rec["completion_tokens"],
            cost_units=rec["cost_units"]))

    if shape == "parallel":
        await asyncio.gather(*[w(i) for i in range(n)])
    elif shape == "sequential":
        prev = await w(0)
        for d in range(1, n):
            prev = await w(0, kind="depth", depth=d, parent=prev.nid,
                           parent_out=prev.output)
    elif shape == "wide-tree":
        root = await w(0)
        await asyncio.gather(*[
            w(i, kind="depth", depth=1, parent=root.nid, parent_out=root.output)
            for i in range(1, max(2, n))])
    elif shape == "deep-tree":
        nodes = [await w(0)]
        # binary heap: parent i -> children 2i+1, 2i+2
        for i in range(n):
            if len(nodes) >= n:
                break
            par = nodes[i]
            for _ in range(2):
                if len(nodes) >= n:
                    break
                child = await w(len(nodes), kind="depth", depth=par.depth + 1,
                                parent=par.nid, parent_out=par.output)
                nodes.append(child)

    leaves = graph.leaves()
    final_answer, agg_extra = await aggregate(cfg, runner, ps, domain, dataset,
                                              task_text, task, unit_q, qid,
                                              graph, leaves)
    return {
        "qid": qid, "shape": shape, "final_score": oracle(final_answer),
        "used_budget": graph.total_cost(), "n_nodes": len(graph.answer_nodes()),
        "width": graph.width(), "max_depth": graph.max_depth(),
        "structure": graph.structure(),
        "best_leaf_oracle": max((nd.oracle_score for nd in graph.answer_nodes()),
                                default=0.0),
    }


# =========================================================================== #
# Cache / IO helpers
# =========================================================================== #
def _load_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            logger.warning("cache unreadable, starting fresh: %s", path)
    return {}


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    # os.replace can transiently fail on Windows (WinError 5) when antivirus/indexer
    # briefly holds the destination; retry with backoff instead of crashing the run.
    import time as _time
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            _time.sleep(0.3 * (attempt + 1))
    os.replace(tmp, path)  # final attempt; if it still fails, surface the error


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_qids(path: str) -> set:
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    return done


# =========================================================================== #
# Orchestration
# =========================================================================== #
async def run_all(cfg: Config) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    datasets = {d: load_datasets()[d] for d in cfg.domains}
    cache = _load_cache(cfg.cache_path)
    sem = asyncio.Semaphore(cfg.concurrency)
    runner = NodeRunner(cfg, cache, sem)

    # optional memory-bank warm-start (read-only): per query, override the
    # query-specific Config fields from retrieved experience.
    mb = embedder = tagger = None
    if cfg.use_memory:
        from experiments_query.memory_bank import (
            MemoryBank, QueryEmbedder, QueryTagger, apply_overrides)
        mb = MemoryBank.load(cfg.memory_dir)
        embedder = QueryEmbedder()
        tagger = QueryTagger(cfg.backbone, LLMRegistry.get)
        logger.info("memory warm-start: %d experience rows from %s",
                    len(mb.rows), cfg.memory_dir)

    adaptive_path = os.path.join(cfg.outdir, "adaptive_queries.jsonl")
    traj_path = os.path.join(cfg.outdir, "trajectories.jsonl")
    baseline_path = os.path.join(cfg.outdir, "baselines.jsonl")
    if cfg.overwrite:
        for p in (adaptive_path, traj_path, baseline_path):
            if os.path.exists(p):
                os.remove(p)
    done = _done_qids(adaptive_path)

    for domain, ds in datasets.items():
        n_q = min(cfg.limit, len(ds))
        for idx in range(n_q):
            qid = f"{domain}|{idx}"
            if qid in done:
                logger.info("skip cached query %s", qid)
                continue
            logger.info("=== adaptive build %s ===", qid)
            q_cfg = cfg
            if mb is not None:
                inputs = ds.record_to_swarm_input(ds[idx])
                task_text = inputs["task"]
                diff = _difficulty_score(query_task(domain, inputs))
                tags = await tagger.tag(task_text, domain, diff)
                emb = embedder.encode(task_text)
                overrides = mb.config_overrides(domain, tags, emb, cfg.max_budget)
                q_cfg = apply_overrides(cfg, overrides)
                logger.info("  memory -> %s", {k: overrides[k] for k in
                            ("backbone", "leaf_select_policy", "aggregation_policy",
                             "budget_policy") if k in overrides})
            result = await build_adaptive(q_cfg, runner, domain, ds, ds[idx], idx)

            # per-query trajectory print
            _print_query(result)

            # baselines under the same budget (sized to the adaptive run's real
            # measured per-node cost so the comparison is budget-matched).
            baselines = {}
            if cfg.run_baselines:
                agent_costs = [n["cost_units"] for n in result["nodes"]
                               if n["kind"] != "agg" and n["cost_units"] > 0]
                avg_nc = (sum(agent_costs) / len(agent_costs)) if agent_costs else None
                for shape in ("parallel", "sequential", "wide-tree", "deep-tree"):
                    try:
                        b = await build_baseline(cfg, runner, shape, domain,
                                                 ds, ds[idx], idx,
                                                 avg_node_cost=avg_nc)
                    except Exception as exc:
                        logger.warning("baseline %s failed on %s: %s",
                                       shape, qid, str(exc)[:160])
                        b = {"shape": shape, "final_score": 0.0,
                             "used_budget": float("nan"), "n_nodes": 0}
                    baselines[shape] = b
                    _append_jsonl(baseline_path, {"qid": qid, **b})

            slim = {k: v for k, v in result.items()
                    if k not in ("trajectory", "nodes")}
            slim["baselines"] = {s: {"final_score": b["final_score"],
                                     "used_budget": b["used_budget"],
                                     "n_nodes": b.get("n_nodes")}
                                 for s, b in baselines.items()}
            _append_jsonl(adaptive_path, slim)
            _append_jsonl(traj_path, {"qid": qid, "domain": domain,
                                      "trajectory": result["trajectory"],
                                      "nodes": result["nodes"]})
            _save_cache(cfg.cache_path, cache)

    _save_cache(cfg.cache_path, cache)
    analyze(cfg)


def _print_query(r: Dict[str, Any]) -> None:
    print(f"\n[{r['qid']}] final_score={r['final_score']:.0f} "
          f"budget={r['used_budget']:.1f}/{r['max_budget']:.0f} "
          f"nodes={r['n_nodes']} (w={r['width']},d={r['max_depth']}) "
          f"iters={r['n_iters']} stop={r['stop_reason']}")
    for t in r["trajectory"]:
        extra = ""
        if t["action"] == "width":
            extra = f" +{t.get('action_count','?')}"
        elif t["action"] == "depth":
            extra = f" leaves={t.get('picked_leaves')}"
        print(f"   it{t['iter']}: agree={t['agreement']:.2f} "
              f"unc={t['uncertainty']:.2f} div={t['diversity']:.2f} "
              f"-> {t['action']}{extra}  ({t['rationale']})")


# =========================================================================== #
# Analysis, report, plots, memory export
# =========================================================================== #
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def analyze(cfg: Config) -> None:
    rows = _read_jsonl(os.path.join(cfg.outdir, "adaptive_queries.jsonl"))
    if not rows:
        logger.warning("no results to analyze")
        return
    trajs = {t["qid"]: t for t in
             _read_jsonl(os.path.join(cfg.outdir, "trajectories.jsonl"))}

    per_domain: Dict[str, Any] = {}
    for domain in sorted({r["domain"] for r in rows}):
        drows = [r for r in rows if r["domain"] == domain]
        n = len(drows)
        shapes = ("parallel", "sequential", "wide-tree", "deep-tree")

        adaptive_acc = sum(r["final_score"] for r in drows) / n
        adaptive_cost = sum(r["used_budget"] for r in drows) / n
        base_acc = {s: _avg([r["baselines"].get(s, {}).get("final_score", 0.0)
                             for r in drows]) for s in shapes}
        base_cost = {s: _avg([r["baselines"].get(s, {}).get("used_budget", float("nan"))
                              for r in drows]) for s in shapes}

        # win/efficiency vs best fixed baseline (per query)
        wins = ties = 0
        eff_wins = 0
        for r in drows:
            best_base = max((r["baselines"].get(s, {}).get("final_score", 0.0)
                             for s in shapes), default=0.0)
            if r["final_score"] > best_base:
                wins += 1
            elif r["final_score"] == best_base:
                ties += 1
            # efficiency: matched-or-better accuracy at lower cost than best baseline
            bc = min((r["baselines"].get(s, {}).get("used_budget", float("inf"))
                      for s in shapes
                      if r["baselines"].get(s, {}).get("final_score", 0.0) >= r["final_score"]),
                     default=float("inf"))
            if r["final_score"] >= best_base and r["used_budget"] < bc:
                eff_wins += 1

        stop_reasons = Counter(r["stop_reason"] for r in drows)
        init_widths = Counter()
        wd_actions = Counter()
        promising_leaf_hits = []       # did depth on a picked leaf improve best oracle?
        for r in drows:
            tr = trajs.get(r["qid"], {}).get("trajectory", [])
            if tr:
                init_widths[tr[0].get("width", r["width"])] += 1
            for t in tr:
                if t["action"] in ("width", "depth"):
                    wd_actions[t["action"]] += 1

        per_domain[domain] = {
            "n_queries": n,
            "adaptive_acc": round(adaptive_acc, 3),
            "adaptive_cost": round(adaptive_cost, 2),
            "baseline_acc": {s: round(base_acc[s], 3) for s in shapes},
            "baseline_cost": {s: round(base_cost[s], 2) for s in shapes},
            "adaptive_wins_vs_best_base": wins,
            "adaptive_ties": ties,
            "efficiency_wins": eff_wins,
            "stop_reasons": dict(stop_reasons),
            "init_width_dist": {str(k): v for k, v in sorted(init_widths.items())},
            "width_depth_action_mix": dict(wd_actions),
            "avg_nodes": round(_avg([r["n_nodes"] for r in drows]), 2),
            "avg_iters": round(_avg([r["n_iters"] for r in drows]), 2),
        }

    report_path = os.path.join(cfg.outdir, "report.json")
    json.dump({"config": _cfg_public(cfg), "per_domain": per_domain},
              open(report_path, "w"), indent=2)
    _write_report_csv(cfg, per_domain)
    _print_report(per_domain)
    export_memory(cfg, rows, trajs, per_domain)
    if cfg.make_plots:
        try:
            make_plots(cfg, rows, trajs, per_domain)
        except Exception as exc:
            logger.warning("plotting failed: %s", str(exc)[:200])
    print("\nsaved report ->", report_path)


def _avg(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return sum(xs) / len(xs) if xs else 0.0


def _cfg_public(cfg: Config) -> Dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(cfg).items()}


def _write_report_csv(cfg: Config, per_domain: Dict[str, Any]) -> None:
    path = os.path.join(cfg.outdir, "report_by_domain.csv")
    shapes = ("parallel", "sequential", "wide-tree", "deep-tree")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "n", "adaptive_acc", "adaptive_cost"]
                   + [f"{s}_acc" for s in shapes]
                   + [f"{s}_cost" for s in shapes]
                   + ["adaptive_wins", "efficiency_wins", "avg_nodes", "avg_iters"])
        for d, v in per_domain.items():
            w.writerow([d, v["n_queries"], v["adaptive_acc"], v["adaptive_cost"]]
                       + [v["baseline_acc"][s] for s in shapes]
                       + [v["baseline_cost"][s] for s in shapes]
                       + [v["adaptive_wins_vs_best_base"], v["efficiency_wins"],
                          v["avg_nodes"], v["avg_iters"]])


def _print_report(per_domain: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("ADAPTIVE TTS GRAPH CONSTRUCTION  --  dataset-level conclusions")
    print("=" * 78)
    shapes = ("parallel", "sequential", "wide-tree", "deep-tree")
    for d, v in per_domain.items():
        print(f"\n### {d.upper()}  (n={v['n_queries']})")
        print(f"  adaptive : acc={v['adaptive_acc']:.3f}  cost={v['adaptive_cost']:.2f} units"
              f"  (avg {v['avg_nodes']} nodes, {v['avg_iters']} iters)")
        for s in shapes:
            print(f"  {s:11}: acc={v['baseline_acc'][s]:.3f}  "
                  f"cost={v['baseline_cost'][s]:.2f} units")
        print(f"  adaptive beats best fixed baseline: {v['adaptive_wins_vs_best_base']}/{v['n_queries']}"
              f"  (ties {v['adaptive_ties']}); efficiency wins {v['efficiency_wins']}/{v['n_queries']}")
        print(f"  init-width dist  : {v['init_width_dist']}")
        print(f"  width/depth mix  : {v['width_depth_action_mix']}")
        print(f"  stop reasons     : {v['stop_reasons']}")


def export_memory(cfg: Config, rows, trajs, per_domain) -> None:
    """Distil reusable, warm-start-able experience per domain."""
    memory = {}
    for domain, v in per_domain.items():
        drows = [r for r in rows if r["domain"] == domain]
        # which stop reasons coincided with correct answers (reliable stops)
        reliable_stop = Counter()
        for r in drows:
            if r["final_score"] > 0:
                reliable_stop[r["stop_reason"]] += 1
        # promising-leaf signal: among depth actions, did picking low-consistency
        # (novel) leaves precede an eventual correct answer?
        widths_when_correct = [r["width"] for r in drows if r["final_score"] > 0]
        memory[domain] = {
            "preferred_init_width": _mode(v["init_width_dist"]),
            "width_vs_depth_tendency": _tendency(v["width_depth_action_mix"]),
            "median_width_when_correct": _median(widths_when_correct),
            "effective_stop_reasons": dict(reliable_stop.most_common()),
            "recommended_stop_policy": cfg.stop_policy,
            "note": ("Warm-start: start at preferred_init_width; bias width/depth "
                     "per tendency; trust the listed stop reasons for this domain."),
        }
    path = os.path.join(cfg.outdir, "memory_patterns.json")
    json.dump(memory, open(path, "w"), indent=2)
    print("\nreusable memory patterns ->", path)
    for d, m in memory.items():
        print(f"  [{d}] init_width~{m['preferred_init_width']} "
              f"tendency={m['width_vs_depth_tendency']} "
              f"stop={list(m['effective_stop_reasons'])[:3]}")


def _mode(dist: Dict[str, int]) -> Optional[int]:
    if not dist:
        return None
    return int(max(dist.items(), key=lambda kv: kv[1])[0])


def _median(xs: List[float]):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _tendency(mix: Dict[str, int]) -> str:
    w, d = mix.get("width", 0), mix.get("depth", 0)
    if w == d == 0:
        return "none"
    if w > 1.5 * d:
        return "width-leaning"
    if d > 1.5 * w:
        return "depth-leaning"
    return "balanced"


def make_plots(cfg: Config, rows, trajs, per_domain) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = ("parallel", "sequential", "wide-tree", "deep-tree")
    # (1) dataset-level accuracy & cost bars (adaptive vs baselines)
    for domain, v in per_domain.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        labels = ["adaptive"] + list(shapes)
        accs = [v["adaptive_acc"]] + [v["baseline_acc"][s] for s in shapes]
        costs = [v["adaptive_cost"]] + [v["baseline_cost"][s] for s in shapes]
        ax1.bar(labels, accs, color="steelblue")
        ax1.set_title(f"{domain}: accuracy"); ax1.set_ylim(0, 1)
        ax1.tick_params(axis="x", rotation=30)
        ax2.bar(labels, costs, color="indianred")
        ax2.set_title(f"{domain}: used budget (1B-units)")
        ax2.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.outdir, f"summary_{domain}.png"), dpi=110)
        plt.close(fig)

    # (2) a few per-query trajectories (agreement & cost over iterations)
    shown = 0
    for qid, t in trajs.items():
        if shown >= cfg.plot_queries:
            break
        tr = t.get("trajectory", [])
        if len(tr) < 2:
            continue
        its = [x["iter"] for x in tr]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(its, [x["agreement"] for x in tr], "-o", label="agreement")
        ax.plot(its, [x["uncertainty"] for x in tr], "-s", label="uncertainty")
        ax.plot(its, [x["diversity"] for x in tr], "-^", label="diversity")
        ax.plot(its, [x["best_oracle_so_far"] for x in tr], "-d",
                label="best oracle", alpha=0.6)
        for x in tr:
            if x["action"] in ("width", "depth"):
                ax.annotate(x["action"][0].upper(), (x["iter"], 1.02),
                            fontsize=8, ha="center")
        ax.set_title(qid); ax.set_xlabel("iteration"); ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = qid.replace("|", "_")
        fig.savefig(os.path.join(cfg.outdir, f"traj_{safe}.png"), dpi=110)
        plt.close(fig)
        shown += 1
    print(f"saved plots -> {cfg.outdir}/summary_*.png, traj_*.png")


# =========================================================================== #
# CLI
# =========================================================================== #
def build_config(argv: Optional[List[str]] = None) -> Tuple[Config, bool]:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default=Config.backbone)
    p.add_argument("--aggregator", default=None)
    p.add_argument("--domains", default="math",
                   help="comma list of math|mmlu|humaneval")
    p.add_argument("--limit", type=int, default=Config.limit)
    p.add_argument("--max-budget", type=float, default=Config.max_budget)
    p.add_argument("--max-iters", type=int, default=Config.max_iters)
    p.add_argument("--max-nodes", type=int, default=Config.max_nodes)
    p.add_argument("--max-tokens", type=int, default=Config.max_tokens)
    p.add_argument("--width-temp", type=float, default=Config.width_temp)
    p.add_argument("--depth-temp", type=float, default=Config.depth_temp)
    p.add_argument("--agg-temp", type=float, default=Config.agg_temp)
    p.add_argument("--init-width-policy", default=Config.init_width_policy,
                   choices=list(INIT_WIDTH_POLICIES))
    p.add_argument("--init-width", type=int, default=Config.init_width)
    p.add_argument("--init-width-cap", type=int, default=Config.init_width_cap)
    p.add_argument("--width-depth-policy", default=Config.width_depth_policy,
                   choices=list(WIDTH_DEPTH_POLICIES))
    p.add_argument("--width-count-policy", default=Config.width_count_policy,
                   choices=list(WIDTH_COUNT_POLICIES))
    p.add_argument("--width-increment", type=int, default=Config.width_increment)
    p.add_argument("--leaf-select-policy", default=Config.leaf_select_policy,
                   choices=list(LEAF_SELECT_POLICIES))
    p.add_argument("--depth-leaves-per-iter", type=int,
                   default=Config.depth_leaves_per_iter)
    p.add_argument("--expansion-width-policy", default=Config.expansion_width_policy,
                   choices=["fixed", "budget_frac", "uncertainty_scaled"],
                   help="leaves per depth step: fixed | proportion-of-budget | uncertainty-scaled")
    p.add_argument("--expansion-frac", type=float, default=Config.expansion_frac)
    p.add_argument("--aggregation-policy", default=Config.aggregation_policy,
                   choices=["llm_fusion", "plurality", "best_consistency", "adaptive"])
    p.add_argument("--stop-policy", default=Config.stop_policy,
                   choices=list(STOP_POLICIES))
    p.add_argument("--agreement-stop", type=float, default=Config.agreement_stop)
    p.add_argument("--uncertainty-stop", type=float, default=Config.uncertainty_stop)
    p.add_argument("--route-uncertainty-hi", type=float, default=Config.route_uncertainty_hi,
                   help="uncertainty_routed WIDTH trigger: widen when uncertainty >= this")
    p.add_argument("--route-uncertainty-lo", type=float, default=Config.route_uncertainty_lo,
                   help="uncertainty_routed DEPTH-consolidate threshold: uncertainty <= this")
    p.add_argument("--patience", type=int, default=Config.patience)
    # --- knob 1: per-query budget ---
    p.add_argument("--budget-policy", default=Config.budget_policy,
                   choices=list(BUDGET_POLICIES))
    p.add_argument("--budget-min", type=float, default=Config.budget_min)
    p.add_argument("--budget-max", type=float, default=Config.budget_max)
    # --- knob 2: closed-loop temperature / n-sampling ---
    p.add_argument("--temp-policy", default=Config.temp_policy,
                   choices=["fixed", "adaptive"])
    p.add_argument("--width-temp-hi", type=float, default=Config.width_temp_hi)
    p.add_argument("--tau-div", type=float, default=Config.tau_div)
    p.add_argument("--width-n", type=int, default=Config.width_n,
                   help="num_comps per width call (>1 = cheap n-sampling)")
    # --- knob 3: query-adaptive aggregation ---
    p.add_argument("--agg-switch-agreement", type=float, default=Config.agg_switch_agreement)
    p.add_argument("--strong-aggregator", default=None)
    p.add_argument("--agg-model-switch-uncertainty", type=float,
                   default=Config.agg_model_switch_uncertainty)
    # --- knob 4: answer-equivalence + pruning ---
    p.add_argument("--equivalence-policy", default=Config.equivalence_policy,
                   choices=["auto", "exact", "numeric", "code"])
    p.add_argument("--prune-policy", default=Config.prune_policy,
                   choices=["none", "beam"])
    p.add_argument("--beam-k", type=int, default=Config.beam_k)
    p.add_argument("--prune-min-agreement", type=float, default=Config.prune_min_agreement)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--concurrency", type=int, default=Config.concurrency)
    p.add_argument("--outdir", default=Config.outdir)
    p.add_argument("--cache-path", default=None)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plot-queries", type=int, default=Config.plot_queries)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--use-memory", action="store_true",
                   help="warm-start each query's config from the memory bank")
    p.add_argument("--memory-dir", default=Config.memory_dir)
    p.add_argument("--analyze-only", action="store_true")
    a = p.parse_args(argv)

    cfg = Config(
        backbone=a.backbone, aggregator=a.aggregator,
        domains=tuple(d.strip() for d in a.domains.split(",") if d.strip()),
        limit=a.limit, max_budget=a.max_budget, max_iters=a.max_iters,
        max_nodes=a.max_nodes, max_tokens=a.max_tokens,
        width_temp=a.width_temp, depth_temp=a.depth_temp, agg_temp=a.agg_temp,
        init_width_policy=a.init_width_policy, init_width=a.init_width,
        init_width_cap=a.init_width_cap, width_depth_policy=a.width_depth_policy,
        width_count_policy=a.width_count_policy, width_increment=a.width_increment,
        leaf_select_policy=a.leaf_select_policy,
        depth_leaves_per_iter=a.depth_leaves_per_iter,
        aggregation_policy=a.aggregation_policy, stop_policy=a.stop_policy,
        agreement_stop=a.agreement_stop, uncertainty_stop=a.uncertainty_stop,
        patience=a.patience, seed=a.seed, concurrency=a.concurrency,
        outdir=a.outdir, run_baselines=not a.no_baselines,
        make_plots=not a.no_plots, plot_queries=a.plot_queries,
        overwrite=a.overwrite,
    )
    cfg.route_uncertainty_hi = a.route_uncertainty_hi
    cfg.route_uncertainty_lo = a.route_uncertainty_lo
    # knobs 1-4
    cfg.budget_policy = a.budget_policy
    cfg.budget_min = a.budget_min
    cfg.budget_max = a.budget_max
    cfg.temp_policy = a.temp_policy
    cfg.width_temp_hi = a.width_temp_hi
    cfg.tau_div = a.tau_div
    cfg.width_n = a.width_n
    cfg.agg_switch_agreement = a.agg_switch_agreement
    cfg.strong_aggregator = a.strong_aggregator
    cfg.agg_model_switch_uncertainty = a.agg_model_switch_uncertainty
    cfg.equivalence_policy = a.equivalence_policy
    cfg.prune_policy = a.prune_policy
    cfg.beam_k = a.beam_k
    cfg.prune_min_agreement = a.prune_min_agreement
    cfg.use_memory = a.use_memory
    cfg.memory_dir = a.memory_dir
    cfg.expansion_width_policy = a.expansion_width_policy
    cfg.expansion_frac = a.expansion_frac
    cfg.cache_path = a.cache_path or os.path.join(cfg.outdir, "node_cache.json")
    return cfg, a.analyze_only


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
    cfg, analyze_only = build_config()
    logger.info("config: %s", json.dumps(_cfg_public(cfg)))
    if analyze_only:
        analyze(cfg)
    else:
        asyncio.run(run_all(cfg))


if __name__ == "__main__":
    main()
