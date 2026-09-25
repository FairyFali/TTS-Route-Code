"""Dynamic, label-free oracle search over Best-Route data.

This is a NEW builder. `build_bestroute_tts.py` is left completely untouched so its
labels stay reproducible; nothing here calls into its search. It does reuse that
module's *prompt builders* and *judge* (`gen_messages`, `fuse_messages`, `judge`) on
purpose -- duplicating the prompts or the grading rubric would let the two datasets
drift apart and stop being comparable.

How this differs from the old search
------------------------------------
1. ONE coupled width/depth search per (model, run). The old builder evaluated two
   *separate* shapes (a flat width-W layer, and a narrow base_width=2 depth chain) and
   took the cheaper. Here WIDEN and DEEPEN are two moves of a single graph: each layer
   widens adaptively up to `--max-width`, and if agreement is still below tau the
   layer's frontier is reconciled into a new layer, up to `--max-depth` layers.

2. LABEL-FREE. Ground truth is consulted exactly once, on the final aggregated answer.
   The old builder scanned sample prefixes for the first point at which the majority
   *happened to be correct* (`maj_k`), which peeks at labels inside the search and
   therefore produces configurations a deployed, agreement-driven planner cannot
   reach. Expect solve rates here to be LOWER than the old labels, and the model
   distribution to shift upward. That is the labels becoming honest, not a regression.

3. NO PRUNING. Every model is searched on every query, each under the SAME fixed
   budget B. Because nothing tightens B, the models are fully independent -- the
   cheapest-solving configuration is derived offline from the recorded statistics
   rather than during the search.

4. REPEATED RUNS. Each (query, model) is searched `--repeats` times with independent
   generations, yielding a success probability and a mean solving cost instead of a
   single deterministic winner.

Cost unit
---------
One single inference call of the 1B model ON THIS QUERY costs exactly 1.0. The unit is
the *realized* FLOPs of the probe call (measured prompt/completion tokens), not an
analytical estimate from average lengths -- so `cost_units` is directly interpretable
as "how many 1B calls was this worth". Every other model and every graph cost is
expressed against that unit.

Independence of repeats
-----------------------
`NodeRunner`'s cache key carries no run index, so re-running a search would replay
byte-identical samples and collapse every success probability to 0.0 or 1.0. The run
index is threaded through the `kind` field (`dyn{run}`), which IS part of the key, so
each repeat issues genuine API calls. The builder also writes its own cache under
`--outdir`; sharing `results/bestroute_tts/node_cache.json` would make run 0 silently
replay the OLD search's draws, since those keys have the same shape.

Usage
-----
  python -m experiments_query.build_bestroute_dynamic --n 500 --tau 0.6 --repeats 5
  # tau is fixed within an experiment; sweep it by running again with a different
  # --tau (results land in a tau-suffixed file, so runs never overwrite each other).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_query.experiment_adaptive_tts import (
    Config, NodeRunner, _load_cache, _save_cache)
from experiments_query.budget import MODEL_SPECS, Task, node_cost
from experiments_query.experiment_tts_preference import est_input_tokens
# Prompt builders + judge are shared with the old builder ON PURPOSE (see module docstring).
from experiments_query.build_bestroute_tts import gen_messages, fuse_messages, judge, JUDGE_EMPTY

logger = logging.getLogger("bestroute_dynamic")

GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"
SPLITS = "datasets/splits/split_bestroute.jsonl"

# gemma-3-4b REMOVED from the pool. It is rate-limited on this account far below the
# global limit and was the sole source of every RateLimitError observed (91 hard failures
# and ~12% of all calls spent on retries). Because tenacity backs off inside the
# semaphores, its retries stalled whole queries -- gaps of 53 and 23 minutes -- and it was
# the binding constraint on the entire sweep, not compute.
# qwen-2.5-7b removed 2026-09-01 (user decision): pool is 1b/8b/27b/70b/72b.
MODEL_POOL = ["meta-llama/llama-3.2-1b-instruct",
              "meta-llama/llama-3.1-8b-instruct",
              "google/gemma-3-27b-it", "meta-llama/llama-3.1-70b-instruct",
              "qwen/qwen-2.5-72b-instruct"]
UNIT_MODEL = MODEL_POOL[0]          # the 1B: one call on a query == cost 1.0 by definition
LARGEST = MODEL_POOL[-1]            # its single-call cost sets the per-query budget B
SHORT = {m: s for m, s in zip(MODEL_POOL, ["1b", "8b", "27b", "70b", "72b"])}

WIDTH_TEMP = 0.7                    # independent samples -> diverse
DEPTH_TEMP = 0.5                    # reconciliation -> more deterministic
NULL_KEYS = ("", "(empty)", "none", None)

_FINAL = re.compile(r"final answer\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


# --------------------------------------------------------------------------- #
# Label-free answer clustering
# --------------------------------------------------------------------------- #
def answer_key(text: str) -> str:
    """Cluster free-form answers WITHOUT looking at ground truth.

    Prefers an explicit "final answer:" line; otherwise falls back to the normalised
    head of the response. Note this is a weak equivalence for open-ended text: distinct
    phrasings of the same answer land in different clusters, so plurality agreement
    decays roughly as c/n as width grows. The trajectory records per-layer agreement so
    the pilot can quantify that rather than assume it.
    """
    m = list(_FINAL.finditer(text or ""))
    if m:
        k = m[-1].group(1).strip().lower()
    else:
        k = re.sub(r"\s+", " ", (text or "")[:60]).strip().lower()
    return k[:80] if k else "(empty)"


def plurality(keys: List[str]) -> Tuple[Optional[str], float]:
    """(plurality key, its share of ALL drawn samples). Null keys can never win."""
    if not keys:
        return None, 0.0
    counts = collections.Counter(k for k in keys if k not in NULL_KEYS)
    if not counts:
        return None, 0.0
    key, n = counts.most_common(1)[0]
    return key, n / len(keys)


def select_frontier(outs: List[str], keys: List[str], k: int) -> List[str]:
    """Pick <= k CONFLICTING solutions from a layer to hand to the next one.

    A reconciliation node embeds its inputs in the prompt and node_cost is quadratic in
    prompt length, so feeding it an entire 16-wide layer makes depth nodes ~6x the cost
    of a width node (measured: ~64 units vs ~8 for 8b). Reconciling every sample is also
    redundant -- duplicates of the same answer add tokens but no information.

    We therefore keep ONE representative per distinct answer, ordered by how much support
    that answer has, capped at k. That preserves exactly the disagreement the layer is
    meant to resolve while cutting the prompt to the number of *distinct* positions.
    """
    by: Dict[str, List[str]] = {}
    for o, key in zip(outs, keys):
        if key in NULL_KEYS:
            continue
        by.setdefault(key, []).append(o)
    if not by:                                     # everything empty -> fall back to raw
        return outs[:k]
    ordered = sorted(by.items(), key=lambda kv: -len(kv[1]))
    return [v[0] for _, v in ordered[:k]]


# --------------------------------------------------------------------------- #
# One dynamic search: coupled WIDEN / DEEPEN under a fixed budget
# --------------------------------------------------------------------------- #
async def dynamic_search(runner: NodeRunner, jc: Dict[str, Any], sem: asyncio.Semaphore,
                         rec: Dict[str, Any], model: str, run: int, tau: float,
                         budget: float, est_node: float, max_width: int, max_depth: int,
                         min_width: int, batch: int, depth_frontier: int, wmax: int,
                         unit: float, task: Task, verifier=None,
                         patience: int = 1, min_gain: float = 0.0,
                         stop_signal: str = "agreement",
                         readout: str = "auto") -> Dict[str, Any]:
    """Search ONE model ONCE, label-free, then grade the aggregated answer.

    Two stopping signals are supported.

    agreement (verifier=None) -- the original: widen until the plurality share of the
        answer keys reaches tau. On free-form text this barely works: a 16-sample pool
        holds ~9.2 distinct answers, so plurality wins with ~2 votes and its share DECAYS
        as ~c/n, moving away from any fixed tau as the layer widens.

    verifier -- score every drawn sample with the proxy reward model and track the best
        score seen. A batch that fails to beat the running best by `min_gain` is evidence
        that more sampling is not finding better answers, so the layer stops; `patience`
        consecutive such batches stop the search. Read-out is then argmax of the verifier
        score (best-of-n) rather than plurality.

        This is the measured-better signal: on the same pools, plurality scores 0.718
        (vs 0.715 for a single sample) while verifier best-of-8 scores 0.741.

    Per layer: draw samples in batches of `batch`, recomputing plurality agreement after
    each batch. Stop the whole search when agreement >= tau, or when the budget cannot
    fund another node. If a layer exhausts `max_width` without consensus, reconcile its
    frontier into the next layer. Read-out is the plurality answer of the FINAL layer.

    Batching trades a little stopping precision for wall-clock: the width loop is
    inherently sequential (each sample informs whether to draw another), so drawing one
    at a time makes the critical path ~max_width deep per layer. With batch=4 the search
    may overshoot the stopping point by up to 3 samples, which costs a little compute but
    never changes correctness -- agreement is still computed over everything drawn.
    """
    prompt = rec["prompt"]
    gt, fa = rec["ground_truth"], rec.get("final_answer", "")
    qid = rec["id"]

    cost = 0.0
    drawn = 0
    max_node = 0.0            # largest realized node cost; refines the a-priori estimate
    best_score = float("-inf")   # best verifier score seen anywhere in this search
    best_answer = ""
    stale = 0                    # consecutive batches with no better answer
    all_scores: List[float] = []
    over_budget_first = False
    prev_frontier: Optional[List[str]] = None
    layers: List[Dict[str, Any]] = []
    final_keys: List[str] = []
    final_outs: List[str] = []
    stop = "max_depth"

    prev_best = float("-inf")
    for layer in range(max_depth):
        keys: List[str] = []
        outs: List[str] = []
        agree = 0.0
        layer_stop = "max_width"
        w = 0

        while w < max_width:
            # Project the next node's cost conservatively: the analytical estimate for
            # this model until we have measured one, then the largest realized cost. A
            # DEEPEN node embeds the whole frontier in its prompt and node_cost is
            # quadratic in prompt length, so depth nodes are much dearer than width
            # nodes -- projecting from a running MEAN under-estimates and overshoots B.
            proj = max(max_node, est_node)
            k = min(batch, max_width - w)

            if w == 0:
                # FIRST DRAW OF EVERY LAYER IS A SINGLE NODE. est_node is derived from the PROBE's
                # lengths (a 1B answer); bigger models write longer answers, so it can
                # under-estimate realized cost several-fold. Sizing the first batch from
                # it let 27b draw 2 nodes at ~124 units each against B=70.9. Measuring one
                # node first costs a single extra round-trip and makes every subsequent
                # projection use a realized cost instead of a guess.
                #
                # This matters again at every WIDTH->DEPTH transition: a reconciliation
                # node embeds the whole frontier in its prompt, so it can cost ~6x a width
                # node of the same model. Projecting depth nodes from width nodes let 7b
                # draw 2 depth nodes and reach 144 units against B=70.8.
                #
                # We still draw that one node even when it alone exceeds B, so that
                # "cannot afford" is never silently recorded as "cannot solve".
                k = 1
                if drawn > 0 and cost >= budget:
                    layer_stop = "budget"
                    break
            else:
                room = budget - cost
                afford = int(room // proj) if proj > 0 else k
                k = min(k, max(0, afford))
                if k <= 0:
                    layer_stop = "budget"
                    break

            msgs = (gen_messages(prompt) if prev_frontier is None
                    else fuse_messages(prompt, prev_frontier))
            temp = WIDTH_TEMP if prev_frontier is None else DEPTH_TEMP
            nodes = await asyncio.gather(*[
                runner.run(qid, model, f"dyn{run}", layer, w + i, msgs, task, unit,
                           temperature=temp, max_tokens=wmax) for i in range(k)])

            for node in nodes:
                c = float(node["cost_units"])
                if drawn == 0 and c > budget:
                    # a single call of this model already exceeds B -- realized, not projected
                    over_budget_first = True
                cost += c
                max_node = max(max_node, c)
                drawn += 1
                outs.append(node["output"] or "")
                keys.append(answer_key(node["output"]))
            _, agree = plurality(keys)
            w += k

            # --- both stop conditions are evaluated independently, then combined ---
            agree_stop = len(keys) >= min_width and agree >= tau
            verif_stop = False
            if verifier is not None:
                # Score just the new batch and charge it. A batch that does not beat the
                # running best by min_gain means more sampling is not surfacing better
                # answers -- the direct analogue of "agreement has stopped improving",
                # but on a signal that actually orders free-form responses.
                new_texts = [n["output"] or "" for n in nodes]
                sc = verifier.score(prompt, new_texts)
                all_scores += sc
                for node, text, v in zip(nodes, new_texts, sc):
                    cost += verifier.cost_units(int(node.get("prompt_tokens") or 0),
                                                int(node.get("completion_tokens") or 0), unit)
                    if v > best_score:
                        best_score, best_answer = v, text
                improved = bool(sc) and max(sc) > (prev_best + min_gain)
                prev_best = max(prev_best, max(sc) if sc else prev_best)
                stale = 0 if improved else stale + 1
                verif_stop = len(all_scores) >= min_width and stale >= patience

            if stop_signal == "agreement":
                fire, why = agree_stop, "agreement"
            elif stop_signal == "verifier":
                fire, why = verif_stop, "no_improvement"
            elif stop_signal == "both_or":       # whichever fires first -> cheapest
                fire = agree_stop or verif_stop
                why = "agreement" if agree_stop else "no_improvement"
            else:                                 # both_and -> keep going until both agree
                fire = agree_stop and verif_stop
                why = "both"
            if fire:
                layer_stop = why
                break

        nxt = select_frontier(outs, keys, depth_frontier)
        layers.append({"layer": layer, "width": len(keys), "agreement": round(agree, 4),
                       "cum_cost": round(cost, 4), "stop": layer_stop,
                       "n_distinct": len({x for x in keys if x not in NULL_KEYS}),
                       "frontier_passed_on": len(nxt)})
        if keys:
            final_keys, final_outs = keys, outs

        if layer_stop in ("agreement", "budget", "no_improvement"):
            stop = layer_stop
            break
        prev_frontier = nxt                        # DEEPEN: reconcile the CONFLICTING subset

    # ---- read-out ----
    key, agree = plurality(final_keys)
    use_verifier_readout = (verifier is not None and
                            (readout == "verifier" or
                             (readout == "auto" and stop_signal != "agreement")))
    if use_verifier_readout:
        # best-of-n over EVERY sample drawn in this search, not just the final layer:
        # the verifier gives a comparable score across layers, which plurality cannot.
        answer = best_answer
    elif key is None:
        answer = final_outs[0] if final_outs else ""
    else:
        answer = next(o for o, k in zip(final_outs, final_keys) if k == key)
    correct = bool(await judge(prompt, gt, fa, answer, sem, jc)) if answer.strip() else False

    return {"run": run, "solved": correct, "cost": round(cost, 4), "nodes": drawn,
            "stop": stop, "layers_used": len(layers), "final_width": len(final_keys),
            "final_agreement": round(agree, 4),
            "over_budget_first_node": over_budget_first,
            "best_verifier_score": (None if best_score == float("-inf") else round(best_score, 4)),
            "n_verifier_scored": len(all_scores),
            # Overshoot is bounded by ONE node: a node's cost is unknowable before it runs,
            # so the gate stops as soon as the budget cannot fund another PROJECTED node.
            "budget_exceeded_by": round(max(0.0, cost - budget), 4),
            "layers": layers}


# --------------------------------------------------------------------------- #
# One query: every model, `repeats` runs each
# --------------------------------------------------------------------------- #
VERIFIER = None      # set in main_async when --stop-signal verifier


async def search_query(runner: NodeRunner, jc: Dict[str, Any], sem: asyncio.Semaphore,
                       rec: Dict[str, Any], a) -> Dict[str, Any]:
    prompt = rec["prompt"]
    task = Task(est_input_tokens(prompt), 512)     # only shapes NodeRunner's signature

    # ---- 1. cost unit: realized FLOPs of ONE 1B call on this query ----
    # unit_q=1.0 makes `cost_units` the raw FLOPs, which then BECOMES the unit.
    probe = await runner.run(rec["id"], UNIT_MODEL, "probe", 0, 0, gen_messages(prompt),
                             task, 1.0, temperature=WIDTH_TEMP, max_tokens=2048)
    unit_flops = float(probe["cost_units"])        # raw FLOPs of ONE 1B call on this query
    if unit_flops <= 0:                            # probe failed -> cannot normalise
        return {"id": rec["id"], "source": rec["source"], "error": "probe_failed"}
    unit = unit_flops                              # divisor => a 1B call costs exactly 1.0
    wmax = int(min(max(int(probe.get("completion_tokens") or 256) + 512, 512), 2048))

    # ---- 2. budget: ANALYTICAL cost of one largest-model call at this query's lengths ----
    # Deliberately not a realized 72B sample: that is one draw, and FLOPs scale with
    # tokens, so a terse 72B answer yields a B smaller than a single 27B call and the
    # search never runs. The analytical form is ~68-71 units regardless of length and
    # gives each model a sane node allowance (4b ~20, 7b ~10, 27b ~2, 70b/72b 1).
    # It also removes one 72B call per query -- the most expensive call in the loop.
    n_p = int(probe.get("prompt_tokens") or 0)
    n_d = int(probe.get("completion_tokens") or 0)
    budget = node_cost(n_p, n_d, MODEL_SPECS[LARGEST]) / unit_flops
    est_node = {m: node_cost(n_p, n_d, MODEL_SPECS[m]) / unit_flops for m in MODEL_POOL}
    # --tiered-budget: each model's search may cost at most ONE call of the next model up
    # (analytical, at this query's probe lengths -- same basis as the global budget).
    if a.tiered_budget:
        by_short = {SHORT[m]: m for m in MODEL_POOL}
        nxt = {"1b": "8b", "8b": "27b", "27b": "72b", "70b": "72b", "72b": "72b"}
        budget_of = {m: (est_node[by_short[nxt[SHORT[m]]]] if SHORT[m] in nxt and nxt[SHORT[m]] != SHORT[m]
                         else budget) for m in MODEL_POOL}
    else:
        budget_of = {m: budget for m in MODEL_POOL}

    # ---- 3. every model, `repeats` independent runs ----
    # Nothing tightens B any more, so the models are fully INDEPENDENT: the 1B->72B order
    # is a reporting convention, not an algorithm. --model-concurrency>1 therefore gives
    # identical results for much less wall-clock. Default 1 keeps the spec'd ordering.
    msem = asyncio.Semaphore(max(1, a.model_concurrency))
    active = [m for m in MODEL_POOL if SHORT[m] in a.models] if a.models else MODEL_POOL

    async def one_model(model):
        async with msem:
            return await asyncio.gather(*[
                dynamic_search(runner, jc, sem, rec, model, r, a.tau, budget_of[model],
                               est_node[model], a.max_width, a.max_depth, a.min_width,
                               a.batch, a.depth_frontier, wmax, unit, task,
                               verifier=VERIFIER, patience=a.patience,
                               min_gain=a.min_gain, stop_signal=a.stop_signal,
                               readout=a.readout)
                for r in range(a.run_offset, a.run_offset + a.repeats)])

    all_runs = await asyncio.gather(*[one_model(m) for m in active])

    per_model = []
    for model, runs in zip(active, all_runs):
        entry = {
            "model": model, "short": SHORT[model], "n_runs": len(runs),
            "budget": round(budget_of[model], 4),
            "mean_cost_all": round(sum(r["cost"] for r in runs) / len(runs), 4),
            "mean_nodes": round(sum(r["nodes"] for r in runs) / len(runs), 2),
            "stops": dict(collections.Counter(r["stop"] for r in runs)),
            "over_budget_first": sum(r["over_budget_first_node"] for r in runs),
            "runs": runs,
        }
        if VERIFIER is not None:
            # The label is the ARGMAX answer, not a success probability. Repeats give the
            # search more chances to surface a high-scoring answer (best-of-n across runs);
            # they are not used to estimate P_solve.
            scored = [r for r in runs if r.get("best_verifier_score") is not None]
            if scored:
                bestrun = max(scored, key=lambda r: r["best_verifier_score"])
                entry.update({
                    "best_verifier_score": bestrun["best_verifier_score"],
                    "cost_of_best": bestrun["cost"],
                    "correct_of_best": bool(bestrun["solved"]),
                    "best_run": bestrun["run"],
                })
        else:
            ok = [r for r in runs if r["solved"]]
            costs_ok = [r["cost"] for r in ok]
            entry.update({
                "p_solve": len(ok) / len(runs), "n_success": len(ok),
                "mean_cost_success": round(sum(costs_ok) / len(costs_ok), 4) if ok else None,
                "min_cost_success": round(min(costs_ok), 4) if ok else None,
            })
        per_model.append(entry)

    selected = None
    if VERIFIER is not None:
        # The query-level answer: the globally highest verifier score across ALL models.
        # Scores are comparable across models -- it is one scalar function -- which is
        # exactly what plurality could never do.
        cand = [m for m in per_model if m.get("best_verifier_score") is not None]
        if cand:
            top = max(cand, key=lambda m: m["best_verifier_score"])
            selected = {"model": top["short"], "score": top["best_verifier_score"],
                        "cost": top["cost_of_best"], "correct": top["correct_of_best"]}
        solved_any = [m for m in per_model if m.get("correct_of_best")]
    else:
        solved_any = [m for m in per_model if m.get("n_success", 0) > 0]
    return {
        "id": rec["id"], "source": rec["source"], "prompt": prompt[:300],
        "tau": a.tau, "stop_signal": a.stop_signal, "repeats": a.repeats, "max_width": a.max_width,
        "max_depth": a.max_depth, "min_width": a.min_width,
        "batch": a.batch, "depth_frontier": a.depth_frontier,
        "unit_flops": unit_flops, "budget": round(budget, 4),
        "budget_basis": "analytical: node_cost(72B @ probe lengths) / node_cost(1B @ same)",
        "est_node_units": {SHORT[m]: round(v, 3) for m, v in est_node.items()},
        "probe_completion_tokens": probe.get("completion_tokens"),
        "working_max_tokens": wmax,
        # no model solved it in ANY run
        "hard_unsolved": not solved_any,
        "selected": selected,
        "per_model": per_model,
    }


# --------------------------------------------------------------------------- #
def load_pilot(a) -> List[Dict[str, Any]]:
    """Split-filtered, refusal-filtered, source-stratified pilot sample."""
    split = {}
    if os.path.exists(SPLITS):
        for l in open(SPLITS, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                split[r["id"]] = r["split"]

    rows = [json.loads(l) for l in open(a.gt, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ground_truth")]
    if a.split != "all":
        rows = [r for r in rows if split.get(r["id"]) == a.split]
    n_before = len(rows)
    if not a.keep_refusals:
        # `final_answer == "refusal"` flips the judge's rubric (correct == the candidate
        # also refuses), which is a different task from the correctness evaluation this
        # pilot is about. Dropping them is NOT uniform across sources -- it removes ~52%
        # of beaver_tails and ~16% of reward_bench -- so we stratify AFTER filtering to
        # stop the pilot silently rebalancing toward mix_instruct.
        rows = [r for r in rows if str(r.get("final_answer") or "").lower() != "refusal"]
    logger.info("pool: %d -> %d after refusal filter (split=%s)", n_before, len(rows), a.split)

    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    rng = random.Random(a.seed)
    pools = {}
    for src in sorted(by_src):                     # sorted -> deterministic
        rs = by_src[src][:]
        rng.shuffle(rs)
        pools[src] = rs

    # Round-robin across sources rather than a fixed n//len(sources) quota. A flat quota
    # silently under-fills the pilot when any source has fewer rows than the quota (the
    # refusal filter removes ~52% of beaver_tails, so this is a live risk), and leaves the
    # remaining sources unable to make up the shortfall. Round-robin keeps the split as
    # even as the data allows AND always reaches n while any source still has rows.
    sample: List[Dict[str, Any]] = []
    while len(sample) < a.n and any(pools.values()):
        for src in sorted(pools):
            if len(sample) >= a.n:
                break
            if pools[src]:
                sample.append(pools[src].pop())
    rng.shuffle(sample)
    logger.info("pilot: %d queries  %s", len(sample),
                dict(collections.Counter(r["source"] for r in sample)))
    return sample


async def main_async(a):
    global VERIFIER
    if a.stop_signal != "agreement" or a.readout == "verifier":
        from experiments_query.bestroute_rm.verifier import Verifier
        VERIFIER = Verifier(a.verifier_path, batch_size=a.verifier_batch,
                            max_length=a.verifier_max_length)
        logger.info("stop signal = %s (tau=%.2f, patience=%d, min_gain=%.3f); read-out = %s",
                    a.stop_signal.upper(), a.tau, a.patience, a.min_gain,
                    "argmax verifier" if a.readout != "plurality" else "plurality")
    else:
        logger.info("stop signal = AGREEMENT (tau=%.2f); read-out = plurality", a.tau)
    os.makedirs(a.outdir, exist_ok=True)
    cfg = Config(max_tokens=1024, width_temp=WIDTH_TEMP, depth_temp=DEPTH_TEMP)
    # gemma-3-4b is rate-limited far below the account's global limit and is the ONLY model
    # that 429s here. Retries back off inside the semaphores, so a few simultaneously
    # retrying calls hold their slots for minutes and stall the whole query -- observed as
    # 53- and 23-minute gaps between completed queries. Throttling just that model trades a
    # little of its throughput for far fewer stalls.
    if a.throttle_4b > 0:
        cfg.model_concurrency = {"google/gemma-3-4b-it": a.throttle_4b}
    else:
        cfg.model_concurrency = {}
    # Own cache: sharing the old builder's would make run 0 replay ITS draws.
    cfg.cache_path = os.path.join(a.outdir, "node_cache.json")
    cache = _load_cache(cfg.cache_path)
    jc_path = os.path.join(a.outdir, "judge_cache.json")
    jc = _load_cache(jc_path)
    sem = asyncio.Semaphore(a.concurrency)
    runner = NodeRunner(cfg, cache, sem)

    tag = (f"tau{a.tau}" if a.stop_signal == "agreement" and a.readout != "verifier"
           else f"{a.stop_signal}_ro-{a.readout}_p{a.patience}")
    if a.models:                                   # subset / extra-repeat runs get their own file
        tag += "_" + "-".join(a.models) + f"_runs{a.run_offset}-{a.run_offset + a.repeats - 1}"
    out_path = os.path.join(a.outdir, f"bestroute_dynamic_{tag}.jsonl")
    done = set()
    if os.path.exists(out_path) and not a.overwrite:
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    elif a.overwrite and os.path.exists(out_path):
        os.remove(out_path)

    todo = [r for r in load_pilot(a) if r["id"] not in done]
    if getattr(a, "ids_file", None):
        keep = set(json.load(open(a.ids_file)))
        todo = [r for r in todo if r["id"] in keep]
    logger.info("tau=%.2f repeats=%d width<=%d depth<=%d | %d queries todo (%d done)",
                a.tau, a.repeats, a.max_width, a.max_depth, len(todo), len(done))

    # Queries are independent (no shared RNG; the runner/judge caches are plain dicts
    # mutated only between awaits), so --query-concurrency>1 changes wall-clock only.
    # A query's record is written the moment it finishes; the caches are flushed every
    # --cache-save-every completions (the full-pool cache is ~100MB, so per-query dumps
    # would cost hours), plus once at the end. Losing a few queries' node generations
    # in a crash is harmless: their records are already on disk.
    qsem = asyncio.Semaphore(max(1, a.query_concurrency))
    n_done = {"n": 0}

    async def run_one(rec):
        async with qsem:
            res = await search_query(runner, jc, sem, rec, a)
        n_done["n"] += 1
        i = n_done["n"]
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        if i % max(1, a.cache_save_every) == 0:
            _save_cache(cfg.cache_path, cache)
            _save_cache(jc_path, jc)
        if "error" in res:
            logger.warning("  [%d/%d] %s -> %s", i, len(todo), rec["id"], res["error"])
        else:
            if res.get("selected"):                     # verifier mode: report the argmax
                sel = res["selected"]
                msg = (f"{sel['model']} score={sel['score']:.2f} @{sel['cost']:.1f} "
                       f"{'OK' if sel['correct'] else 'WRONG'}")
            else:
                best = min((m for m in res["per_model"] if m.get("n_success", 0) > 0),
                           key=lambda m: m["mean_cost_success"], default=None)
                msg = (f"{best['short']} p={best['p_solve']:.1f} @{best['mean_cost_success']:.1f}"
                       if best else "UNSOLVED by all models")
            logger.info("  [%d/%d] %s B=%.1f -> %s", i, len(todo), rec["id"], res["budget"], msg)

    await asyncio.gather(*[run_one(rec) for rec in todo])
    _save_cache(cfg.cache_path, cache)
    _save_cache(jc_path, jc)

    if runner.failures:
        logger.warning("%d node calls failed (not cached; rerun to retry them)", runner.failures)
    if JUDGE_EMPTY["n"]:
        logger.warning("%d judge calls returned empty -> graded by refusal heuristic, uncached",
                       JUDGE_EMPTY["n"])
    report(out_path)


def report(out_path: str):
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "error" not in r]
    if not rows:
        print("no records"); return
    verif = any(m.get("best_verifier_score") is not None
                for m in rows[0].get("per_model", []))
    head = (f"stop_signal={rows[0].get('stop_signal')}, repeats={rows[0]['repeats']}"
            if verif else f"tau={rows[0]['tau']}, repeats={rows[0]['repeats']}")
    print(f"\n=== dynamic best-route oracle search: {len(rows)} queries, {head} ===")
    print(f"  hard_unsolved (no model produced a correct answer): "
          f"{sum(r['hard_unsolved'] for r in rows)}")

    if not verif:
        print(f"\n  {'model':6} {'P_solve':>8} {'cost|success':>13} {'nodes':>7}  stops")
        for i, m in enumerate(MODEL_POOL):
            ms = [r["per_model"][i] for r in rows]
            ps = sum(x["p_solve"] for x in ms) / len(ms)
            cs = [x["mean_cost_success"] for x in ms if x["mean_cost_success"] is not None]
            st = collections.Counter()
            for x in ms:
                st.update(x["stops"])
            cost = f"{sum(cs)/len(cs):.2f}" if cs else "-"
            print(f"  {SHORT[m]:6} {ps:8.3f} {cost:>13} "
                  f"{sum(x['mean_nodes'] for x in ms)/len(ms):7.1f}  {dict(st)}")
        for bar in (0.5, 0.8, 1.0):
            pick = collections.Counter()
            for r in rows:
                ok = [m for m in r["per_model"] if m["p_solve"] >= bar]
                pick[min(ok, key=lambda m: m["mean_cost_success"])["short"] if ok else "none"] += 1
            print(f"\n  cheapest model with P_solve >= {bar}: {dict(pick.most_common())}")
        return

    # ---- verifier mode: the answer is the argmax, so report THAT ----
    sel = [r["selected"] for r in rows if r.get("selected")]
    acc = sum(1 for s_ in sel if s_["correct"]) / len(sel)
    cost = sum(s_["cost"] for s_ in sel) / len(sel)
    print(f"\n  SELECTED ANSWER (highest verifier score across all models)")
    print(f"    accuracy   : {acc:.3f}   over {len(sel)} queries")
    print(f"    mean cost  : {cost:.2f} units (the winning model's search only)")
    print(f"    model mix  : {dict(collections.Counter(s_['model'] for s_ in sel).most_common())}")

    tot = sum(sum(m["mean_cost_all"] for m in r["per_model"]) for r in rows) / len(rows)
    print(f"    full-search cost (all models, all repeats): {tot:.1f} units/query")

    print(f"\n  per model (its own best-scoring answer)")
    print(f"    {'model':6} {'acc':>6} {'cost':>8} {'nodes':>7}  stops")
    for i, m in enumerate(MODEL_POOL):
        ms = [r["per_model"][i] for r in rows if r["per_model"][i].get("best_verifier_score") is not None]
        if not ms:
            continue
        a = sum(1 for x in ms if x["correct_of_best"]) / len(ms)
        c = sum(x["cost_of_best"] for x in ms) / len(ms)
        st = collections.Counter()
        for x in ms:
            st.update(x["stops"])
        print(f"    {SHORT[m]:6} {a:6.3f} {c:8.2f} "
              f"{sum(x['mean_nodes'] for x in ms)/len(ms):7.1f}  {dict(st)}")

    # is the verifier's cross-model choice actually good? compare to always-72b etc.
    print(f"\n  reference points")
    for i, m in enumerate(MODEL_POOL):
        ms = [r["per_model"][i] for r in rows if r["per_model"][i].get("correct_of_best") is not None]
        if ms:
            a = sum(1 for x in ms if x["correct_of_best"]) / len(ms)
            print(f"    always {SHORT[m]:5}: acc={a:.3f}")
    anyc = sum(1 for r in rows
               if any(m.get("correct_of_best") for m in r["per_model"])) / len(rows)
    print(f"    oracle over models (any model's best answer correct): {anyc:.3f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", default=GT)
    p.add_argument("--n", type=int, default=500, help="pilot size (queries)")
    p.add_argument("--split", default="train", choices=["train", "test", "all"])
    p.add_argument("--tau", type=float, default=0.6,
                   help="agreement threshold; FIXED within an experiment, swept across them")
    p.add_argument("--repeats", type=int, default=5, help="independent runs per (query, model)")
    p.add_argument("--max-width", type=int, default=16, help="max width PER LAYER")
    p.add_argument("--max-depth", type=int, default=3, help="max layers")
    p.add_argument("--batch", type=int, default=8,
                   help="width samples drawn per agreement check (1 = strictly sequential)")
    p.add_argument("--stop-signal",
                   choices=["agreement", "verifier", "both_or", "both_and"], default="agreement",
                   help="verifier: stop a layer when a batch finds no better-scoring answer, "
                        "and read out argmax verifier score instead of plurality")
    p.add_argument("--readout", choices=["auto", "plurality", "verifier"], default="auto",
                   help="auto = plurality for agreement-stop, argmax verifier otherwise. "
                        "Set explicitly to separate the STOP effect from the READ-OUT effect.")
    p.add_argument("--verifier-path",
                   default="experiments_query/results/bestroute_rm/models/checkpoint-best")
    p.add_argument("--verifier-batch", type=int, default=16)
    p.add_argument("--verifier-max-length", type=int, default=512,
                   help="token truncation when scoring; MUST match the length the checkpoint "
                        "was trained at (512 for models/, 1024 for models_len1024_clean/)")
    p.add_argument("--patience", type=int, default=1,
                   help="consecutive no-improvement batches before stopping (verifier mode)")
    p.add_argument("--min-gain", type=float, default=0.0,
                   help="a batch counts as an improvement only if it beats the running best "
                        "verifier score by more than this")
    p.add_argument("--depth-frontier", type=int, default=4,
                   help="max DISTINCT prior solutions handed to a reconciliation node")
    p.add_argument("--min-width", type=int, default=2,
                   help="samples required before agreement may trigger STOP")
    p.add_argument("--keep-refusals", action="store_true",
                   help="keep final_answer=='refusal' rows (dropped by default)")
    p.add_argument("--concurrency", type=int, default=24, help="max concurrent API calls")
    p.add_argument("--tiered-budget", action="store_true",
                   help="per-model budget = one call of the next model up (1b<=8b, 8b<=27b, 27b<=72b)")
    p.add_argument("--models", default=None,
                   help="comma-separated SHORT names to search (default: all six), e.g. 1b,7b,8b")
    p.add_argument("--run-offset", type=int, default=0,
                   help="first run index; run indices are part of the node-cache key, so "
                        "--run-offset 1 --repeats 2 draws FRESH samples for runs 1 and 2")
    p.add_argument("--query-concurrency", type=int, default=1,
                   help="queries searched at the same time (results are order-independent)")
    p.add_argument("--cache-save-every", type=int, default=1,
                   help="flush node/judge caches every N finished queries")
    p.add_argument("--throttle-4b", type=int, default=0,
                   help="per-model concurrency cap for gemma-3-4b (0 = uncapped). It is the "
                        "only model that rate-limits here; retries stall the query.")
    p.add_argument("--model-concurrency", type=int, default=1,
                   help="models searched at once. Results are IDENTICAL for any value "
                        "(no pruning => models independent); >1 only cuts wall-clock.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ids-file", default=None, help="JSON list of query ids: restrict the run to these")
    p.add_argument("--outdir", default="experiments_query/results/bestroute_dynamic")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    a.models = [m.strip() for m in a.models.split(",")] if a.models else None

    if a.report_only:
        tag = f"tau{a.tau}" if a.stop_signal == "agreement" else f"verif_p{a.patience}"
        report(os.path.join(a.outdir, f"bestroute_dynamic_{tag}.jsonl")); return
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
