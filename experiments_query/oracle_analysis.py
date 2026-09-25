"""
Offline ORACLE ANALYSIS of TTS search trajectories, using the oracle model
(DeepSeek-V4 via OpenRouter; formerly GPT-5.6 on Azure).

Reads the trajectories produced by build_tts_experience.py (tts_experience.jsonl)
and, for each query, asks the oracle model to analyze the COMPLETE search retrospectively
(ground-truth known) and answer: given the outcomes, what was the optimal TTS
action at each state -- when to keep widening, when width became unnecessary, when
to switch to depth, which leaves to deepen, when to stop, which model fits the
query, and how much compute it actually needed.

The oracle model is called through OpenRouter (OPENROUTER_API_KEY in .env); set
TTSFLY_JUDGE_MODEL to override the default. Output is a structured JSON
"oracle_analysis" appended per query to oracle_analysis.jsonl -- the distilled
query-specific TTS experience.

Usage
-----
  python experiments_query/oracle_analysis.py \
      --lib experiments_query/results/tts_experience/tts_experience.jsonl \
      --out experiments_query/results/tts_experience/oracle_analysis.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("oracle_analysis")

# The offline oracle/judge model. Previously GPT-5.6 on Azure (Entra-token auth, `az login`);
# now DeepSeek-V4 through OpenRouter, so the offline path needs only OPENROUTER_API_KEY and
# no Azure dependency. Override per run with TTSFLY_JUDGE_MODEL.
ORACLE_MODEL = os.environ.get("TTSFLY_JUDGE_MODEL", "deepseek/deepseek-v4-flash")

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        # Reuse the pool's OpenRouter constants so key/base-url/attribution stay in one place.
        from swarm.llm.openrouter import OPENROUTER_URL, OPENROUTER_API_KEY, OPENROUTER_HEADERS
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set; the oracle/judge cannot run.")
        _CLIENT = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY,
                         default_headers=OPENROUTER_HEADERS)
    return _CLIENT


def _call_sync(messages, max_tokens, model) -> str:
    r = _client().chat.completions.create(
        messages=messages, max_tokens=max_tokens, model=model, temperature=0.0)
    if not getattr(r, "choices", None):                   # provider error surfaced in the body
        raise RuntimeError(f"no choices in response: {str(getattr(r, 'error', r))[:200]}")
    return r.choices[0].message.content or ""


async def oracle_llm(messages, sem: asyncio.Semaphore, max_tokens=6000,
                     model=None) -> str:
    """Call the offline oracle/judge model. Returns "" only after all retries fail.

    NOTE for callers: an empty string is indistinguishable from a content-filtered reply,
    and several callers treat "" as a negative verdict. Check that the judge cache actually
    grows during a run rather than assuming success.
    """
    model = model or ORACLE_MODEL
    async with sem:
        for attempt in range(3):
            try:
                return await asyncio.to_thread(_call_sync, messages, max_tokens, model)
            except Exception as exc:                      # transient/rate-limit
                logger.warning("oracle_llm(%s) failed (try %d): %s",
                               model, attempt + 1, str(exc)[:160])
                await asyncio.sleep(2 * (attempt + 1))
        logger.error("oracle_llm(%s) gave up after 3 attempts -> returning empty", model)
        return ""


# Backwards-compatible aliases (the judge used to be GPT-5.6).
GPT56 = ORACLE_MODEL
gpt56 = oracle_llm


# --------------------------------------------------------------------------- #
# Compact the (large) trajectory into a token-efficient summary for the oracle model
# --------------------------------------------------------------------------- #
def _correct_answer_key(rec: Dict[str, Any]) -> Optional[str]:
    for t in rec.get("per_model", []):
        for n in t.get("nodes", []):
            if n.get("correct"):
                return n.get("answer_key")
    return None


def compact(rec: Dict[str, Any]) -> Dict[str, Any]:
    per = []
    for t in rec.get("per_model", []):
        name = t.get("model", "").split("/")[-1]
        if t.get("skipped"):
            per.append({"model": name, "skipped": True, "reason": t.get("reason"),
                        "min_cost": t.get("min_cost")})
            continue
        # compact step list: (layer, width, cost, new_answer, agg_correct, any_correct)
        steps = [[s["layer"], s["width"], s["cost"], int(s["new_answer"]),
                  int(s["agg_correct"]), int(s.get("any_correct", 0))]
                 for s in t.get("steps", [])]
        per.append({
            "model": name, "solved": t.get("solved"), "cost": t.get("cost"),
            "single_node_correct": t.get("single_node_correct"),
            "first_correct_cost": t.get("first_correct_cost"),
            "first_any_correct_cost": t.get("first_any_correct_cost"),
            "reason": t.get("reason"), "n_nodes": t.get("n_nodes"),
            "layers_answer_dist": t.get("layers_answer_dist"),
            "steps_[layer,width,cost,new,agg_ok,any_ok]": steps,
        })
    return {
        "qid": rec["qid"], "domain": rec["domain"],
        "query": rec["query"][:1200],
        "correct_answer": _correct_answer_key(rec),
        "difficulty": rec.get("difficulty"),
        "tags": {k: rec.get("tags", {}).get(k) for k in ("difficulty", "problem_type")},
        "optimal": rec.get("optimal"),
        "ceiling_correct": rec.get("ceiling_correct"),
        "per_model": per,
    }


_SYSTEM = (
    "You are an expert analyst of test-time compute-optimal scaling (TTS). You are "
    "given the COMPLETE offline search trajectory for one query, WITH the ground-truth "
    "answer known. Each model was grown as an adaptive graph: start from 1 node, widen "
    "the current layer (adding parallel samples) with a plurality aggregation check "
    "after each add, stop widening when answers stop changing, then deepen (refine "
    "diverse leaves) and widen again, until the plurality answer is correct or a budget "
    "cap is hit. Costs are in units of one smallest-model call. Analyze retrospectively: "
    "given how things turned out, what was the OPTIMAL policy."
)

_SCHEMA = """Return ONLY a JSON object with these keys:
{
  "tts_needed": true/false,                // was more than a single model call required
  "optimal_model": "<model label>",        // best model (cheapest that solves)
  "required_compute_units": <number>,      // min cost to reach the correct answer
  "width_useful": true/false,              // did adding parallel samples help
  "width_stop_at": <int or null>,          // width beyond which no new useful answers appeared
  "switch_to_depth": true/false,           // was depth (refinement) needed after width
  "depth_useful": true/false,
  "leaves_to_deepen": "<which answers/leaves were worth refining, briefly>",
  "stop_signal": "<the runtime signal that indicated it was time to stop>",
  "predictive_signals": ["<runtime signals that predicted eventual success>"],
  "wasted_actions": "<which expansions wasted compute, briefly>",
  "reusable_pattern": "<one-sentence transferable rule for similar queries>",
  "rationale": "<2-3 sentence justification grounded in the trajectory>"
}
No prose outside the JSON."""


def build_messages(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    user = ("TRAJECTORY (JSON):\n" + json.dumps(summary, ensure_ascii=False)
            + "\n\n" + _SCHEMA)
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


def parse_json(raw: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
async def run(args) -> None:
    rows = [json.loads(l) for l in open(args.lib, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(args.out) and not args.overwrite:
        for l in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["qid"])
            except Exception:
                pass
    elif args.overwrite and os.path.exists(args.out):
        os.remove(args.out)

    sem = asyncio.Semaphore(args.concurrency)
    todo = [r for r in rows if r["qid"] not in done]
    logger.info("analyzing %d/%d queries with GPT-5.6 (concurrency=%d)",
                len(todo), len(rows), args.concurrency)

    async def one(rec):
        summary = compact(rec)
        raw = await oracle_llm(build_messages(summary), sem, max_tokens=args.max_tokens)
        analysis = parse_json(raw)
        return {"qid": rec["qid"], "domain": rec["domain"], "split": rec.get("split"),
                "optimal": rec.get("optimal"), "ceiling_correct": rec.get("ceiling_correct"),
                "oracle_analysis": analysis,
                "oracle_raw": (None if analysis else raw[:2000])}

    n = 0
    # process in chunks so we write incrementally / stay resumable
    for i in range(0, len(todo), args.concurrency * 2):
        batch = todo[i:i + args.concurrency * 2]
        results = await asyncio.gather(*[one(r) for r in batch])
        with open(args.out, "a", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n += 1
        ok = sum(1 for r in results if r["oracle_analysis"])
        logger.info("  ... %d done (%d/%d parsed this batch)", n, ok, len(results))
    logger.info("done: %d analyses -> %s", n, args.out)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--lib", default="experiments_query/results/tts_experience/tts_experience.jsonl")
    p.add_argument("--out", default="experiments_query/results/tts_experience/oracle_analysis.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
