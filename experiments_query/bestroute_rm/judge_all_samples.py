"""Judge EVERY cached generation of the full oracle search -> correctness label per sample.

The search only judged each (query, model, run)'s read-out. For a correctness-targeted verifier
we need a verdict on every sample, so the verifier can learn to rank correct above incorrect
samples of the SAME prompt -- which is exactly its job at read-out time.

Reads the node cache of `--outdir` (all samples are router-TRAIN queries, so this is training
data for the verifier), reuses the search's judge cache (read-outs already graded), and writes
one line per sample to judged_samples.jsonl. Resumable: samples already in the output are skipped.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, time
from experiments_query.build_bestroute_tts import judge, _h, ORACLE_MODEL, JUDGE_EMPTY
from experiments_query.experiment_adaptive_tts import _load_cache, _save_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("judge_all")
GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"


async def main_async(a):
    gt = {r["id"]: r for r in (json.loads(l) for l in open(GT, encoding="utf-8") if l.strip())}
    cache = json.load(open(os.path.join(a.search_dir, "node_cache.json"), encoding="utf-8"))
    jc = _load_cache(a.judge_cache)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out, encoding="utf-8") if l.strip()}
    todo = [(k, v) for k, v in cache.items() if k not in done and (v.get("output") or "").strip()]
    logger.info("samples: %d cached, %d already judged, %d to do (judge cache %d)", len(cache), len(done), len(todo), len(jc))
    sem = asyncio.Semaphore(a.concurrency)
    n = {"done": 0, "fallback": 0}; t0 = time.time()
    out = open(a.out, "a", encoding="utf-8")

    async def one(k, v):
        qid, model, kind, d, s = k.split("|")[:5]
        r = gt[qid]
        key = _h(ORACLE_MODEL, r["prompt"][:80], (v["output"] or "")[:200])
        ok = await judge(r["prompt"], r["ground_truth"], r.get("final_answer", ""), v["output"], sem, jc)
        fallback = key not in jc
        n["done"] += 1; n["fallback"] += int(fallback)
        out.write(json.dumps({"key": k, "qid": qid, "source": r["source"], "model": model, "kind": kind,
                              "depth": int(d[1:]), "sample_idx": int(s[1:]), "correct": bool(ok),
                              "judge_fallback": fallback, "cost_units": v.get("cost_units"),
                              "prompt_tokens": v.get("prompt_tokens"), "completion_tokens": v.get("completion_tokens"),
                              "text": v["output"]}) + "\n")
        if n["done"] % 500 == 0:
            out.flush(); _save_cache(a.judge_cache, jc)
            rate = n["done"] / (time.time() - t0)
            logger.info("judged %d/%d  (%.1f/s, ETA %.0f min, fallbacks %d)", n["done"], len(todo), rate, (len(todo) - n["done"]) / max(rate, 1e-6) / 60, n["fallback"])

    await asyncio.gather(*[one(k, v) for k, v in todo])
    out.flush(); out.close(); _save_cache(a.judge_cache, jc)
    logger.info("DONE: %d judged, %d fallbacks (heuristic, uncached)", n["done"], n["fallback"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--search-dir", default="experiments_query/results/oracle_verifier_full")
    p.add_argument("--judge-cache", default="experiments_query/results/verifier_v2/judge_cache.json")
    p.add_argument("--out", default="experiments_query/results/verifier_v2/judged_samples.jsonl")
    p.add_argument("--concurrency", type=int, default=32)
    a = p.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
