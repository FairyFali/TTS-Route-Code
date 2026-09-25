"""tts-planner-d: depth-leaning TTS planner (self-refine chain) for the profiling library.

Per (query, model): draw a FIXED width of W=2 samples (cached layer-0 nodes), let the
verifier pick the best, then repeatedly SELF-REVISE it — a feedback call (the model
critiques its own answer) followed by a refine call (rewrite given the feedback) — and
keep the revision only if its verifier score improves on the running best. Stop when
the verifier stops improving (`--patience`), the budget is reached, or `--max-revisions`.
Read-out: verifier-argmax over everything drawn. Cost: generation FLOPs of every call
(width samples, feedback, refine); verifier free (TTS-only / accept-charged protocol).

Budget per model: `--budget tier` = the width planner's tiered cap (one call of the next
tier up, analytical); `--budget global` = B = one 72b call. Same basis as
build_bestroute_dynamic so the outcomes are comparable with the width-planner labels.

Output: <outdir>/depth_planner_<budget>_w<W>.jsonl, one record per query with per-model
{solved, cost, nodes, revisions, accepted, stop, solved_w0, cost_w0}. Resumable.
"""
import argparse, asyncio, json, os

from swarm.llm.format import Message
from experiments_query.experiment_adaptive_tts import Config, NodeRunner, _load_cache, _save_cache
from experiments_query.build_bestroute_tts import gen_messages, judge
from experiments_query.build_bestroute_dynamic import (
    MODEL_POOL, SHORT, UNIT_MODEL, LARGEST, WIDTH_TEMP, DEPTH_TEMP, est_input_tokens, Task, load_pilot)
from experiments_query.budget import MODEL_SPECS, node_cost

VERIFIER = None
NEXT_TIER = {"1b": "8b", "8b": "27b", "27b": "72b", "70b": "72b", "72b": None}


def feedback_messages(prompt, answer):
    return [Message(role="system", content="You are a rigorous reviewer. Examine the ANSWER to the "
                    "REQUEST and write concise, specific feedback: factual or logical errors, missing "
                    "requirements, unsafe content, or code bugs. Do NOT rewrite the answer. If it is "
                    "already correct and complete, say so in one line."),
            Message(role="user", content=f"REQUEST:\n{prompt}\n\nANSWER:\n{answer}\n\nFEEDBACK:")]


def revise_messages(prompt, answer, feedback):
    return [Message(role="system", content="You are a careful expert assistant. Using the FEEDBACK, "
                    "produce an improved, correct and complete final answer to the REQUEST. Output "
                    "only the final answer. If the request is unsafe or harmful, refuse briefly."),
            Message(role="user", content=f"REQUEST:\n{prompt}\n\nPRIOR ANSWER:\n{answer}\n\n"
                    f"FEEDBACK:\n{feedback}\n\nIMPROVED FINAL ANSWER:")]


async def one_chain(runner, jc, sem, rec, model, budget, est, W, wmax, unit, task, a):
    prompt, qid = rec["prompt"], rec["id"]
    gt, fa = rec["ground_truth"], rec.get("final_answer", "")
    # ---- layer 0: width W (cache hits for the profiled queries). Same rule as the width
    # planner: the first node is always drawn (so "cannot afford" is never recorded as
    # "cannot solve"); each further width sample only if the projected cost still fits. ----
    nodes = []; cost = 0.0
    run = int(getattr(a, "run", 0)); sfx = "" if run == 0 else str(run)   # run>0: fresh samples / revisions
    for i in range(W):
        if i > 0 and cost + max(est, max(float(n["cost_units"]) for n in nodes)) > budget:
            break
        n = await runner.run(qid, model, f"dyn{run}", 0, i, gen_messages(prompt), task, unit,
                             temperature=WIDTH_TEMP, max_tokens=wmax)
        nodes.append(n); cost += float(n["cost_units"])
    outs = [n["output"] or "" for n in nodes]; drawn = len(nodes)
    sc = VERIFIER.score(prompt, outs)
    best_i = max(range(drawn), key=lambda i: sc[i]); best_score, best = sc[best_i], outs[best_i]
    solved_w0 = bool(await judge(prompt, gt, fa, best, sem, jc)) if best.strip() else False
    cost_w0 = cost
    # ---- self-revision chain ----
    revisions = accepted = stale = 0; stop = "max_revisions"; proj = 2.0 * est
    for r in range(1, a.max_revisions + 1):
        if cost >= budget or budget - cost < proj:
            stop = "budget"; break
        fb = await runner.run(qid, model, "sr_fb" + sfx, r, 0, feedback_messages(prompt, best), task, unit,
                              temperature=DEPTH_TEMP, max_tokens=min(wmax, 768))
        rv = await runner.run(qid, model, "sr_rv" + sfx, r, 0, revise_messages(prompt, best, fb["output"] or ""),
                              task, unit, temperature=DEPTH_TEMP, max_tokens=wmax)
        step = float(fb["cost_units"]) + float(rv["cost_units"])
        cost += step; drawn += 2; revisions += 1; proj = max(proj, step)
        text = rv["output"] or ""
        v = VERIFIER.score(prompt, [text])[0]
        if v > best_score + a.min_gain:
            best_score, best = v, text; accepted += 1; stale = 0
        else:
            stale += 1
            if stale >= a.patience:
                stop = "no_improvement"; break
    solved = solved_w0 if best is outs[best_i] else (bool(await judge(prompt, gt, fa, best, sem, jc)) if best.strip() else False)
    return {"solved": solved, "cost": round(cost, 4), "nodes": drawn, "width_drawn": len(outs), "revisions": revisions,
            "accepted": accepted, "stop": stop, "best_verifier_score": round(best_score, 4),
            "solved_w0": solved_w0, "cost_w0": round(cost_w0, 4), "budget": round(budget, 3)}


async def main_async(a):
    global VERIFIER
    from experiments_query.bestroute_rm.verifier import Verifier
    VERIFIER = Verifier(a.verifier_path, batch_size=a.verifier_batch, max_length=a.verifier_max_length)
    os.makedirs(a.outdir, exist_ok=True)
    cfg = Config(max_tokens=1024, width_temp=WIDTH_TEMP, depth_temp=DEPTH_TEMP)
    cfg.model_concurrency = {}
    cfg.cache_path = os.path.join(a.outdir, "node_cache.json")
    cache = _load_cache(cfg.cache_path)
    jc_path = os.path.join(a.outdir, "judge_cache.json"); jc = _load_cache(jc_path)
    sem = asyncio.Semaphore(a.concurrency)
    runner = NodeRunner(cfg, cache, sem)
    active = [m for m in MODEL_POOL if SHORT[m] in a.models.split(",")]
    out_path = os.path.join(a.outdir, f"depth_planner_{a.budget}_w{a.width}{'' if a.run == 0 else f'_run{a.run}'}.jsonl")
    done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
    todo = [r for r in load_pilot(a) if r["id"] not in done]
    if a.ids_file:
        keep = set(json.load(open(a.ids_file))); todo = [r for r in todo if r["id"] in keep]
    print(f"depth-planner [{a.budget} budget, W={a.width}, patience {a.patience}]: "
          f"{len(todo)} queries todo ({len(done)} done)", flush=True)
    qsem = asyncio.Semaphore(a.query_concurrency); nd = {"n": 0}

    async def run_one(rec):
        async with qsem:
            task = Task(est_input_tokens(rec["prompt"]), 512)
            probe = await runner.run(rec["id"], UNIT_MODEL, "probe", 0, 0, gen_messages(rec["prompt"]),
                                     task, 1.0, temperature=WIDTH_TEMP, max_tokens=2048)
            unit = float(probe["cost_units"]) or 1.0
            wmax = int(min(max(int(probe.get("completion_tokens") or 256) + 512, 512), 2048))
            n_p, n_d = int(probe.get("prompt_tokens") or 0), int(probe.get("completion_tokens") or 0)
            est = {m: node_cost(n_p, n_d, MODEL_SPECS[m]) / unit for m in MODEL_POOL}
            B = node_cost(n_p, n_d, MODEL_SPECS[LARGEST]) / unit
            by_short = {SHORT[m]: m for m in MODEL_POOL}
            per = {}
            for m in active:
                nxt = NEXT_TIER.get(SHORT[m])
                budget = est[by_short[nxt]] if (a.budget == "tier" and nxt) else B
                per[SHORT[m]] = await one_chain(runner, jc, sem, rec, m, budget, est[m], a.width, wmax, unit, task, a)
            res = {"id": rec["id"], "source": rec["source"], "per_model": per}
        nd["n"] += 1; i = nd["n"]
        with open(out_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        if i % a.cache_save_every == 0:
            _save_cache(cfg.cache_path, cache); _save_cache(jc_path, jc)
        msg = "  ".join(f"{s}:{'OK' if r['solved'] else 'x'}@{r['cost']:.1f}/r{r['revisions']}" for s, r in per.items())
        print(f"[{i}/{len(todo)}] {rec['id'][:36]} {msg}", flush=True)

    await asyncio.gather(*[run_one(r) for r in todo])
    _save_cache(cfg.cache_path, cache); _save_cache(jc_path, jc)
    print("DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="1b,8b")
    p.add_argument("--n", type=int, default=4604)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-refusals", action="store_true")
    p.add_argument("--budget", choices=["tier", "global"], default="tier")
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--min-gain", type=float, default=0.0)
    p.add_argument("--max-revisions", type=int, default=8)
    p.add_argument("--run", type=int, default=0, help="repeat index: >0 draws fresh samples/revisions (new cache kinds)")
    p.add_argument("--ids-file", default=None, help="JSON list of query ids: restrict the run to these")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--query-concurrency", type=int, default=8)
    p.add_argument("--cache-save-every", type=int, default=50)
    p.add_argument("--verifier-path", default="experiments_query/results/verifier_v2/models/checkpoint-best")
    p.add_argument("--verifier-batch", type=int, default=16)
    p.add_argument("--verifier-max-length", type=int, default=1024)
    p.add_argument("--outdir", default="experiments_query/results/depth_planner_train")
    a = p.parse_args()
    from experiments_query.build_bestroute_dynamic import GT as _GT
    a.gt = _GT
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
