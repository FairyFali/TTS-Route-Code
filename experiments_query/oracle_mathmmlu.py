"""Label-free oracle search on MATH / MMLU with BOTH planners per (query, model).

For every query of the MATH/MMLU query list (datasets/splits/mathmmlu_qids.json) and
every model in the pool, run
  * tts-planner-w  -- the width-leaning planner (`build_bestroute_dynamic.dynamic_search`,
                      p1 recipe: tau .35, width<=16, depth<=3, batch 4, both_or stop,
                      verifier read-out), and
  * tts-planner-d  -- the depth-leaning planner (`depth_planner.one_chain`: W=2 +
                      self-feedback/revise chain, verifier-gated, patience 1),
under the GLOBAL per-query budget B (one 72b call, analytical; no tier cap).
The search is label-free (agreement + verifier only); the exact-match label is applied
AFTERWARDS to the verifier-argmax answer (MATH: boxed answer == target; MMLU: option
letter == target — the same `score` rule as the old library).

Prompts are the domain prompt set (`width_messages` / `fusion_messages`, MATH asks for
'The answer is: \\boxed{...}'), so layer-0 samples hit the old library's node cache when
the probe length matches (probe cap 4096, working cap = probe+512 clipped to [512, 4096]).

Output: <outdir>/<domain>.jsonl, one record per query:
  {"id", "source", "native_split", "budget", "unit_flops",
   "per_model": {"1b": {"w": <dynamic_search run>, "d": <one_chain record>}, ...}}
Resumable per domain. Run: python -m experiments_query.oracle_mathmmlu --domain math
"""
import argparse, asyncio, json, os, types

import experiments_query.build_bestroute_dynamic as bbd
import experiments_query.depth_planner as dpl
from swarm.llm.format import Message
from experiments_query.experiment_adaptive_tts import (
    Config, NodeRunner, _load_cache, _save_cache, _prompt_set, _role_constraint,
    width_messages, fusion_messages, answer_key as domain_answer_key)
from experiments_query.build_bestroute_dynamic import (
    MODEL_POOL, SHORT, UNIT_MODEL, LARGEST, WIDTH_TEMP, DEPTH_TEMP, est_input_tokens, Task)
from experiments_query.budget import MODEL_SPECS, node_cost

QIDS = "datasets/splits/mathmmlu_qids.json"      # the 1,250 MATH + 1,385 MMLU profiled queries ("domain|native_split|index")
PROBE_CAP, LEN_BUFFER, MIN_TOK, HARD_MAX = 4096, 512, 512, 4096      # old-library caps


class DomainRunner(NodeRunner):
    """NodeRunner that (1) stores layer-0 width samples under the old library's `width`
    kind so cached generations are reused, and (2) recomputes cost_units from the
    recorded token counts, because cached records carry the OLD cost unit."""

    async def run(self, qid, model, kind, depth, sample_idx, messages, task, unit_q,
                  temperature=None, max_tokens=None):
        if kind == "dyn0" and depth == 0:
            kind = "width"
        rec = await super().run(qid, model, kind, depth, sample_idx, messages, task, unit_q,
                                temperature, max_tokens)
        spec = MODEL_SPECS.get(model); pt = int(rec.get("prompt_tokens") or 0)
        flops = node_cost(pt, int(rec.get("completion_tokens") or 0), spec) if (spec and pt) else 0.0
        out = dict(rec); out["cost_units"] = (flops / unit_q) if unit_q else 0.0
        return out


def load_queries(domain):
    from experiments.evaluator.datasets.math_dataset import MATHDataset
    from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
    qids = json.load(open(QIDS))
    qids = [q for q in qids if q.split("|")[0] == domain]
    ds = {}
    recs = []
    for q in qids:
        _, split, idx = q.split("|")
        if split not in ds:
            # old-library qids carry the router label ("train"/"test"); for MMLU "train" is
            # the native `dev` split (57 subjects x 5 = 285), indexed under the same rng(888) order
            native = "dev" if (domain == "mmlu" and split == "train") else split
            ds[split] = MATHDataset(native) if domain == "math" else MMLUDataset(native)
        d = ds[split]; row = d[int(idx)]
        recs.append({"id": q, "source": domain, "native_split": split,
                     "prompt": d.record_to_swarm_input(row)["task"],
                     "ground_truth": d.record_to_target_answer(row), "final_answer": ""})
    return recs, next(iter(ds.values()))


async def main_async(a):
    from experiments_query.bestroute_rm.verifier import Verifier
    domain = a.domain
    recs, ds = load_queries(domain)
    ps = _prompt_set(domain)
    _, constraint = _role_constraint(ps, domain, refine=True)

    # ---- domain-specific prompts / keys / exact-match judge, patched into both planners ----
    bbd.gen_messages = lambda prompt: width_messages(ps, domain, prompt)
    bbd.fuse_messages = lambda prompt, outs: fusion_messages(ps, domain, prompt, outs)
    bbd.answer_key = lambda text: domain_answer_key(domain, ds, text or "")

    async def exact_judge(prompt, gt, fa, cand, sem, jc):
        try:
            return ds.postprocess_answer(cand or "") == gt
        except Exception:
            return False
    bbd.judge = exact_judge
    dpl.gen_messages = bbd.gen_messages
    dpl.judge = exact_judge
    _revise = dpl.revise_messages

    def revise_with_constraint(prompt, answer, feedback):
        msgs = _revise(prompt, answer, feedback)
        return [Message(role="system", content=msgs[0].content + " " + constraint)] + msgs[1:]
    dpl.revise_messages = revise_with_constraint

    verifier = Verifier(a.verifier_path, batch_size=a.verifier_batch, max_length=a.verifier_max_length)
    dpl.VERIFIER = verifier
    os.makedirs(a.outdir, exist_ok=True)
    cfg = Config(max_tokens=1024, width_temp=WIDTH_TEMP, depth_temp=DEPTH_TEMP)
    cfg.model_concurrency = {}
    cfg.cache_path = os.path.join(a.outdir, "node_cache.json")
    cache = _load_cache(cfg.cache_path)
    jc = {}
    sem = asyncio.Semaphore(a.concurrency)
    runner = DomainRunner(cfg, cache, sem)
    active = [m for m in MODEL_POOL if SHORT[m] in a.models.split(",")]
    out_path = os.path.join(a.outdir, f"{domain}{"_tier" if a.tiered else ""}{"" if a.planners == "w,d" else "_" + a.planners.replace(",", "")}{"" if a.stop_signal == "both_or" else "_" + a.stop_signal}{a.tag}.jsonl")
    done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
    if a.ids_json: keep = set(json.load(open(a.ids_json))); recs = [r for r in recs if r["id"] in keep]
    todo = [r for r in recs if r["id"] not in done][: a.n]
    print(f"oracle {domain}: {len(todo)} queries todo ({len(done)} done); models {[SHORT[m] for m in active]}", flush=True)
    qsem = asyncio.Semaphore(a.query_concurrency); nd = {"n": 0}
    dargs = types.SimpleNamespace(max_revisions=a.max_revisions, min_gain=0.0, patience=1)

    async def run_one(rec):
        async with qsem:
            task = Task(est_input_tokens(rec["prompt"]), 512)
            probe = await runner.run(rec["id"], UNIT_MODEL, "probe", 0, 0, bbd.gen_messages(rec["prompt"]),
                                     task, 1.0, temperature=WIDTH_TEMP, max_tokens=PROBE_CAP)
            unit = float(probe["cost_units"]) or 1.0
            n_p, n_d = int(probe.get("prompt_tokens") or 0), int(probe.get("completion_tokens") or 0)
            wmax = int(min(max(n_d + LEN_BUFFER, MIN_TOK), HARD_MAX))
            est = {m: node_cost(n_p, n_d, MODEL_SPECS[m]) / unit for m in MODEL_POOL}
            B = node_cost(n_p, n_d, MODEL_SPECS[LARGEST]) / unit
            per = {}
            by_short = {SHORT[m]: m for m in MODEL_POOL}; nxt = {"1b": "8b", "8b": "27b", "27b": "72b"}
            for m in active:
                Bm = est[by_short[nxt[SHORT[m]]]] if (a.tiered and not a.no_tier_cap and SHORT[m] in nxt) else B
                entry = {}
                if "w" in a.planners:
                    if a.tiered:   # deployed tiered recipe: width<=8, min_width 4, batch 1
                        w = await bbd.dynamic_search(runner, jc, sem, rec, m, 0, a.tau, Bm, est[m], 8, 3, 4, 1, 4,
                                                     wmax, unit, task, verifier=verifier, patience=1, min_gain=0.0,
                                                     stop_signal=a.stop_signal, readout="verifier")
                    else:
                        w = await bbd.dynamic_search(runner, jc, sem, rec, m, 0, 0.35, Bm, est[m], 16, 3, 2, 4, 4,
                                                     wmax, unit, task, verifier=verifier, patience=1, min_gain=0.0,
                                                     stop_signal="both_or", readout="verifier")
                    w.pop("layers", None); entry["w"] = w
                if "d" in a.planners:
                    entry["d"] = await dpl.one_chain(runner, jc, sem, rec, m, Bm, est[m], 2, wmax, unit, task, dargs)
                per[SHORT[m]] = entry
            res = {"id": rec["id"], "source": domain, "native_split": rec["native_split"],
                   "unit_flops": unit, "budget": round(B, 4), "per_model": per}
        nd["n"] += 1; i = nd["n"]
        with open(out_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        if i % a.cache_save_every == 0:
            _save_cache(cfg.cache_path, cache)
        msg = "  ".join(f"{s}:" + "/".join(f"{k}{'OK' if v['solved'] else 'x'}@{v['cost']:.1f}" for k, v in r.items()) for s, r in per.items())
        print(f"[{i}/{len(todo)}] {rec['id']} {msg}", flush=True)

    await asyncio.gather(*[run_one(r) for r in todo])
    _save_cache(cfg.cache_path, cache)
    print("DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=["math", "mmlu"], required=True)
    p.add_argument("--models", default="1b,8b,27b,72b")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--max-revisions", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--query-concurrency", type=int, default=12)
    p.add_argument("--cache-save-every", type=int, default=50)
    p.add_argument("--verifier-path", default="experiments_query/results/verifier_v2/models/checkpoint-best")
    p.add_argument("--verifier-batch", type=int, default=16)
    p.add_argument("--verifier-max-length", type=int, default=1024)
    p.add_argument("--outdir", default="experiments_query/results/oracle_mathmmlu")
    p.add_argument("--planners", default="w,d", help="w, d or w,d")
    p.add_argument("--tiered", action="store_true", help="tier caps (1b<=one 8b call, 8b<=one 27b call) and the deployed width recipe")
    p.add_argument("--tau", type=float, default=0.35, help="agreement threshold of the tiered width search (>1 = agreement never stops it)")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    p.add_argument("--ids-json", default="", help="restrict to these query ids (JSON list)")
    p.add_argument("--no-tier-cap", action="store_true", help="with --tiered: keep the deployed width recipe but use the GLOBAL budget B (ablation: no tier cap)")
    p.add_argument("--stop-signal", default="both_or", choices=["agreement", "verifier", "both_or", "both_and"], help="stop rule of the tiered width search (ablation: agreement = no verifier-stall stop)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
