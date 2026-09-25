"""PLANNER-AS-NEEDED pipeline (user spec 2026-09-01), five models.

Per stage (1b -> 8b -> 27b -> 70b -> 72b): one probe call, verifier v2 score v.
    v >= theta_hi : accept the PROBE answer as final          (cost: probe + scoring)
    theta_lo <= v < theta_hi : model is good but uncertain -> run its PLANNER search
                               (probe is the search's first draw; no double count)
    v < theta_lo : escalate                                    (probe + scoring wasted)
Last stage always answers (its search = one call). theta_hi = theta_lo collapses to the
one-call cascade; theta_hi = inf collapses to the probe-first always-planner pipeline --
training chooses per stage where the planner earns its premium. Both thresholds are
learned on TRAIN (grid -> Pareto) and evaluated FROZEN on the held-out test set.
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np

CHAIN = ["1b", "8b", "27b", "70b", "72b"]          # overridden by --chain
L = {"1b": "llama-3.2-1b-instruct", "8b": "llama-3.1-8b-instruct", "27b": "gemma-3-27b-it",
     "70b": "llama-3.1-70b-instruct", "72b": "qwen-2.5-72b-instruct"}
VC = 0.48
GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"


def probes(cache_path):
    cache = json.load(open(cache_path, encoding="utf-8"))
    out, key_of = {}, {}
    for k, v in cache.items():
        p = k.split("|")
        if len(p) >= 5 and p[2] == "dyn0" and p[3] == "d0" and p[4] == "s0":
            out[(p[0], p[1].split("/")[-1])] = v; key_of[(p[0], p[1].split("/")[-1])] = k
    return out, key_of


def searches(path):
    Y, C = {}, {}
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        for m in r["per_model"]:
            run = m["runs"][0]
            Y.setdefault(m["short"], {})[r["id"]] = bool(run["solved"])
            C.setdefault(m["short"], {})[r["id"]] = float(run["cost"])
    return Y, C


def side(tag, cache_path, recs, score_npz, gt, probe_verdict):
    pr, key_of = probes(cache_path)
    sY, sC = searches(recs)
    qids = sorted(q for q in sY["1b"] if all((q, L[s]) in pr for s in CHAIN))
    z = np.load(score_npz)
    S = {s: z[s] for s in CHAIN[:-1]}
    assert all(len(S[s]) == len(qids) for s in S)
    PC = {s: np.array([float(pr[(q, L[s])]["cost_units"]) for q in qids]) for s in CHAIN}
    PY = {s: np.array([probe_verdict(key_of[(q, L[s])], q, pr[(q, L[s])]) for q in qids]) for s in CHAIN[:-1]}
    print(f"{tag}: n={len(qids)}  probe acc: " + " ".join(f"{s}={PY[s].mean():.3f}" for s in CHAIN[:-1]), flush=True)
    return (qids, S, PC, PY,
            {s: np.array([sY[s][q] for q in qids]) for s in CHAIN},
            {s: np.array([sC[s][q] for q in qids]) for s in CHAIN})


def evaluate(S, PC, PY, sY, sC, los, his):
    n = len(next(iter(PC.values())))
    acc = np.empty(n); cost = np.zeros(n); done = np.zeros(n, bool)
    mode = np.full(n, "", dtype=object)
    for si, s in enumerate(CHAIN):
        live = ~done
        if not live.any():
            break
        if si < len(CHAIN) - 1:
            hi = live & (S[s] >= his[si])
            band = live & ~hi & (S[s] >= los[si])
            rej = live & ~hi & ~band
            acc[hi] = PY[s][hi]; cost[hi] += PC[s][hi] + VC; mode[hi] = f"{s}-probe"
            acc[band] = sY[s][band]; cost[band] += sC[s][band]; mode[band] = f"{s}-planner"
            cost[rej] += PC[s][rej] + VC
            done |= hi | band
        else:
            acc[live] = sY[s][live]; cost[live] += sC[s][live]; mode[live] = f"{s}"
            done |= live
    return float(acc.mean()), float(cost.mean()), mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/pipeline_planner_as_needed")
    ap.add_argument("--chain", default=None)
    a = ap.parse_args()
    if a.chain:
        global CHAIN
        CHAIN = [x.strip() for x in a.chain.split(",")]
    os.makedirs(a.outdir, exist_ok=True)
    gt = {r["id"]: r for r in (json.loads(l) for l in open(GT, encoding="utf-8") if l.strip())}
    jv = {}
    for l in open("experiments_query/results/verifier_v2/judged_samples.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); jv[r["key"]] = bool(r["correct"])
    from experiments_query.build_bestroute_tts import _h, ORACLE_MODEL, is_refusal
    jc = json.load(open("experiments_query/results/oracle_v2_test/judge_cache.json", encoding="utf-8"))
    def tv(k, q, v):
        key = _h(ORACLE_MODEL, gt[q]["prompt"][:80], (v["output"] or "")[:200])
        if key in jc:
            return bool(jc[key])
        return is_refusal(v["output"] or "") if str(gt[q].get("final_answer") or "").lower() == "refusal" else False
    tr = side("train", "experiments_query/results/oracle_verifier_full/node_cache.json",
              "experiments_query/results/oracle_v2_full/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
              "experiments_query/results/pipeline_probe_router/train_probe_scores.npz", gt, lambda k, q, v: jv[k])
    te = side("test", "experiments_query/results/oracle_v2_test/node_cache.json",
              "experiments_query/results/oracle_v2_test/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
              "experiments_query/results/pipeline_probe_router/test_probe_scores.npz", gt, tv)
    _, trS, trPC, trPY, trY, trC = tr
    _, teS, tePC, tePY, teY, teC = te
    LO = np.arange(0.05, 1.0, 0.1)                       # 10 per gate
    HB = [0.0, 0.7, 0.9, 1.1]                            # hi = quantile(band) with 0.0 -> hi=lo (one-call), 1.1 -> inf
    qth = {s: {qq: float(np.quantile(trS[s], qq)) for qq in np.unique(np.concatenate([LO, [0.7, 0.9]]))} for s in CHAIN[:-1]}
    accs = []; costs = []; params = []
    NG = len(CHAIN) - 1
    for los_q in itertools.product(LO, repeat=NG):
        los = [qth[s][q] for s, q in zip(CHAIN, los_q)]
        for hb in itertools.product(HB, repeat=NG):
            his = [los[i] if hb[i] == 0.0 else (np.inf if hb[i] > 1.0 else max(los[i], qth[CHAIN[i]][hb[i]])) for i in range(NG)]
            acc, cost, _ = evaluate(trS, trPC, trPY, trY, trC, los, his)
            accs.append(acc); costs.append(cost); params.append((los_q, hb))
    accs = np.array(accs); costs = np.array(costs)
    print(f"grid {len(accs)} configs", flush=True)
    order = np.argsort(costs)
    best = -1.0; keep_idx = []
    for i in order:
        if accs[i] > best + 1e-12:
            best = accs[i]; keep_idx.append(int(i))
    print(f"train Pareto {len(keep_idx)}", flush=True)
    sel = [keep_idx[i] for i in np.unique(np.linspace(0, len(keep_idx) - 1, 16).astype(int))]
    out = []
    print(f"\n{'lo q':>22} {'hi band':>18} | {'TRAIN':>13} | {'TEST':>13} | test mode mix (top)")
    for i in keep_idx:
        los_q, hb = params[i]
        los = [qth[s][q] for s, q in zip(CHAIN, los_q)]
        his = [los[j] if hb[j] == 0.0 else (np.inf if hb[j] > 1.0 else max(los[j], qth[CHAIN[j]][hb[j]])) for j in range(len(CHAIN) - 1)]
        acc, cost, mode = evaluate(teS, tePC, tePY, teY, teC, los, his)
        rec = {"lo_q": [round(float(x), 2) for x in los_q], "hi_band": list(hb), "los": los,
               "his": [None if np.isinf(h) else h for h in his],
               "train_acc": float(accs[i]), "train_cost": float(costs[i]), "test_acc": acc, "test_cost": cost}
        out.append(rec)
        if i in sel:
            import collections as cl
            mx = cl.Counter(mode); top = "  ".join(f"{k}:{v}" for k, v in mx.most_common(4))
            print(f"{str(rec['lo_q']):>22} {str(hb):>18} | {accs[i]:.3f}@{costs[i]:6.1f} | {acc:.3f}@{cost:6.1f} | {top}")
    json.dump(out, open(os.path.join(a.outdir, "results.json"), "w"), indent=1)
    print(f"\nsaved -> {a.outdir}/results.json", flush=True)


if __name__ == "__main__":
    main()
