"""COMPLETE PIPELINE, probe-first (user spec 2026-09-01), five models.

Per query, walk 1b -> 8b -> 27b -> 70b -> 72b:
  probe: ONE generation by the stage model; verifier v2 scores it.
  if score >= theta_model: the model is 'good' -> run its full TTS-Planner search; its
      read-out is the final answer. The probe IS the search's first draw (same cached
      sample), so the accepted stage adds no probe overhead -- probe cost + scoring are
      charged only for REJECTED stages.
  else: escalate. The last stage always runs its planner.
theta_model learned on TRAIN (quantile grid -> Pareto), evaluated FROZEN on test.
"""
from __future__ import annotations
import argparse, itertools, json, os
import numpy as np

CHAIN = ["1b", "8b", "27b", "70b", "72b"]
L = {"1b": "llama-3.2-1b-instruct", "8b": "llama-3.1-8b-instruct", "27b": "gemma-3-27b-it",
     "70b": "llama-3.1-70b-instruct", "72b": "qwen-2.5-72b-instruct"}
VC = 0.48
GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"
CK = "experiments_query/results/verifier_v2/models/checkpoint-best"


def probes(cache_path):
    cache = json.load(open(cache_path, encoding="utf-8"))
    out = {}
    for k, v in cache.items():
        p = k.split("|")
        if len(p) >= 5 and p[2] == "dyn0" and p[3] == "d0" and p[4] == "s0":
            out[(p[0], p[1].split("/")[-1])] = v
    return out


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


def get_scores(tag, qids, pr, gt, npz_reuse, outpath):
    S = {}
    if os.path.exists(npz_reuse):
        z = np.load(npz_reuse)
        for s in ("1b", "8b", "27b"):
            if s in z.files and len(z[s]) == len(qids):
                S[s] = z[s]
    need = [s for s in CHAIN[:-1] if s not in S]
    if need:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(CK)
        mdl = AutoModelForSequenceClassification.from_pretrained(CK, num_labels=1).cuda().eval()
        for s in need:
            tx = [f"Human: {gt[q]['prompt']} Assistant: {pr[(q, L[s])]['output'] or ''}" for q in qids]
            o = []
            with torch.no_grad():
                for i in range(0, len(tx), 32):
                    e = tok(tx[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                    o += mdl(**e).logits[:, 0].float().cpu().tolist()
            S[s] = np.array(o)
            print(f"scored {tag}/{s}", flush=True)
    np.savez(outpath, **S)
    return S


def evaluate(qids, S, PC, sY, sC, ths):
    n = len(qids)
    acc = np.empty(n); cost = np.zeros(n); done = np.zeros(n, bool); at = np.zeros(n, int)
    for si, s in enumerate(CHAIN):
        live = ~done
        if not live.any():
            break
        if si < len(CHAIN) - 1:
            ok = live & (S[s] >= ths[si])
            rej = live & ~ok
            cost[rej] += PC[s][rej] + VC          # rejected stages pay probe + scoring
            acc[ok] = sY[s][ok]; cost[ok] += sC[s][ok]; at[ok] = si; done |= ok
        else:
            acc[live] = sY[s][live]; cost[live] += sC[s][live]; at[live] = si; done |= live
    return float(acc.mean()), float(cost.mean()), at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/pipeline_probe_router")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    gt = {r["id"]: r for r in (json.loads(l) for l in open(GT, encoding="utf-8") if l.strip())}
    sides = {}
    for tag, cache, recs, reuse in (
            ("train", "experiments_query/results/oracle_verifier_full/node_cache.json",
             "experiments_query/results/oracle_v2_full/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
             "experiments_query/results/cascade_router/train_scores.npz"),
            ("test", "experiments_query/results/oracle_v2_test/node_cache.json",
             "experiments_query/results/oracle_v2_test/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
             "experiments_query/results/cascade_router/test_scores.npz")):
        pr = probes(cache); sY, sC = searches(recs)
        qids = sorted(q for q in sY["1b"] if all((q, L[s]) in pr for s in CHAIN))
        S = get_scores(tag, qids, pr, gt, reuse, os.path.join(a.outdir, f"{tag}_probe_scores.npz"))
        PC = {s: np.array([float(pr[(q, L[s])]["cost_units"]) for q in qids]) for s in CHAIN}
        sides[tag] = (qids, {s: S[s] for s in CHAIN[:-1]}, PC,
                      {s: np.array([sY[s][q] for q in qids]) for s in CHAIN},
                      {s: np.array([sC[s][q] for q in qids]) for s in CHAIN})
        print(f"{tag}: n={len(qids)}", flush=True)
    tr, te = sides["train"], sides["test"]
    qgrid = np.arange(0.05, 1.0, 0.05)
    masks = {s: {qq: tr[1][s] >= np.quantile(tr[1][s], qq) for qq in qgrid} for s in CHAIN[:-1]}
    cand = []
    for qs in itertools.product(qgrid, repeat=len(CHAIN) - 1):
        ths = [float(np.quantile(tr[1][s], qq)) for s, qq in zip(CHAIN, qs)]
        acc, cost, _ = evaluate(tr[0], tr[1], tr[2], tr[3], tr[4], ths)
        cand.append({"q": [round(float(x), 2) for x in qs], "ths": ths, "train_acc": acc, "train_cost": cost})
    print(f"grid {len(cand)} configs", flush=True)
    pareto = [c for c in cand if not any(o["train_acc"] >= c["train_acc"] and o["train_cost"] < c["train_cost"] or
                                         o["train_acc"] > c["train_acc"] and o["train_cost"] <= c["train_cost"] for o in cand)]
    pareto.sort(key=lambda c: c["train_cost"])
    keep = [pareto[i] for i in np.unique(np.linspace(0, len(pareto) - 1, 15).astype(int))]
    print(f"\n{'q(1b,8b,27b,70b)':>22} | {'TRAIN acc@cost':>15} | {'TEST acc@cost':>14} | planner runs at (test)")
    for c in pareto:
        acc, cost, at = evaluate(te[0], te[1], te[2], te[3], te[4], c["ths"])
        c["test_acc"], c["test_cost"] = acc, cost
        if c in keep:
            mixs = {CHAIN[d]: int((at == d).sum()) for d in range(len(CHAIN)) if (at == d).any()}
            print(f"{str(c['q']):>22} | {c['train_acc']:.3f}@{c['train_cost']:6.1f} | {acc:.3f}@{cost:6.1f} | {mixs}")
    json.dump(pareto, open(os.path.join(a.outdir, "pipeline_probe_router.json"), "w"), indent=1)
    print(f"\nsaved -> {a.outdir}/pipeline_probe_router.json")


if __name__ == "__main__":
    main()
