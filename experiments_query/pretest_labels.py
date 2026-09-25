"""Shared label builder for the pre-tests: per query, the cheapest-correct action over the
combined action set {single call, tts-planner-w, tts-planner-d} x {1b, 8b, 27b, 72b}, global
budget B, VC-free costs.  Returns model / shape / shape-type labels.

Sides: "train" (oracle_v2_full p1 + depth_planner_train + judged_samples) and
       "test"  (oracle_v2_test p1 + depth_planner_test + test judge cache).
Repeat runs for the stability subset: side="stab", run in {0,1,2} (pretest_stability dir).
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

VC = 0.48
MODELS = ["1b", "8b", "27b", "72b"]
FULL = {"meta-llama/llama-3.2-1b-instruct": "1b", "meta-llama/llama-3.1-8b-instruct": "8b",
        "google/gemma-3-27b-it": "27b", "qwen/qwen-2.5-72b-instruct": "72b"}
GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"
SHAPES = ["single", "W3-4", "W5-8", "W9+", "W+fusion", "W2+R1", "W2+R2+"]


def shape_of_w(run):
    if run["layers_used"] > 1:
        return "W+fusion"
    n = run["nodes"]
    return "single" if n == 1 else "W2" if n == 2 else "W3-4" if n <= 4 else "W5-8" if n <= 8 else "W9+"


def shape_of_d(d):
    r = d["revisions"]
    if r == 0:
        return "single" if int(d.get("width_drawn", 2)) == 1 else "W2"
    return "W2+R1" if r == 1 else "W2+R2+"


def load_w(path, run=0):
    out = {}
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        for m in r["per_model"]:
            runs = [x for x in m["runs"] if x["run"] == run] or m["runs"][:1]
            x = runs[0]
            out[(r["id"], m["short"])] = {"solved": bool(x["solved"]),
                                          "cost": max(float(x["cost"]) - VC * int(x.get("n_verifier_scored") or 0), 0.0),
                                          "nodes": int(x["nodes"]), "layers_used": int(x["layers_used"]),
                                          "agreement": float(x.get("final_agreement") or 0.0),
                                          "best_score": x.get("best_verifier_score")}
    return out


def load_d(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        for m, d in r["per_model"].items():
            out[(r["id"], m)] = d
    return out


def single_train(qids, run=0):
    """first width sample of each model (judged) on the train side."""
    out = {}
    kind = f"dyn{run}"
    for l in open("experiments_query/results/verifier_v2/judged_samples.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["kind"] == kind and r["depth"] == 0 and r["sample_idx"] == 0 and r["model"] in FULL:
            out[(r["qid"], FULL[r["model"]])] = (bool(r["correct"]), float(r["cost_units"]))
    return out


def single_from_cache(cache_path, judge_path, qids):
    """first width sample of each model from a node cache + judge cache (test side)."""
    from experiments_query.build_bestroute_tts import _h, ORACLE_MODEL, is_refusal
    gt = {r["id"]: r for r in (json.loads(l) for l in open(GT, encoding="utf-8") if l.strip())}
    cache = json.load(open(cache_path)); jc = json.load(open(judge_path))
    out = {}
    want = set(qids)
    for k, v in cache.items():
        p = k.split("|")
        if len(p) < 8 or p[2] != "dyn0" or p[3] != "d0" or p[4] != "s0" or p[1] not in FULL or p[0] not in want:
            continue
        g = gt[p[0]]
        key = _h(ORACLE_MODEL, g["prompt"][:80], (v.get("output") or "")[:200])
        if key in jc:
            c = bool(jc[key])
        elif str(g.get("final_answer") or "").lower() == "refusal":
            c = is_refusal(v.get("output") or "")
        else:
            continue
        out[(p[0], FULL[p[1]])] = (c, float(v["cost_units"]))
    return out


def single_stab(cache_path, qids, run):
    """first sample of run `run` on the stability subset, judged via the stability judge cache."""
    return single_from_cache(cache_path, "experiments_query/results/pretest_stability/judge_cache.json", qids) if run == 0 else None


def actions(side, run=0):
    """{qid: [(cost, model, shape, kind)]} for all correct actions, plus the list of qids."""
    if side == "train":
        W = load_w("experiments_query/results/oracle_v2_full/bestroute_dynamic_both_or_ro-verifier_p1.jsonl")
        D = load_d("experiments_query/results/depth_planner_train/depth_planner_global_w2.jsonl")
        qids = sorted({q for q, _ in D})
        S = single_train(qids)
    elif side == "test":
        W = load_w("experiments_query/results/oracle_v2_test/bestroute_dynamic_both_or_ro-verifier_p1.jsonl")
        D = load_d("experiments_query/results/depth_planner_test/depth_planner_global_w2.jsonl")
        qids = sorted({q for q, _ in D})
        S = single_from_cache("experiments_query/results/oracle_v2_test/node_cache.json",
                              "experiments_query/results/oracle_v2_test/judge_cache.json", qids)
    else:
        d = "experiments_query/results/pretest_stability"
        W = {}
        for wp in sorted(f for f in os.listdir(d) if f.startswith("bestroute_dynamic_") and f.endswith(".jsonl")):
            lo, hi = map(int, wp.rsplit("_runs", 1)[1][:-6].split("-"))
            if lo <= run <= hi:
                W = load_w(os.path.join(d, wp), run=run)
        D = load_d(os.path.join(d, f"depth_planner_global_w2{'' if run == 0 else f'_run{run}'}.jsonl"))
        qids = json.load(open(os.path.join(d, "subset_ids.json")))
        # single call = first width sample of this repeat, judged by pretest_judge_first_samples.py
        fj = json.load(open(os.path.join(d, "first_sample_judgments.json")))
        S = {(q, m): (bool(v["correct"]), float(v["cost"])) for k, v in fj.items()
             for q, m, r in [tuple(k.split("||"))] if int(r) == run}
    acts = {}
    for q in qids:
        a = []
        for m in MODELS:
            if (q, m) in S and S[(q, m)][0]:
                a.append((S[(q, m)][1], m, "single", "single"))
            # type is a function of the realized shape: a 1-node planner run IS a single call
            w = W.get((q, m))
            if w and w["solved"]:
                sh = shape_of_w(w); a.append((w["cost"], m, sh, "single" if sh == "single" else "width"))
            dd = D.get((q, m))
            if dd and dd["solved"]:
                sh = shape_of_d(dd)
                a.append((float(dd["cost"]), m, sh, "depth" if sh.startswith("W2+R") else "single" if sh == "single" else "width"))
        acts[q] = sorted(a)
    return qids, acts, W, D, S


def labels(side, run=0):
    qids, acts, W, D, S = actions(side, run)
    lab = {}
    for q in qids:
        if acts[q]:
            c, m, sh, ty = acts[q][0]
            lab[q] = {"model": m, "shape": sh, "type": ty, "cost": c, "solved_models": sorted({x[1] for x in acts[q]})}
        else:
            lab[q] = {"model": "none", "shape": "none", "type": "none", "cost": None, "solved_models": []}
    return qids, lab


if __name__ == "__main__":
    import collections
    for side in ("train", "test"):
        try:
            qids, lab = labels(side)
        except FileNotFoundError as e:
            print(side, "not ready:", e); continue
        print(side, len(qids), "model:", dict(collections.Counter(l["model"] for l in lab.values())),
              "type:", dict(collections.Counter(l["type"] for l in lab.values())))
