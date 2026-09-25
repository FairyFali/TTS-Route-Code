"""Task-specific benchmark on MATH / MMLU: ours (11-mode TTS-Router + Lagrangian decode, trained on the
task's train split) and re-fitted baselines, evaluated on the task's test split under strict budgets.

Data: cached generations of results/oracle_mathmmlu (width samples per model, tiered width runs for 1b/8b,
global-B width runs for 27b/72b, tts-planner-d chains); exact-match grading; per-query cost unit = one 1b
call on that query (unit_flops).  Verifier scores are computed here and cached.
Splits: MATH train = native train (750), test = native test (500); MMLU train = dev (285) + 600 test queries
(seed 0), test = remaining 500.
Baselines (all fitted on the task's train split): vanilla single calls; BoN-verifier / SC / Self-Refine on
1b, 8b; query-only binary router 8b/72b (RouteLLM / Hybrid-LLM style); BEST-Route-style router over
(model, n) best-of-n actions with a threshold decode; verifier-gated cascades (AutoMix-style fixed 8b->27b->72b,
FrugalGPT-style learned sequence) with train-selected thresholds.
Run: python -m experiments_query.task_bench --domain math   -> results/task_bench/<domain>/...
"""
import argparse, collections, itertools, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments_query.oracle_mathmmlu import load_queries
from experiments_query.experiment_adaptive_tts import answer_key as domain_answer_key
from experiments_query.build_bestroute_dynamic import plurality
from experiments_query.budget import MODEL_SPECS, node_cost

D = "experiments_query/results/oracle_mathmmlu"
FULL = {"1b": "meta-llama/llama-3.2-1b-instruct", "8b": "meta-llama/llama-3.1-8b-instruct", "27b": "google/gemma-3-27b-it", "72b": "qwen/qwen-2.5-72b-instruct"}
MODELS = ["1b", "8b", "27b", "72b"]
ACTIONS = ["1b-single", "1b-width", "1b-depth", "8b-single", "8b-width", "8b-depth", "27b-single", "27b-width", "27b-depth", "72b-single", "72b-width"]
BUDGETS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]
LAM = np.concatenate([np.linspace(0, 0.003, 61), np.linspace(0.003, 0.03, 55)[1:], np.linspace(0.03, 0.3, 28)[1:]])


def build_data(domain, out, verifier_path="experiments_query/results/verifier_v2/models/checkpoint-best", cross_fit=None, wtag=""):
    """cross_fit = (list of K fold-verifier paths, K): train-split queries of fold k are scored by the verifier trained on the OTHER folds
    (fold k's verifier was trained on fold k, so it must not score fold k); test queries get the mean score of all fold verifiers."""
    recs, ds = load_queries(domain); gt = {r["id"]: r for r in recs}
    W = {}; Dd = {}; unit = {}
    for l in open(f"{D}/{domain}.jsonl"):
        r = json.loads(l); unit[r["id"]] = r["unit_flops"]
        for m, e in r["per_model"].items(): W[(r["id"], m, "global")] = e["w"]; Dd[(r["id"], m)] = e["d"]
    for l in open(f"{D}/{domain}_tier_w{wtag}.jsonl"):
        r = json.loads(l)
        for m, e in r["per_model"].items(): W[(r["id"], m, "tier")] = e["w"]
    cache = json.load(open(f"{D}/node_cache.json")); samp = collections.defaultdict(dict)
    for k, v in cache.items():
        p = k.split("|")          # qid itself is domain|split|idx -> 3 fields
        if len(p) >= 10 and p[4] == "width" and p[5] == "d0":
            qid = "|".join(p[:3]); m = {v_: k_ for k_, v_ in FULL.items()}.get(p[3])
            if m and qid in gt: samp[(qid, m)][int(p[6][1:])] = v
    def cost_of(qid, v): return node_cost(int(v.get("prompt_tokens") or 0), int(v.get("completion_tokens") or 0), MODEL_SPECS[v["model"]]) / unit[qid]
    def correct(qid, text):
        try: return ds.postprocess_answer(text or "") == gt[qid]["ground_truth"]
        except Exception: return False
    # verifier scores (cached)
    sc_path = f"{out}/verifier_scores.json"
    if os.path.exists(sc_path): SC = json.load(open(sc_path))
    else:
        from experiments_query.bestroute_rm.verifier import Verifier
        SC = {}
        if cross_fit is None:
            ver = Verifier(verifier_path, batch_size=32, max_length=1024)
            for (qid, m), d in samp.items():
                idx = sorted(i for i in d if i < 8); texts = [d[i]["output"] or "" for i in idx]
                if texts:
                    for i, s in zip(idx, ver.score(gt[qid]["prompt"], texts)): SC[f"{qid}|{m}|{i}"] = float(s)
        else:
            paths, K = cross_fit; tr, te = splits(domain, sorted(gt)); perm = np.random.RandomState(0).permutation(len(tr)); folds = np.array_split(perm, K)
            fold_of = {tr[i]: k for k in range(K) for i in folds[k]}; acc = collections.defaultdict(list)
            for k, path in enumerate(paths):
                ver = Verifier(path, batch_size=32, max_length=1024)
                for (qid, m), d in samp.items():
                    if fold_of.get(qid, -1) == k: continue                # never score the fold this verifier was trained on
                    idx = sorted(i for i in d if i < 8); texts = [d[i]["output"] or "" for i in idx]
                    if texts:
                        for i, s in zip(idx, ver.score(gt[qid]["prompt"], texts)): acc[f"{qid}|{m}|{i}"].append(float(s))
                del ver
            SC = {k_: float(np.mean(v)) for k_, v in acc.items()}     # train: one held-out verifier; test: mean of K
        json.dump(SC, open(sc_path, "w"))
    table = {}
    for qid in gt:
        row = {}
        for m in MODELS:
            d = samp.get((qid, m), {})
            if 0 not in d: continue
            s0 = d[0]; row[f"{m}-single"] = (correct(qid, s0["output"]), cost_of(qid, s0), SC.get(f"{qid}|{m}|0"))
            w = W.get((qid, m, "tier" if m in ("1b", "8b") else "global"))
            if w:
                k = int(w["nodes"]); cw = sum(cost_of(qid, d[i]) for i in range(k) if i in d)     # generation cost only (verifier free)
                keys = [domain_answer_key(domain, ds, d[i]["output"] or "") for i in range(k) if i in d]
                top, _ = plurality(keys); mv = correct(qid, d[[i for i in range(k) if i in d][keys.index(top)]]["output"]) if top is not None else correct(qid, d[0]["output"])
                ids = [i for i in range(k) if i in d and f"{qid}|{m}|{i}" in SC]
                wv = correct(qid, d[max(ids, key=lambda i: SC[f"{qid}|{m}|{i}"])]["output"]) if ids else bool(w["solved"])   # read-out = argmax of the CURRENT verifier over the drawn samples
                row[f"{m}-width"] = (wv, cw, None); row[f"{m}-width-mv"] = (mv, cw, None)
            dd = Dd.get((qid, m))
            if dd: row[f"{m}-depth"] = (bool(dd["solved"]), float(dd["cost"]), None)
            # best-of-n (verifier argmax) and self-consistency over the first n samples
            for n in (2, 3, 5):
                ids = [i for i in range(n) if i in d and f"{qid}|{m}|{i}" in SC]
                if len(ids) == n:
                    j = max(ids, key=lambda i: SC[f"{qid}|{m}|{i}"]); row[f"{m}-bon{n}"] = (correct(qid, d[j]["output"]), sum(cost_of(qid, d[i]) for i in ids), None)
                    keys = [domain_answer_key(domain, ds, d[i]["output"] or "") for i in ids]; top, _ = plurality(keys)
                    row[f"{m}-sc{n}"] = (correct(qid, d[ids[keys.index(top)]]["output"]) if top is not None else row[f"{m}-single"][0], sum(cost_of(qid, d[i]) for i in ids), None)
        table[qid] = row
    texts = {qid: {m: (samp[(qid, m)][0]["output"] or "") for m in ("1b", "8b") if 0 in samp.get((qid, m), {})} for qid in gt}
    return gt, table, texts


def splits(domain, qids, seed=0):
    if domain == "math": tr = [q for q in qids if q.split("|")[1] == "train"]; te = [q for q in qids if q.split("|")[1] == "test"]
    else:
        dev = [q for q in qids if q.split("|")[1] == "train"]; test = sorted(q for q in qids if q.split("|")[1] == "test"); rng = random.Random(seed); rng.shuffle(test)
        tr = dev + test[:len(test) - 500]; te = test[len(test) - 500:]
    return tr, te


def frontier_eval(rows, name, ak="val_acc", ck="val_cost"):
    def sel(b): ok = [r for r in rows if r[ck] <= b]; return max(ok, key=lambda r: (r[ak], -r[ck])) if ok else None
    return {b: (sel(b)["test_acc"], sel(b)["test_cost"]) if sel(b) else None for b in BUDGETS}


def apgr(rows, AW, CW, AS, CS, ak="val_acc", ck="val_cost", K=100):
    fr = []; best = -1
    for r in sorted(rows, key=lambda r: r[ck]):
        if r[ak] > best: best = r[ak]; fr.append(r)
    pg = []
    for t in np.linspace(0, 1, K):
        b = CW + t * (CS - CW); ok = [r for r in fr if r["test_cost"] <= b]; acc = max(ok, key=lambda r: r[ak])["test_acc"] if ok else AW   # feasibility on the REALISED test cost; pg.append((acc - AW) / (AS - AW))
    return float(np.mean(pg))


def oof_predict(Xtr, Ytr, Mtr, Xte, seed=0, K=5):
    """K-fold out-of-fold probabilities for every train query (selection side) + test probabilities from a
    model fitted on all train data (each fold and the final model use a 10% inner early-stopping slice)."""
    n = len(Xtr); rng = np.random.RandomState(seed); perm = rng.permutation(n); folds = np.array_split(perm, K); P_oof = np.zeros_like(Ytr)
    for k in range(K):
        te_idx = folds[k]; tr_idx = np.concatenate([folds[j] for j in range(K) if j != k]); nv = max(len(tr_idx) // 10, 1)
        f = mlp_fit(Xtr[tr_idx[nv:]], Ytr[tr_idx[nv:]], Mtr[tr_idx[nv:]], Xtr[tr_idx[:nv]], Ytr[tr_idx[:nv]], Mtr[tr_idx[:nv]], seed + k)
        P_oof[te_idx] = f(Xtr[te_idx])
    nv = max(n // 10, 1); f = mlp_fit(Xtr[perm[nv:]], Ytr[perm[nv:]], Mtr[perm[nv:]], Xtr[perm[:nv]], Ytr[perm[:nv]], Mtr[perm[:nv]], seed)
    return P_oof, f(Xte)


def mlp_fit(Xtr, Ytr, Mtr, Xva, Yva, Mva, seed=0, hidden=256, epochs=120):
    import torch, torch.nn as nn
    torch.manual_seed(seed); K = Ytr.shape[1]
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, K))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss(reduction="none")
    lf = lambda o, y, m: (bce(o, y) * m).sum() / m.sum().clamp(min=1)
    Xt, Yt, Mt = torch.tensor(Xtr), torch.tensor(Ytr), torch.tensor(Mtr); Xv, Yv, Mv = torch.tensor(Xva), torch.tensor(Yva), torch.tensor(Mva); best = (-1e9, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lf(net(Xt[b]), Yt[b], Mt[b]).backward(); opt.step()
        if ep % 5 == 4:
            net.eval()
            with torch.no_grad(): s = -float(lf(net(Xv), Yv, Mv))
            if s > best[0]: best = (s, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return lambda X: torch.sigmoid(net(torch.tensor(X))).detach().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verifier-path", default="experiments_query/results/verifier_v2/models/checkpoint-best"); ap.add_argument("--tag", default="", help="output subdir suffix, e.g. _indomain"); ap.add_argument("--cross-fit", default=None, help="comma-separated K fold-verifier checkpoints (fold order)"); a = ap.parse_args()
    out = f"experiments_query/results/task_bench{a.tag}/{a.domain}"; os.makedirs(out, exist_ok=True)
    gt, table, texts = build_data(a.domain, out, a.verifier_path, (a.cross_fit.split(","), len(a.cross_fit.split(","))) if a.cross_fit else None)
    qids = [q for q in gt if all(f"{m}-single" in table[q] for m in MODELS) and "1b-width" in table[q] and "8b-width" in table[q]]
    tr, te = splits(a.domain, qids, a.seed); rng = np.random.RandomState(a.seed); rng.shuffle(tr); va, fit = tr, tr   # selection = out-of-fold on the whole train split
    print(f"{a.domain}: usable {len(qids)} (train {len(tr)}, selection by 5-fold out-of-fold predictions; test {len(te)})")
    def YC(qs, acts):
        Y = np.zeros((len(qs), len(acts)), np.float32); C = np.zeros_like(Y); M = np.zeros_like(Y)
        for j, q in enumerate(qs):
            for i, act in enumerate(acts):
                if act in table[q]: Y[j, i], C[j, i], M[j, i] = float(table[q][act][0]), table[q][act][1], 1.0
        return Y, C, M
    # ---- features (lite-p): MiniLM(query), MiniLM(1b probe), MiniLM(8b probe), standardized probe verifier scores ----
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = lambda ts: np.asarray(enc.encode(ts, normalize_embeddings=False, show_progress_bar=False, batch_size=128), dtype=np.float32)
    def feats(qs, mu=None, sd=None):
        F = np.array([[table[q]["1b-single"][2] or 0.0, table[q]["8b-single"][2] or 0.0] for q in qs], np.float32)
        if mu is None: mu, sd = F.mean(0), F.std(0) + 1e-6
        Xq = E([gt[q]["prompt"] for q in qs]); return Xq, np.concatenate([Xq, E([texts[q].get("1b", "") for q in qs]), E([texts[q].get("8b", "") for q in qs]), (F - mu) / sd], 1), mu, sd
    Xq_fit, X_fit, mu, sd = feats(fit); Xq_va, X_va, _, _ = feats(va, mu, sd); Xq_te, X_te, _, _ = feats(te, mu, sd)
    results = {}; mixes = {}
    def lagr(P_va, P_te, Yv, Cv, Yt, Ct, Mv, Mt, acts, cbar):
        rows = []
        for lam in LAM:
            pv = np.argmax(np.where(Mv > 0, P_va - lam * cbar, -1e9), 1); pt = np.argmax(np.where(Mt > 0, P_te - lam * cbar, -1e9), 1)
            rows.append({"lambda": float(lam), "val_acc": float(Yv[np.arange(len(pv)), pv].mean()), "val_cost": float(Cv[np.arange(len(pv)), pv].mean()),
                         "test_acc": float(Yt[np.arange(len(pt)), pt].mean()), "test_cost": float(Ct[np.arange(len(pt)), pt].mean()), "mix": dict(collections.Counter(acts[x] for x in pt))})
        return rows
    def ladder(P_va, P_te, Yv, Cv, Yt, Ct, Mv, Mt, acts, taus):
        rows = []
        for t in taus:
            def dec(P, M):
                pick = np.full(len(P), len(acts) - 1)
                for i in range(len(acts) - 2, -1, -1): pick = np.where((P[:, i] >= t) & (M[:, i] > 0), i, pick)
                return pick
            pv, pt = dec(P_va, Mv), dec(P_te, Mt)
            rows.append({"tau": float(t), "val_acc": float(Yv[np.arange(len(pv)), pv].mean()), "val_cost": float(Cv[np.arange(len(pv)), pv].mean()), "test_acc": float(Yt[np.arange(len(pt)), pt].mean()), "test_cost": float(Ct[np.arange(len(pt)), pt].mean()), "mix": dict(collections.Counter(acts[x] for x in pt))})
        return rows
    # ---- ours: 11 modes (+ variant with majority-vote read-out for the width actions) ----
    for tag, acts in (("ours", ACTIONS), ("ours-mv", [a_.replace("-width", "-width-mv") if a_.endswith("-width") else a_ for a_ in ACTIONS])):
        Yf, Cf, Mf = YC(fit, acts); Yv, Cv, Mv = YC(va, acts); Yt, Ct, Mt = YC(te, acts)
        P_va, P_te = oof_predict(X_fit, Yf, Mf, X_te, a.seed)
        cbar = np.array([Cf[Mf[:, i] > 0, i].mean() if (Mf[:, i] > 0).any() else 1e6 for i in range(len(acts))])
        from sklearn.metrics import roc_auc_score
        aucs = {acts[i]: round(float(roc_auc_score(Yt[Mt[:, i] > 0, i], P_te[Mt[:, i] > 0, i])), 3) if 0 < Yt[Mt[:, i] > 0, i].sum() < (Mt[:, i] > 0).sum() else None for i in range(len(acts))}
        rows = lagr(P_va, P_te, Yv, Cv, Yt, Ct, Mv, Mt, acts, cbar); results[tag] = rows; mixes[tag] = aucs
        print(f"{tag}: test AUC {aucs}")
    # ---- query-only binary router (RouteLLM / Hybrid-LLM style): P(8b single correct) from query text; else 72b ----
    acts2 = ["8b-single", "72b-single"]
    Yf, Cf, Mf = YC(fit, acts2); Yv, Cv, Mv = YC(va, acts2); Yt, Ct, Mt = YC(te, acts2)
    Po, Pt = oof_predict(Xq_fit, Yf[:, :1], Mf[:, :1], Xq_te, a.seed)
    P_va = np.column_stack([Po[:, 0], np.ones(len(va))]); P_te = np.column_stack([Pt[:, 0], np.ones(len(te))])
    results["query-only router 8b/72b (RouteLLM-style)"] = ladder(P_va, P_te, Yv, Cv, Yt, Ct, Mv, Mt, acts2, np.linspace(0, 1, 101))
    # ---- BEST-Route-style: query-only router over (model, best-of-n) actions, cost-ordered threshold decode ----
    acts3 = ["1b-single", "1b-bon2", "1b-bon3", "1b-bon5", "8b-single", "8b-bon2", "8b-bon3", "8b-bon5", "27b-single", "72b-single"]
    Yf, Cf, Mf = YC(fit, acts3); Yv, Cv, Mv = YC(va, acts3); Yt, Ct, Mt = YC(te, acts3)
    order = list(np.argsort([Cf[Mf[:, i] > 0, i].mean() for i in range(len(acts3))])); acts3o = [acts3[i] for i in order]
    Yf, Cf, Mf, Yv, Cv, Mv, Yt, Ct, Mt = [x[:, order] for x in (Yf, Cf, Mf, Yv, Cv, Mv, Yt, Ct, Mt)]
    Po, Pt = oof_predict(Xq_fit, Yf, Mf, Xq_te, a.seed)
    results["BEST-Route-style (query-only, model x best-of-n)"] = ladder(Po, Pt, Yv, Cv, Yt, Ct, Mv, Mt, acts3o, np.linspace(0, 1, 101))
    # ---- verifier-gated cascades: thresholds on the single-call verifier score, selected on train (fit+val) ----
    def cascade(seq, qs, ths):
        acc = []; cost = []
        for q in qs:
            c = 0.0; ok = False
            for s, th in zip(seq, list(ths) + [None]):
                y, cc, v = table[q][f"{s}-single"]; c += cc
                if th is None or (v is not None and v >= th): ok = y; break
            acc.append(ok); cost.append(c)
        return float(np.mean(acc)), float(np.mean(cost))
    def cascade_rows(seqs, qgrid):
        rows = []
        for seq in seqs:
            grids = [np.quantile([table[q][f"{s}-single"][2] for q in tr if table[q][f"{s}-single"][2] is not None], qgrid) for s in seq[:-1]]
            for ths in itertools.product(*grids):
                va_, vc = cascade(seq, tr, ths); ta, tc = cascade(seq, te, ths)
                rows.append({"seq": seq, "ths": [float(t) for t in ths], "val_acc": va_, "val_cost": vc, "test_acc": ta, "test_cost": tc})
        return rows
    qgrid = [0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    results["AutoMix-style cascade 8b->27b->72b"] = cascade_rows([("8b", "27b", "72b")], qgrid)
    seqs = [s for L in (1, 2, 3) for s in itertools.permutations(MODELS, L) if list(s) == sorted(s, key=MODELS.index)]
    results["FrugalGPT-style cascade (learned sequence)"] = cascade_rows(seqs, qgrid)
    # ---- fixed TTS + vanilla (single points; selection-side = train) ----
    def point(act, qs):
        v = [table[q][act] for q in qs if act in table[q]]; return (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) if v else None
    fixed = {}
    for act in ["1b-single", "8b-single", "27b-single", "72b-single", "1b-bon2", "1b-bon3", "1b-bon5", "8b-bon2", "8b-bon3", "8b-bon5", "1b-sc3", "1b-sc5", "8b-sc3", "8b-sc5", "1b-depth", "8b-depth", "1b-width", "8b-width", "1b-width-mv", "8b-width-mv"]:
        ptr, pte = point(act, tr), point(act, te)
        if ptr and pte: fixed[act] = {"val_acc": ptr[0], "val_cost": ptr[1], "test_acc": pte[0], "test_cost": pte[1]}
    for fam, acts_ in (("BoN-1b", ["1b-single", "1b-bon2", "1b-bon3", "1b-bon5"]), ("BoN-8b", ["8b-single", "8b-bon2", "8b-bon3", "8b-bon5"]), ("SC-1b", ["1b-single", "1b-sc3", "1b-sc5"]), ("SC-8b", ["8b-single", "8b-sc3", "8b-sc5"]),
                       ("Self-Refine", ["1b-depth", "8b-depth"]), ("tts-planner-w (fixed model)", ["1b-width", "8b-width"]), ("Vanilla", ["1b-single", "8b-single", "27b-single", "72b-single"])):
        results[fam] = [fixed[x] for x in acts_ if x in fixed]
    # ---- report ----
    AW1, CW1 = fixed["1b-single"]["test_acc"], fixed["1b-single"]["test_cost"]; AW8, CW8 = fixed["8b-single"]["test_acc"], fixed["8b-single"]["test_cost"]; AS, CS = fixed["72b-single"]["test_acc"], fixed["72b-single"]["test_cost"]
    print(f"\nanchors: 1b {AW1:.3f}@{CW1:.1f}  8b {AW8:.3f}@{CW8:.1f}  72b {AS:.3f}@{CS:.1f}")
    print("fixed points (test):", {k: (round(v['test_acc'], 3), round(v['test_cost'], 1)) for k, v in fixed.items()})
    summary = {"splits": {"train": len(tr), "selection": "5-fold out-of-fold", "test": len(te)}, "anchors": {"1b": (AW1, CW1), "8b": (AW8, CW8), "72b": (AS, CS)}, "fixed": fixed, "aucs": mixes, "table": {}}
    print(f"\n{'method':48s}" + "".join(f"{b:>11d}" for b in BUDGETS) + "   APGR 1b / 8b")
    for name, rows in results.items():
        fe = frontier_eval(rows, name); a1 = apgr(rows, AW1, CW1, AS, CS); a8 = apgr(rows, AW8, CW8, AS, CS)
        summary["table"][name] = {"per_budget": {str(b): v for b, v in fe.items()}, "apgr": (a1, a8)}
        print(f"{name:48s}" + "".join((f"{v[0]:.3f}@{v[1]:4.0f}" if v else "    --    ").rjust(11) for v in fe.values()) + f"   {a1:.3f} / {a8:.3f}")
    for tag in ("ours", "ours-mv"):
        def sel(b): ok = [r for r in results[tag] if r["val_cost"] <= b]; return max(ok, key=lambda r: (r["val_acc"], -r["val_cost"])) if ok else None
        print(f"{tag} mixes:", {b: {k: v for k, v in sel(b)["mix"].items() if v >= 0.03 * len(te)} for b in (10, 25, 50) if sel(b)})
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1); json.dump({k: v for k, v in results.items()}, open(f"{out}/rows.json", "w"))


if __name__ == "__main__":
    main()
