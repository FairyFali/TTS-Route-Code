"""TTS-Router with three MODES per model: single call / width (tts-planner-w) / depth (tts-planner-d).

Actions (72b-depth does not exist: the chain cannot revise under B, so 11 heads):
  1b-single, 1b-width, 1b-depth, 8b-single, 8b-width, 8b-depth,
  27b-single, 27b-width, 27b-depth, 72b-single, 72b-width
single = one call (the probe for 1b/8b; the first cached sample for 27b/72b), width = the planner search
of the profiling library (tiered for 1b/8b, global B for 27b/72b), depth = tts-planner-d (global B).
Same lite-p features (query + 1b/8b probe texts + probe scores) -> MLP-256 -> 11 sigmoid heads (BCE).
Decoders (both val-selected per budget, then frozen on test):
  (i) cheapest-first ladder over the actions in mean-cost order with per-action thresholds, found by
      random search (per-gate quantile grids incl. +-inf) followed by greedy coordinate refinement;
  (ii) Lagrangian: a*(q) = argmax_a p_a(q) - lambda * cbar_a  (cbar_a = train mean cost of a), lambda swept.
Output: results/tts_router_modes/{router_model.pt, meta.json, ladder.json, lagrange.json, comparison.json}
"""
import argparse, collections, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments_query.train_router_probe_chs_lite import build_features, NPZ
from experiments_query.train_router_litep6 import build_six
from experiments_query.pretest_labels import single_train, single_from_cache

DEPTH = {"train": "experiments_query/results/depth_planner_train/depth_planner_global_w2.jsonl",
         "test": "experiments_query/results/depth_planner_test/depth_planner_global_w2.jsonl"}
ACTIONS = ["1b-single", "1b-width", "1b-depth", "8b-single", "8b-width", "8b-depth",
           "27b-single", "27b-width", "27b-depth", "72b-single", "72b-width"]


def build(side):
    qids, prompts, Y6, C6 = build_six(side)      # 1b-probe, 1b-search, 8b-probe, 8b-search, 27b-search, 72b-search
    S = single_train(qids) if side == "train" else single_from_cache("experiments_query/results/oracle_v2_test/node_cache.json", "experiments_query/results/oracle_v2_test/judge_cache.json", qids)
    dp = {}
    for l in open(DEPTH[side], encoding="utf-8"):
        if l.strip():
            r = json.loads(l); dp[r["id"]] = r["per_model"]
    Y = np.zeros((len(qids), 11), bool); C = np.zeros((len(qids), 11), np.float32); M = np.ones((len(qids), 11), bool)
    for j, q in enumerate(qids):
        d = dp[q]
        s27 = S.get((q, "27b")); s72 = S.get((q, "72b"))
        vals = [(Y6[j, 0], C6[j, 0]), (Y6[j, 1], C6[j, 1]), (d["1b"]["solved"], d["1b"]["cost"]),
                (Y6[j, 2], C6[j, 2]), (Y6[j, 3], C6[j, 3]), (d["8b"]["solved"], d["8b"]["cost"]),
                (s27[0], s27[1]) if s27 else (False, C6[j, 4]), (Y6[j, 4], C6[j, 4]), (d["27b"]["solved"], d["27b"]["cost"]) if "27b" in d else (Y6[j, 4], C6[j, 4]),
                (s72[0], s72[1]) if s72 else (Y6[j, 5], C6[j, 5]), (Y6[j, 5], C6[j, 5])]
        for i, (y, c) in enumerate(vals): Y[j, i] = bool(y); C[j, i] = float(c)
        if not s27: M[j, 6] = False
        if not s72: M[j, 9] = False
    return qids, prompts, Y, C, M


def decode_ladder(P, order, taus):
    pick = np.full(len(P), order[-1])
    for i, t in zip(order[-2::-1], taus[::-1]):
        pick = np.where(P[:, i] >= t, i, pick)
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/tts_router_modes")
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--random-configs", type=int, default=300000)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    qids, prompts, Y, C, M = build("train"); n = len(qids)
    idx = np.arange(n); rng = np.random.RandomState(a.seed); rng.shuffle(idx); nv = int(n * a.val_frac); va, tr = idx[:nv], idx[nv:]
    print(f"train-side queries: {n} (fit {len(tr)} / val {len(va)})")
    print("success:", {ACTIONS[i]: round(float(Y[M[:, i], i].mean()), 3) for i in range(11)})
    print("mean cost:", {ACTIONS[i]: round(float(C[M[:, i], i].mean()), 1) for i in range(11)})
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    z = np.load(NPZ["train"]); Fraw = np.stack([z["1b"], z["8b"]], axis=1).astype(np.float32); mu, sd = Fraw[tr].mean(0), Fraw[tr].std(0) + 1e-6
    X, _ = build_features("train", qids, prompts, enc, mu, sd); Yf = Y.astype(np.float32); Mf = M.astype(np.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.trunk = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(a.hidden, a.hidden), nn.ReLU()); s.head = nn.Linear(a.hidden, 11)
        def forward(s, x): return s.head(s.trunk(x))
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss(reduction="none")
    Xt, Yt, Mt = torch.tensor(X[tr]), torch.tensor(Yf[tr]), torch.tensor(Mf[tr]); Xv, Yv, Mv = torch.tensor(X[va]), torch.tensor(Yf[va]), torch.tensor(Mf[va])
    loss_fn = lambda out, y, m: (bce(out, y) * m).sum() / m.sum()
    best = (-1e9, None, -1)
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); loss_fn(net(Xt[b]), Yt[b], Mt[b]).backward(); opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad(): sc = -float(loss_fn(net(Xv), Yv, Mv))
            if sc > best[0]: best = (sc, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): Ptr = torch.sigmoid(net(torch.tensor(X))).numpy()
    qte, pte, Yte, Cte, Mte = build("test"); Xte, _ = build_features("test", qte, pte, enc, mu, sd)
    with torch.no_grad(): Pte = torch.sigmoid(net(torch.tensor(Xte))).numpy()
    auc = lambda Yy, Pp, Mm: {ACTIONS[i]: round(float(roc_auc_score(Yy[Mm[:, i], i].astype(int), Pp[Mm[:, i], i])), 3) if 0 < Yy[Mm[:, i], i].sum() < Mm[:, i].sum() else None for i in range(11)}
    aucs_v, aucs_t = auc(Y[va], Ptr[va], M[va]), auc(Yte, Pte, Mte)
    print(f"best epoch {best[2]}\nval AUC : {aucs_v}\ntest AUC: {aucs_t}")
    torch.save({"state": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": a.hidden, "actions": ACTIONS, "encoder": "sentence-transformers/all-MiniLM-L6-v2", "feat_mu": mu.tolist(), "feat_sd": sd.tolist(), "arch": "modes_mlp"}, os.path.join(a.outdir, "router_model.pt"))
    json.dump({"actions": ACTIONS, "n_fit": int(len(tr)), "n_val": int(len(va)), "best_epoch": best[2], "val_auc": aucs_v, "test_auc": aucs_t, "val_ids": [qids[i] for i in va]}, open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)

    # ---------- decoders ----------
    cbar = np.array([C[M[:, i], i].mean() for i in range(11)]); order = list(np.argsort(cbar))     # cost order
    print("cost order:", [ACTIONS[i] for i in order])
    Pv, Yv_, Cv = Ptr[va], Y[va], C[va]
    def score(pick, Yy, Cc): return float(np.mean(Yy[np.arange(len(pick)), pick])), float(np.mean(Cc[np.arange(len(pick)), pick]))
    def row(pick_v, pick_t, extra=None):
        va_acc, va_cost = score(pick_v, Yv_, Cv); te_acc, te_cost = score(pick_t, Yte, Cte)
        r = {"val_acc": va_acc, "val_cost": va_cost, "test_acc": te_acc, "test_cost": te_cost, "mix": dict(collections.Counter(ACTIONS[x] for x in pick_t))}
        if extra: r.update(extra)
        return r
    # (i) ladder: random search + greedy refinement on val, evaluated per budget
    gates = order[:-1]; grids = {i: [-np.inf] + [float(x) for x in np.quantile(Ptr[:, i], [0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.97])] + [np.inf] for i in gates}
    rs = np.random.RandomState(1); rows = []
    for _ in range(a.random_configs):
        taus = [grids[i][rs.randint(9)] for i in gates]
        rows.append(row(decode_ladder(Pv, order, taus), decode_ladder(Pte, order, taus), {"taus": [None if not np.isfinite(t) else round(t, 4) for t in taus]}))
    budgets = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]
    def sel(rr, b): ok = [r for r in rr if r["val_cost"] <= b]; return max(ok, key=lambda r: (r["val_acc"], -r["val_cost"])) if ok else None
    refined = []
    for b in budgets:      # greedy coordinate refinement of the best random config under each budget
        r = sel(rows, b)
        if r is None: continue
        taus = [(-np.inf if t is None else t) for t in r["taus"]]
        # None encodes both infs; recover +inf when the gate is never taken in val
        taus = [t if (t != -np.inf or (Pv[:, g] >= -np.inf).any()) else t for g, t in zip(gates, taus)]
        improved = True; cur = r
        while improved:
            improved = False
            for k, g in enumerate(gates):
                for t in grids[g]:
                    cand = list(taus); cand[k] = t
                    rr = row(decode_ladder(Pv, order, cand), decode_ladder(Pte, order, cand))
                    if rr["val_cost"] <= b and (rr["val_acc"], -rr["val_cost"]) > (cur["val_acc"], -cur["val_cost"]):
                        cur, taus, improved = rr, cand, True
        cur["budget"] = b; refined.append(cur)
    json.dump(rows[:50000] + refined, open(os.path.join(a.outdir, "ladder.json"), "w"))
    # (ii) Lagrangian decode
    lag = []
    for lam in np.concatenate([np.linspace(0, 0.02, 41), np.linspace(0.02, 0.2, 37)[1:]]):
        pv = np.argmax(Pv - lam * cbar, 1); pt = np.argmax(Pte - lam * cbar, 1); lag.append(row(pv, pt, {"lambda": float(lam)}))
    json.dump(lag, open(os.path.join(a.outdir, "lagrange.json"), "w"), indent=1)
    # ---------- comparison ----------
    p6 = json.load(open("experiments_query/results/tts_router_litep6/pipeline.json"))
    old = [{"val_acc": 0.762, "val_cost": 9.83, "test_acc": 0.773, "test_cost": 9.49}] + json.load(open("experiments_query/results/tts_router_probe_chs_lite3/permodel_pipelines.json"))["lite"]
    final6 = [r for r in p6 if r["val_cost"] < 30] + old
    f = lambda r: f"{r['test_acc']:.3f}@{r['test_cost']:5.1f}" if r else "   --  "
    comp = {}
    print(f"\n{'b':>4} | {'modes ladder':>14} | {'modes Lagrangian':>16} | {'lite-p FINAL':>14} | ladder mix (modes used)")
    for b in budgets:
        rl = next((r for r in refined if r.get("budget") == b), None); rg = sel(lag, b); rf = sel(final6, b)
        mix = {k: v for k, v in sorted(rl["mix"].items(), key=lambda kv: -kv[1]) if v >= 12} if rl else {}
        print(f"{b:>4} | {f(rl):>14} | {f(rg):>16} | {f(rf):>14} | {mix}")
        comp[b] = {"ladder": rl, "lagrange": rg, "litep_final": rf}
    json.dump(comp, open(os.path.join(a.outdir, "comparison.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
