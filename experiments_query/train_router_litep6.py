"""lite-p-6: the accept-aware lite-p router.

Six correctness heads over the lite-p feature space (query + 1b/8b probe texts +
probe scores, MiniLM frozen + MLP):
  a0: P(1b PROBE answer correct)      -- action: return it (cost = 1b call)
  a1: P(1b SEARCH succeeds)           -- tiered planner
  a2: P(8b PROBE answer correct)      -- action: return it (cost = 8b call)
  a3: P(8b SEARCH succeeds)
  a4: P(27b call correct)
  a5: P(72b call correct)
Labels all cached (probe verdicts via pan.side, search outcomes VC-free). Decode:
cheapest-first with per-action taus. Accept-charged protocol: every action's returned
generation is charged; unused probes stay free routing evidence.
"""
import argparse, json, os, random, sys
import numpy as np

from experiments_query.train_router_searchcost import load_side_actions, MODELS
from experiments_query.train_router_probe_chs_lite import build_features, NPZ

ACTIONS = ["1b-probe", "1b-search", "8b-probe", "8b-search", "27b", "72b"]


def load_probe_outcomes(side):
    import experiments_query.pipeline_planner_as_needed as pan
    from experiments_query.build_bestroute_tts import _h, ORACLE_MODEL, is_refusal
    pan.CHAIN = ["1b", "8b", "27b", "72b"]
    gt = {r["id"]: r for r in (json.loads(l) for l in open(pan.GT, encoding="utf-8") if l.strip())}
    if side == "train":
        jv = {}
        for l in open("experiments_query/results/verifier_v2/judged_samples.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); jv[r["key"]] = bool(r["correct"])
        pv = lambda k, q, v: jv[k]
        cfg = ("experiments_query/results/oracle_verifier_full/node_cache.json",
               "experiments_query/results/oracle_v2_full/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
               "experiments_query/results/pipeline_probe_router/train_probe_scores.npz")
    else:
        jc = json.load(open("experiments_query/results/oracle_v2_test/judge_cache.json", encoding="utf-8"))

        def pv(k, q, v):
            key = _h(ORACLE_MODEL, gt[q]["prompt"][:80], (v["output"] or "")[:200])
            if key in jc:
                return bool(jc[key])
            return is_refusal(v["output"] or "") if str(gt[q].get("final_answer") or "").lower() == "refusal" else False
        cfg = ("experiments_query/results/oracle_v2_test/node_cache.json",
               "experiments_query/results/oracle_v2_test/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
               "experiments_query/results/pipeline_probe_router/test_probe_scores.npz")
    qids, S, PC, PY, sY, sC = pan.side(side, cfg[0], cfg[1], cfg[2], gt, pv)
    return {q: (bool(PY["1b"][j]), float(PC["1b"][j]), bool(PY["8b"][j]), float(PC["8b"][j]))
            for j, q in enumerate(qids)}


def build_six(side):
    qids, prompts, Y, C = load_side_actions(side)   # search outcomes (VC-free)
    po = load_probe_outcomes(side)
    Y6 = np.zeros((len(qids), 6), bool); C6 = np.zeros((len(qids), 6), np.float32)
    for j, q in enumerate(qids):
        p1y, p1c, p8y, p8c = po[q]
        Y6[j] = [p1y, Y[j, 0] >= 0.5, p8y, Y[j, 1] >= 0.5, Y[j, 2] >= 0.5, Y[j, 3] >= 0.5]
        C6[j] = [p1c, C[j, 0], p8c, C[j, 1], C[j, 2], C[j, 3]]
    return qids, prompts, Y6, C6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/tts_router_litep6")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.outdir, "router_model.pt")):
        raise SystemExit(f"{a.outdir} already holds a router; pick a new --outdir")
    os.makedirs(a.outdir, exist_ok=True)

    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    qids, prompts, Y6, C6 = build_six("train")
    n = len(qids)
    idx = np.arange(n); rng = np.random.RandomState(a.seed); rng.shuffle(idx)
    nv = int(n * a.val_frac); va, tr = idx[:nv], idx[nv:]
    print(f"train-side queries: {n} (fit {len(tr)} / val {len(va)})")
    print("action success rates:", {ACTIONS[i]: round(float(Y6[:, i].mean()), 3) for i in range(6)})
    print("action mean costs   :", {ACTIONS[i]: round(float(C6[:, i].mean()), 2) for i in range(6)})

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    z = np.load(NPZ["train"]); Fraw = np.stack([z["1b"], z["8b"]], axis=1).astype(np.float32)
    mu, sd = Fraw[tr].mean(0), Fraw[tr].std(0) + 1e-6
    X, _ = build_features("train", qids, prompts, enc, mu, sd)
    Yf = Y6.astype(np.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.trunk = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(a.hidden, a.hidden), nn.ReLU())
            s.head = nn.Linear(a.hidden, 6)

        def forward(s, x):
            return s.head(s.trunk(x))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(X[tr]), torch.tensor(Yf[tr])
    Xv, Yv = torch.tensor(X[va]), torch.tensor(Yf[va])
    best = (-1e9, None, -1)
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            bce(net(Xt[b]), Yt[b]).backward(); opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                score = -float(bce(net(Xv), Yv))
            if score > best[0]:
                best = (score, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        prob = torch.sigmoid(net(Xv)).numpy()
    aucs = {ACTIONS[i]: round(float(roc_auc_score(Y6[va][:, i].astype(int), prob[:, i])), 3)
            for i in range(6) if 0 < Y6[va][:, i].sum() < len(va)}
    print(f"best epoch {best[2]}   val AUCs: {aucs}")
    torch.save({"state": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": a.hidden,
                "actions": ACTIONS, "encoder": "sentence-transformers/all-MiniLM-L6-v2",
                "feat_mu": mu.tolist(), "feat_sd": sd.tolist(), "arch": "litep6_mlp"},
               os.path.join(a.outdir, "router_model.pt"))
    json.dump({"actions": ACTIONS, "n_fit": int(len(tr)), "n_val": int(len(va)),
               "best_epoch": best[2], "val_auc": aucs,
               "val_ids": [qids[i] for i in va]},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"saved -> {a.outdir}/router_model.pt")


if __name__ == "__main__":
    main()
