"""TTS-Router on least-realized-search-cost labels.

Label per training query: the model with the LEAST realized search cost among those
whose TTS-planner/search outcome is correct (per-query argmin, VC-free costs; queries
no model solves are dropped from the CE loss but kept for the cost head).
Input: query text (MiniLM embedding). Output: 4-way model id + per-model estimated
search cost (log1p regression). Class-balanced CE (effective-number weights, the v1/v2
recipe) because the label mass is 61/23/9/5.

New outdir; nothing overwritten.  Run from the repo root.
"""
import argparse, json, os, random
import numpy as np

MODELS = ["1b", "8b", "27b", "72b"]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def load_side_actions(side):
    """Per-(query, model) VC-free search outcomes, via the ttsonly loaders."""
    import experiments_query.pipeline_planner_as_needed as pan
    from experiments_query.build_bestroute_tts import _h, ORACLE_MODEL, is_refusal
    pan.CHAIN = ["1b", "8b", "27b", "72b"]
    VC = 0.48
    gt = {r["id"]: r for r in (json.loads(l) for l in open(pan.GT, encoding="utf-8") if l.strip())}

    def load_runs(path):
        out = {}
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if "error" in r:
                continue
            for m in r["per_model"]:
                run = m["runs"][0]
                c = float(run["cost"]) - VC * int(run.get("n_verifier_scored") or 0)
                out[(r["id"], m["short"])] = (bool(run["solved"]), max(c, 0.0))
        return out

    cfg = {"train": ("experiments_query/results/oracle_verifier_full/node_cache.json",
                     "experiments_query/results/oracle_v2_full/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
                     "experiments_query/results/oracle_v2_full_b1t/bestroute_dynamic_both_or_ro-verifier_p3_1b-8b_runs0-0.jsonl",
                     "experiments_query/results/pipeline_probe_router/train_probe_scores.npz"),
           "test": ("experiments_query/results/oracle_v2_test/node_cache.json",
                    "experiments_query/results/oracle_v2_test/bestroute_dynamic_both_or_ro-verifier_p1.jsonl",
                    "experiments_query/results/oracle_v2_test_b1t/bestroute_dynamic_both_or_ro-verifier_p3_1b-8b_runs0-0.jsonl",
                    "experiments_query/results/pipeline_probe_router/test_probe_scores.npz")}[side]
    jv = {}
    if side == "train":
        for l in open("experiments_query/results/verifier_v2/judged_samples.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); jv[r["key"]] = bool(r["correct"])
        pv = lambda k, q, v: jv[k]
    else:
        jc = json.load(open("experiments_query/results/oracle_v2_test/judge_cache.json", encoding="utf-8"))

        def pv(k, q, v):
            key = _h(ORACLE_MODEL, gt[q]["prompt"][:80], (v["output"] or "")[:200])
            if key in jc:
                return bool(jc[key])
            return is_refusal(v["output"] or "") if str(gt[q].get("final_answer") or "").lower() == "refusal" else False
    qids, S, PC, PY, sY, sC = pan.side(side, cfg[0], cfg[1], cfg[3], gt, pv)
    big = load_runs(cfg[1]); t = load_runs(cfg[2])
    Y = np.zeros((len(qids), 4), bool); C = np.zeros((len(qids), 4), np.float32)
    for j, q in enumerate(qids):
        for i, s in enumerate(MODELS):
            src = t if s in ("1b", "8b") else big
            Y[j, i], C[j, i] = src[(q, s)]
    prompts = [gt[q]["prompt"] for q in qids]
    return qids, prompts, Y, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/tts_router_searchcost")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cost-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.outdir, "router_model.pt")):
        raise SystemExit(f"{a.outdir} already holds a router; pick a new --outdir (no overwriting)")
    os.makedirs(a.outdir, exist_ok=True)

    import torch, torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import accuracy_score, f1_score
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    qids, prompts, Y, C = load_side_actions("train")
    n = len(qids)
    lab = np.full(n, -1)
    for j in range(n):
        ok = [(C[j, i], i) for i in range(4) if Y[j, i]]
        if ok:
            lab[j] = min(ok)[1]
    keep = lab >= 0
    print(f"train-side queries: {n} ({(~keep).sum()} unsolved kept for cost head only)")
    print("label dist:", {MODELS[i]: int((lab == i).sum()) for i in range(4)})

    idx = np.arange(n); rng = np.random.RandomState(a.seed); rng.shuffle(idx)
    nv = int(n * a.val_frac); va, tr = idx[:nv], idx[nv:]
    enc = SentenceTransformer(ENCODER)
    X = np.asarray(enc.encode(prompts, normalize_embeddings=False, show_progress_bar=False,
                              batch_size=128), dtype=np.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.trunk = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(a.hidden, a.hidden), nn.ReLU())
            s.head_m = nn.Linear(a.hidden, 4); s.head_c = nn.Linear(a.hidden, 4)

        def forward(s, x):
            h = s.trunk(x); return s.head_m(h), s.head_c(h)

    cnt = np.array([max((lab[tr] == c).sum(), 1) for c in range(4)], float)
    beta = 0.999; w = (1 - beta) / (1 - beta ** cnt); w = w / w.sum() * 4
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), ignore_index=-1)
    huber = nn.SmoothL1Loss()
    Xt, Lt, Ct = torch.tensor(X[tr]), torch.tensor(lab[tr], dtype=torch.long), torch.log1p(torch.tensor(C[tr]))
    Xv, Lv = torch.tensor(X[va]), lab[va]

    best = (-1e9, None, -1); log = []
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            lg, lc = net(Xt[b])
            loss = ce(lg, Lt[b]) + a.cost_weight * huber(lc, Ct[b])
            loss.backward(); opt.step()
        if ep % 5 == 0 or ep == a.epochs:
            net.eval()
            with torch.no_grad():
                lg, _ = net(Xv)
            pred = lg.argmax(1).numpy(); m = Lv >= 0
            score = float(accuracy_score(Lv[m], pred[m]))
            log.append({"epoch": ep, "val_exact": score})
            print(f"  ep {ep:3d}  val exact {score:.3f}")
            if score > best[0]:
                best = (score, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        lg, lc = net(Xv)
    pred = lg.argmax(1).numpy(); m = Lv >= 0
    print(f"\nbest epoch {best[2]}  val exact {accuracy_score(Lv[m], pred[m]):.3f}  "
          f"macro-F1 {f1_score(Lv[m], pred[m], average='macro'):.3f}")
    cp = np.expm1(lc.numpy())[np.arange(len(va)), pred]; ct = C[va][np.arange(len(va)), pred]
    print(f"cost head: MAE {np.mean(np.abs(cp - ct)):.1f} units on the chosen model")

    torch.save({"state": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": a.hidden,
                "models": MODELS, "arch": "searchcost_trunk_2heads", "encoder": ENCODER,
                "cost_transform": "log1p",
                "label": "argmin realized VC-free search cost among correct models"},
               os.path.join(a.outdir, "router_model.pt"))
    json.dump({"encoder": ENCODER, "models": MODELS,
               "labels": "least realized search cost among correct (planner actions, VC-free)",
               "n_fit": int(len(tr)), "n_val": int(len(va)), "best_epoch": best[2],
               "val_exact": float(accuracy_score(Lv[m], pred[m])), "curve": log},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"saved -> {a.outdir}/router_model.pt")


if __name__ == "__main__":
    main()
