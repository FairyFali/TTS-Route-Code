"""Lightweight probe-CH+scores router: frozen MiniLM embeddings + MLP.

Same inputs and labels as the DeBERTa probe-CHS router — query text, both probe
generations, and their verifier scores (all free under the TTS-only protocol) — but the
backbone is frozen all-MiniLM-L6-v2: input = [emb(query); emb(1b probe); emb(8b probe);
standardized (s1b, s8b)] -> MLP(256) -> 4 correctness sigmoids + log-cost head.
Deployment cost of the router itself: three MiniLM encodes + an MLP (negligible vs a
fine-tuned DeBERTa forward).  New outdir; nothing overwritten.
"""
import argparse, json, os, random
import numpy as np

from experiments_query.train_router_searchcost import load_side_actions, MODELS

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PROBE = {"train": "experiments_query/results/probe_texts_train.json",
         "test": "experiments_query/results/probe_texts_test.json"}
NPZ = {"train": "experiments_query/results/pipeline_probe_router/train_probe_scores.npz",
       "test": "experiments_query/results/pipeline_probe_router/test_probe_scores.npz"}


def build_features(side, qids, prompts, enc, mu=None, sd=None):
    pt = json.load(open(PROBE[side]))
    E = lambda ts: np.asarray(enc.encode(ts, normalize_embeddings=False,
                                         show_progress_bar=False, batch_size=64), dtype=np.float32)
    z = np.load(NPZ[side])
    F = np.stack([z["1b"], z["8b"]], axis=1).astype(np.float32)
    if mu is not None:
        F = (F - mu) / sd
    X = np.concatenate([E(prompts), E([pt[q]["1b"] for q in qids]),
                        E([pt[q]["8b"] for q in qids]), F], axis=1)
    return X, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments_query/results/tts_router_probe_chs_lite")
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
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    qids, prompts, Y, C = load_side_actions("train")
    Yf = (Y >= 0.5).astype(np.float32); Cl = np.log1p(C.astype(np.float32))
    n = len(qids)
    idx = np.arange(n); rng = np.random.RandomState(a.seed); rng.shuffle(idx)
    nv = int(n * a.val_frac); va, tr = idx[:nv], idx[nv:]
    enc = SentenceTransformer(ENCODER)
    z = np.load(NPZ["train"]); Fraw = np.stack([z["1b"], z["8b"]], axis=1).astype(np.float32)
    mu, sd = Fraw[tr].mean(0), Fraw[tr].std(0) + 1e-6
    X, _ = build_features("train", qids, prompts, enc, mu, sd)
    print(f"train-side queries: {n} (fit {len(tr)} / val {len(va)})  input dim {X.shape[1]}")

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.trunk = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(a.hidden, a.hidden), nn.ReLU())
            s.head_m = nn.Linear(a.hidden, 4); s.head_c = nn.Linear(a.hidden, 4)

        def forward(s, x):
            h = s.trunk(x); return s.head_m(h), s.head_c(h)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); huber = nn.SmoothL1Loss()
    Xt, Rt, Ct = torch.tensor(X[tr]), torch.tensor(Yf[tr]), torch.tensor(Cl[tr])
    Xv, Rv = torch.tensor(X[va]), torch.tensor(Yf[va])

    best = (-1e9, None, -1); log = []
    for ep in range(1, a.epochs + 1):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            lg, lc = net(Xt[b])
            loss = bce(lg, Rt[b]) + a.cost_weight * huber(lc, Ct[b])
            loss.backward(); opt.step()
        if ep % 5 == 0 or ep == a.epochs:
            net.eval()
            with torch.no_grad():
                lg, _ = net(Xv)
            score = -float(bce(lg, Rv))
            log.append({"epoch": ep, "select_score": score})
            if score > best[0]:
                best = (score, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        lg, _ = net(Xv)
    prob = torch.sigmoid(lg).numpy()
    Yb = Yf[va].astype(int)
    aucs = {MODELS[i]: round(float(roc_auc_score(Yb[:, i], prob[:, i])), 3) for i in range(4)
            if 0 < Yb[:, i].sum() < len(va)}
    print(f"best epoch {best[2]}   val AUCs: {aucs}")

    torch.save({"state": net.state_dict(), "in_dim": int(X.shape[1]), "hidden": a.hidden,
                "models": MODELS, "arch": "probe_chs_lite_mlp", "encoder": ENCODER,
                "cost_transform": "log1p", "feat_mu": mu.tolist(), "feat_sd": sd.tolist()},
               os.path.join(a.outdir, "router_model.pt"))
    json.dump({"encoder": ENCODER, "models": MODELS,
               "input": "MiniLM emb(query)+emb(1b probe)+emb(8b probe)+probe scores",
               "labels": "per-model search-action correctness (VC-free)",
               "n_fit": int(len(tr)), "n_val": int(len(va)), "best_epoch": best[2],
               "val_auc": aucs, "curve": log,
               "val_ids": [qids[i] for i in va]},
              open(os.path.join(a.outdir, "meta.json"), "w"), indent=2)
    print(f"saved -> {a.outdir}/router_model.pt")


if __name__ == "__main__":
    main()
