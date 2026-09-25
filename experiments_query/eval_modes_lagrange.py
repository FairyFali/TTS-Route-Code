import os
"""Lagrangian decode a*(q)=argmax_a p_a(q) - lambda*cbar_a with a fine lambda grid, for (i) the 11-mode
router and (ii) the lite-p-6 router (6 actions) -- isolates the value of the extra modes from the decode
rule.  Writes results/tts_router_modes/lagrange_fine.json and litep6_lagrange.json; prints the budget table."""
import json, collections, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sentence_transformers import SentenceTransformer
from experiments_query.train_router_probe_chs_lite import build_features
from experiments_query.train_router_litep6 import build_six, ACTIONS as A6
from experiments_query.train_router_modes import build, ACTIONS as A11

def load(outdir, nout):
    ck = torch.load(f"{outdir}/router_model.pt", map_location="cpu", weights_only=False)
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.trunk = nn.Sequential(nn.Linear(ck["in_dim"], ck["hidden"]), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ck["hidden"], ck["hidden"]), nn.ReLU()); s.head = nn.Linear(ck["hidden"], nout)
        def forward(s, x): return s.head(s.trunk(x))
    net = Net(); net.load_state_dict(ck["state"]); net.eval()
    return net, np.array(ck["feat_mu"], np.float32), np.array(ck["feat_sd"], np.float32), set(json.load(open(f"{outdir}/meta.json"))["val_ids"])

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LAM = np.concatenate([np.linspace(0, 0.003, 61), np.linspace(0.003, 0.03, 55)[1:], np.linspace(0.03, 0.3, 28)[1:]])
budgets = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100]
def sweep(name, outdir, builder, acts):
    net, mu, sd, val_ids = load(outdir, len(acts)); D = {}
    for side in ("train", "test"):
        out = builder(side); qids, prompts, Y, C = out[0], out[1], out[2], out[3]
        X, _ = build_features(side, qids, prompts, enc, mu, sd)
        with torch.no_grad(): P = torch.sigmoid(net(torch.tensor(X))).numpy()
        D[side] = (qids, Y, C, P)
    qtr, Ytr, Ctr, Ptr = D["train"]; vm = np.array([q in val_ids for q in qtr]); qte, Yte, Cte, Pte = D["test"]
    cbar = Ctr.mean(0); rows = []
    for lam in LAM:
        pv = np.argmax(Ptr[vm] - lam * cbar, 1); pt = np.argmax(Pte - lam * cbar, 1)
        rows.append({"lambda": float(lam), "val_acc": float(Ytr[vm][np.arange(vm.sum()), pv].mean()), "val_cost": float(Ctr[vm][np.arange(vm.sum()), pv].mean()),
                     "test_acc": float(Yte[np.arange(len(pt)), pt].mean()), "test_cost": float(Cte[np.arange(len(pt)), pt].mean()), "mix": dict(collections.Counter(acts[x] for x in pt))})
    json.dump(rows, open(f"experiments_query/results/tts_router_modes/{name}.json", "w"), indent=1); return rows
R11 = sweep("lagrange_fine", "experiments_query/results/tts_router_modes", build, A11)
R6 = sweep("litep6_lagrange", "experiments_query/results/tts_router_litep6", build_six, A6)
p6 = json.load(open("experiments_query/results/tts_router_litep6/pipeline.json")); old = [{"val_acc": 0.762, "val_cost": 9.83, "test_acc": 0.773, "test_cost": 9.49}] + json.load(open("experiments_query/results/tts_router_probe_chs_lite3/permodel_pipelines.json"))["lite"]
final6 = [r for r in p6 if r["val_cost"] < 30] + old
def sel(rr, b): ok = [r for r in rr if r["val_cost"] <= b]; return max(ok, key=lambda r: (r["val_acc"], -r["val_cost"])) if ok else None
f = lambda r: f"{r['test_acc']:.3f}@{r['test_cost']:5.1f}" if r else "   --  "
print(f"{'b':>4} | {'modes (11) Lagrangian':>21} | {'lite-p-6 Lagrangian':>19} | {'lite-p FINAL ladder':>19} | modes mix (>=3%)")
for b in budgets:
    r11, r6, rf = sel(R11, b), sel(R6, b), sel(final6, b)
    mix = {k: v for k, v in sorted(r11["mix"].items(), key=lambda kv: -kv[1]) if v >= 34} if r11 else {}
    print(f"{b:>4} | {f(r11):>21} | {f(r6):>19} | {f(rf):>19} | {mix}")
