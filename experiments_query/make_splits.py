"""Reproducible 80/20 train/test split, STRATIFIED within each stratum
(Best-Route-Mix, by source), on SOLVED records only, deduped, seeded.  MATH / MMLU use the native
splits of the query ids in datasets/splits/mathmmlu_qids.json (see task_bench.splits).
Writes split manifests under datasets/splits/ and prints per-stratum counts.
"""
import json, os, random
from collections import defaultdict, Counter

SEED = 42
TRAIN_FRAC = 0.80
OUT = "datasets/splits"
os.makedirs(OUT, exist_ok=True)
BR = "experiments_query/results/bestroute_tts/bestroute_tts.jsonl"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stratified_split(items, key_fn, stratum_fn, seed=SEED):
    """items -> dict key->'train'/'test', 80/20 within each stratum, deduped by key."""
    by_stratum = defaultdict(list)
    seen = set()
    for it in items:
        k = key_fn(it)
        if k in seen:
            continue
        seen.add(k)
        by_stratum[stratum_fn(it)].append(k)
    assign, counts = {}, {}
    for stratum, keys in by_stratum.items():
        keys = sorted(keys)                      # determinism before shuffle
        random.Random(seed).shuffle(keys)
        n_train = round(len(keys) * TRAIN_FRAC)
        for i, k in enumerate(keys):
            assign[k] = "train" if i < n_train else "test"
        counts[stratum] = (n_train, len(keys) - n_train)
    return assign, counts


# ---- best-route (solved only), stratified by source ----
br = [r for r in load(BR) if not r.get("hard_unsolved") and (r.get("best_width") or r.get("best_depth"))]
br_assign, br_counts = stratified_split(br, lambda r: r["id"], lambda r: r["source"])
with open(f"{OUT}/split_bestroute.jsonl", "w", encoding="utf-8") as f:
    written = set()
    for r in br:
        if r["id"] not in written:
            written.add(r["id"])
            f.write(json.dumps({"id": r["id"], "source": r["source"], "split": br_assign[r["id"]]}) + "\n")

# ---- combined manifest ----
manifest = {"seed": SEED, "train_frac": TRAIN_FRAC,
            "bestroute": {s: {"train": t, "test": v} for s, (t, v) in br_counts.items()},
            }
json.dump(manifest, open(f"{OUT}/split_manifest.json", "w"), indent=2)

# ---- report ----
print(f"stratified 80/20 split  (seed={SEED})\n" + "=" * 46)
print(f"{'stratum':<22}{'train':>8}{'test':>8}{'total':>8}")
tot_tr = tot_te = 0
for group, counts in (("BEST-ROUTE (by source)", br_counts),):
    print(f"-- {group} --")
    for s in sorted(counts):
        tr, te = counts[s]; tot_tr += tr; tot_te += te
        print(f"  {s:<20}{tr:>8}{te:>8}{tr+te:>8}")
print("=" * 46)
print(f"  {'TOTAL':<20}{tot_tr:>8}{tot_te:>8}{tot_tr+tot_te:>8}")
print(f"\nwrote: {OUT}/split_bestroute.jsonl, split_manifest.json")
