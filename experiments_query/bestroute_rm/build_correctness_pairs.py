"""Correctness-targeted preference pairs for verifier v2.

From judged_samples.jsonl (every cached generation of the oracle search, graded by the LLM
judge) build (chosen=correct, rejected=incorrect) pairs of the SAME prompt. Within-model pairs
are what the read-out needs (rank this model's own samples); cross-model pairs are added when a
prompt has too few within-model pairs. Split by PROMPT (10% validation, seed 1); the validation
prompt ids are also what eval_readout.py uses, so read-out accuracy is measured on prompts the
verifier never trained on. Text format is the deployed one: "Human: {prompt} Assistant: {response}".
"""
from __future__ import annotations
import argparse, collections, json, logging, os, random
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pairs")
GT = "datasets/best_route/mixed_dataset_groundtruth.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judged", default="experiments_query/results/verifier_v2/judged_samples.jsonl")
    p.add_argument("--out", default="experiments_query/results/verifier_v2/reward_modelling_correctness")
    p.add_argument("--max-pairs-per-prompt", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    from datasets import Dataset, DatasetDict
    gt = {r["id"]: r["prompt"] for r in (json.loads(l) for l in open(GT, encoding="utf-8") if l.strip())}
    by = collections.defaultdict(list)
    n = 0
    for l in open(a.judged, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l); n += 1
        by[r["qid"]].append(r)
    qids = sorted(by); rng = random.Random(a.seed); rng.shuffle(qids)
    nval = int(len(qids) * a.val_frac); val = set(qids[:nval])
    logger.info("samples %d over %d prompts; val prompts %d", n, len(qids), nval)
    fmt = lambda q, t: f"Human: {gt[q]} Assistant: {t}"
    stats = collections.Counter(); dd = {}
    for split in ("train", "validation"):
        chosen, rejected = [], []
        for q in qids:
            if (q in val) != (split == "validation"):
                continue
            rows = by[q]
            pairs = []
            bym = collections.defaultdict(list)
            for r in rows:
                bym[r["model"]].append(r)
            for m, rs in bym.items():                       # within-model pairs first
                ok = [r for r in rs if r["correct"]]; bad = [r for r in rs if not r["correct"]]
                for c in ok:
                    for b in bad:
                        pairs.append((c, b, "within"))
            if len(pairs) < 4:                               # top up with cross-model pairs
                ok = [r for r in rows if r["correct"]]; bad = [r for r in rows if not r["correct"]]
                cross = [(c, b, "cross") for c in ok for b in bad if c["model"] != b["model"]]
                rng.shuffle(cross); pairs += cross[: max(0, 8 - len(pairs))]
            if not pairs:
                stats[f"{split}_prompts_no_pair"] += 1
                continue
            rng.shuffle(pairs); pairs = pairs[: a.max_pairs_per_prompt]
            for c, b, kind in pairs:
                chosen.append(fmt(q, c["text"])); rejected.append(fmt(q, b["text"])); stats[f"{split}_{kind}"] += 1
            stats[f"{split}_prompts"] += 1
        dd[split] = Dataset.from_dict({"chosen": chosen, "rejected": rejected})
        logger.info("%-10s prompts=%d pairs=%d", split, stats[f"{split}_prompts"], len(chosen))
    logger.info("stats: %s", dict(stats))
    os.makedirs(a.out, exist_ok=True)
    DatasetDict(dd).save_to_disk(a.out)
    json.dump({"val_prompts": sorted(val), "seed": a.seed}, open(os.path.join(os.path.dirname(a.out), "prompt_split.json"), "w"))
    logger.info("saved -> %s (+ prompt_split.json)", a.out)


if __name__ == "__main__":
    main()
