"""
Preliminary experiment (revised): query-level preferences over TTS graph SHAPE
and BUDGET, across MATH / MMLU / HumanEval, using LLaMA-1B as the unit model.

Setup
-----
* Unit model = LLaMA-3.2-1B. Agents AND the aggregation (FinalDecision) node are
  LLaMA-1B: a fully-parallel graph is several independent 1B instances whose
  answers are aggregated directly.
* Budget range, in 1B-inference units:
      b_min = cost(one direct LLaMA-1B inference)  = 1.0
      b_max = cost(one LLaMA-8B inference) / cost(one LLaMA-1B inference)
  i.e. "how many small (1B) calls buy one 8B call". Several geometric levels in
  [b_min, b_max].
* Query-ADAPTIVE lengths: the FLOPs cost of every invocation uses this query's
  input length (from the text) and the real per-node output tokens (measured
  after running) rather than a fixed reference.  Graph SIZING uses (A_q, B_dom):
  the query's input length and a per-domain typical output length.
* Four shapes, each SIZED TO THE BUDGET (largest that fits, closest-from-below):
      parallel   -> independent 1B leaves, aggregated directly (realize_parallel)
      sequential -> chain of refinements                        (realize_chain)
      wide-tree  -> one root broadcasts to leaves -> aggregate  (realize_star)
      deep-tree  -> binary tree, log-depth                      (realize_tree)
  (parallel is DISTINCT from wide-tree: parallel leaves are independent; wide-tree
  leaves consume the root's output.)
* 15 queries from the TRAINING split of each benchmark (MATH train, MMLU dev,
  HumanEval train) = 45 total.

Per query the best-performing config is chosen; ties broken by lowest ACTUAL
FLOPs cost (real tokens). We report per-benchmark distributions of preferred
shape, preferred budget, and their joint. Preferred configs at N=1 are labelled
"single-node" (all shapes are the same graph there); shape ties are labelled
"tie" -- so the shape signal is honest.

Note: every graph includes one FinalDecision (aggregation) call, so the smallest
graph (1 agent + decision) costs ~2 units; b_min=1 (a single direct 1B call) is a
scale reference the smallest graph slightly exceeds.

Usage
-----
    python experiments_query/experiment_tts_preference.py --limit 15 --num-budgets 6
    python experiments_query/experiment_tts_preference.py --analyze-only
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from typing import Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from experiments_query.run_tts import run_tts_graph, _design_num_nodes
from experiments_query.budget import MODEL_SPECS, Task, unit_cost, node_cost

from experiments.evaluator.datasets.math_dataset import MATHDataset
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from experiments.evaluator.datasets.humaneval_dataset import HumanEvalDataset

AGENT_MODEL = "meta-llama/llama-3.2-1b-instruct"     # TTS graphs are built from 1B
DECISION_MODEL = "meta-llama/llama-3.2-1b-instruct"  # aggregation node is 1B
UNIT_MODEL = "meta-llama/llama-3.2-1b-instruct"      # 1 unit = one direct 1B call
MAX_MODEL = "meta-llama/llama-3.1-70b-instruct"      # b_max = one 70B call
BASELINE_DECISION = "meta-llama/llama-3.2-1b-instruct"  # single-1B baseline aggregator

STRUCTURES = {
    "parallel": "parallel",
    "sequential": "chain",
    "wide-tree": "star",
    "deep-tree": "tree",
}
SPLITS = {"math": "train", "mmlu": "dev", "humaneval": "train"}
DOMAIN_OUTPUT = {"math": 512, "mmlu": 64, "humaneval": 384}   # B_q for sizing
MAX_NODES = 12   # cap graph size (a 40-deep 1B chain is impractical / unhelpful)

CACHE_PATH = "experiments_query/results/tts_pref1b_cache.json"
REPORT_PATH = "experiments_query/results/tts_pref1b_report.json"


# --------------------------------------------------------------------------- #
# Per-query budget levels and sizing (query-adaptive lengths)
# --------------------------------------------------------------------------- #
def est_input_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def query_task(domain: str, inputs: Dict) -> Task:
    return Task(est_input_tokens(inputs["task"]), DOMAIN_OUTPUT.get(domain, 256))


def geom(lo: float, hi: float, k: int) -> List[float]:
    if k == 1:
        return [round(hi, 3)]
    if hi <= lo:
        return [round(lo, 3)] * k
    r = (hi / lo) ** (1.0 / (k - 1))
    return [round(lo * r ** i, 3) for i in range(k)]


def query_levels(task: Task, k: int) -> List[float]:
    b_max = unit_cost(MODEL_SPECS[MAX_MODEL], task) / unit_cost(MODEL_SPECS[UNIT_MODEL], task)
    return geom(1.0, b_max, k)


def query_mapping(task: Task, levels: List[float]):
    """(structure, level_idx) -> {N, level}; plus the distinct (structure, N)."""
    mapping, distinct = {}, set()
    for name, topo in STRUCTURES.items():
        for j, b in enumerate(levels):
            n = _design_num_nodes(topo, [AGENT_MODEL], DECISION_MODEL, b, task, hard_cap=MAX_NODES) or 1
            mapping[(name, j)] = {"N": n, "level": b}
            distinct.add((name, n))
    return mapping, sorted(distinct)


def median_b_max(domain, ds, n_q) -> float:
    vals = []
    for i in range(n_q):
        t = query_task(domain, ds.record_to_swarm_input(ds[i]))
        vals.append(unit_cost(MODEL_SPECS[MAX_MODEL], t) / unit_cost(MODEL_SPECS[UNIT_MODEL], t))
    vals.sort()
    return round(vals[len(vals) // 2], 2) if vals else 0.0


# --------------------------------------------------------------------------- #
# Datasets, scoring, actual cost
# --------------------------------------------------------------------------- #
def load_datasets():
    return {
        "math": MATHDataset(SPLITS["math"]),
        "mmlu": MMLUDataset(SPLITS["mmlu"]),
        "humaneval": HumanEvalDataset(SPLITS["humaneval"]),
    }


def score(domain, dataset, record, raw_answer, inputs) -> float:
    try:
        if domain == "humaneval":
            return 1.0 if dataset.postprocess_answer(raw_answer, inputs) is True else 0.0
        ans = dataset.postprocess_answer(raw_answer)
        return 1.0 if ans == dataset.record_to_target_answer(record) else 0.0
    except Exception:
        return 0.0


def actual_flops(nodes: List[Dict]) -> float:
    """Real FLOPs from each executed node's measured prompt/completion tokens."""
    total = 0.0
    for nd in nodes:
        m, pt, ct = nd.get("model"), nd.get("prompt_tokens"), nd.get("completion_tokens")
        if m in MODEL_SPECS and pt is not None:
            total += node_cost(pt, ct, MODEL_SPECS[m])
    return total


# --------------------------------------------------------------------------- #
# Run phase (cached, resumable)
# --------------------------------------------------------------------------- #
def _load_cache():
    return json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, CACHE_PATH)


async def run_all(datasets, limit, num_budgets, concurrency):
    cache = _load_cache()
    sem = asyncio.Semaphore(concurrency)
    done = [0]

    async def one(domain, idx, record, structure, n, task):
        key = f"{domain}|{idx}|{structure}|{n}"
        if key in cache:
            return
        async with sem:
            try:
                ds = datasets[domain]
                inputs = ds.record_to_swarm_input(record)
                out = await run_tts_graph(
                    AGENT_MODEL, budget=1e9, graph=STRUCTURES[structure],
                    domain=domain, inputs=inputs, num_nodes=n,
                    task_lengths=(task.avg_input_len, task.avg_output_len),
                    decision_model=DECISION_MODEL)
                perf = score(domain, ds, record, out["answer"], inputs)
                af = actual_flops(out["nodes"])
            except Exception as e:
                perf, af = 0.0, float("nan")
                print(f"[err] {key}: {type(e).__name__}: {str(e)[:120]}")
            cache[key] = {"perf": perf, "actual_flops": af}
            done[0] += 1
            if done[0] % 20 == 0:
                _save_cache(cache)
                print(f"  ... {done[0]} new runs cached")

    async def baseline_one(domain, idx, record, task):
        # single-1B baseline: 1 agent + 1B aggregator, defines scaling-sensitivity.
        key = f"{domain}|{idx}|baseline"
        if key in cache:
            return
        async with sem:
            try:
                ds = datasets[domain]
                inputs = ds.record_to_swarm_input(record)
                out = await run_tts_graph(
                    AGENT_MODEL, budget=1e9, graph="parallel",
                    domain=domain, inputs=inputs, num_nodes=1,
                    task_lengths=(task.avg_input_len, task.avg_output_len),
                    decision_model=BASELINE_DECISION)
                perf = score(domain, ds, record, out["answer"], inputs)
                af = actual_flops(out["nodes"])
            except Exception as e:
                perf, af = 0.0, float("nan")
                print(f"[err] {key}: {type(e).__name__}: {str(e)[:120]}")
            cache[key] = {"perf": perf, "actual_flops": af}
            done[0] += 1

    tasks = []
    for domain, ds in datasets.items():
        for idx in range(min(limit, len(ds))):
            inputs = ds.record_to_swarm_input(ds[idx])
            task = query_task(domain, inputs)
            tasks.append(baseline_one(domain, idx, ds[idx], task))
            _, distinct = query_mapping(task, query_levels(task, num_budgets))
            for structure, n in distinct:
                tasks.append(one(domain, idx, ds[idx], structure, n, task))
    print(f"scheduling {len(tasks)} graph runs (cached skipped)...")
    await asyncio.gather(*tasks)
    _save_cache(cache)
    return cache


# --------------------------------------------------------------------------- #
# Analysis: best config per query -> distributions
# --------------------------------------------------------------------------- #
def analyze(cache, datasets, limit, num_budgets):
    per_domain = {}
    for domain, ds in datasets.items():
        pref_shape, pref_budget, joint = Counter(), Counter(), Counter()
        base_solved = scaling_sensitive = unsolved_all = 0
        n_q = min(limit, len(ds))
        for idx in range(n_q):
            inputs = ds.record_to_swarm_input(ds[idx])
            task = query_task(domain, inputs)
            levels = query_levels(task, num_budgets)
            mapping, _ = query_mapping(task, levels)

            base = cache.get(f"{domain}|{idx}|baseline")
            base_perf = base["perf"] if base else 0.0

            configs = []   # (perf, actual_flops, level_idx, structure, N)
            for (structure, j), info in mapping.items():
                e = cache.get(f"{domain}|{idx}|{structure}|{info['N']}")
                if not e:
                    continue
                configs.append((e["perf"], e["actual_flops"], j, structure, info["N"]))
            if not configs:
                continue

            best_perf = max(c[0] for c in configs)
            multinode_best = max([c[0] for c in configs if c[4] >= 2], default=0.0)

            # scaling-sensitive = single 1B fails AND some multi-node graph succeeds
            if base_perf > 0:
                base_solved += 1
                continue
            if multinode_best <= 0:
                unsolved_all += 1
                continue
            scaling_sensitive += 1

            best = [c for c in configs if c[0] == best_perf]
            best.sort(key=lambda c: c[1])           # lowest ACTUAL cost first
            _, top_cost, top_j, top_struct, top_N = best[0]
            if top_N == 1:
                shape = "single-node"
            else:
                close = [c for c in best if c[1] <= top_cost * 1.02]
                shape = "tie" if len({c[3] for c in close}) > 1 else top_struct

            pref_shape[shape] += 1
            pref_budget[top_j] += 1
            joint[(shape, top_j)] += 1

        per_domain[domain] = {
            "n_queries": n_q,
            "baseline_solved": base_solved,
            "scaling_sensitive": scaling_sensitive,
            "unsolved_all": unsolved_all,
            "median_b_max": median_b_max(domain, ds, n_q),
            "pref_shape": dict(pref_shape),
            "pref_budget_level": {str(k): v for k, v in sorted(pref_budget.items())},
            "joint": {f"{s}@L{j}": v for (s, j), v in joint.items()},
        }
    return per_domain


def print_report(per_domain, num_budgets):
    def dist(counter, keys=None):
        total = sum(counter.values()) or 1
        keys = keys or sorted(counter, key=lambda k: (-counter.get(k, 0), str(k)))
        return "  ".join(f"{k}:{counter.get(k,0)} ({100*counter.get(k,0)/total:.0f}%)" for k in keys)

    shape_keys = list(STRUCTURES) + ["single-node", "tie"]
    print("\n" + "=" * 78)
    print("SCALING-SENSITIVE TTS PREFERENCE  (1B agents + 8B aggregator; unit = 1 direct 1B call)")
    print("only queries where a single 1B fails but a multi-node graph succeeds")
    print("=" * 78)
    all_shape, all_budget = Counter(), Counter()
    for domain, d in per_domain.items():
        print(f"\n### {domain.upper()}  scaling-sensitive {d['scaling_sensitive']}/{d['n_queries']} "
              f"(single-1B solved {d['baseline_solved']}, unsolved-by-all {d['unsolved_all']}); "
              f"b_max~{d['median_b_max']} 1B-units over {num_budgets} levels")
        print("1) preferred shape        :", dist(d["pref_shape"], shape_keys))
        pb = {int(k): v for k, v in d["pref_budget_level"].items()}
        print(f"2) preferred budget level (L0=b_min .. L{num_budgets-1}=b_max):", dist(pb))
        print("3) joint shape@level:")
        for k, v in sorted(d["joint"].items(), key=lambda kv: -kv[1]):
            print(f"     {k}: {v}")
        for k, v in d["pref_shape"].items():
            all_shape[k] += v
        for k, v in d["pref_budget_level"].items():
            all_budget[int(k)] += v
    print("\n### OVERALL (scaling-sensitive queries)")
    print("preferred shape        :", dist(all_shape, shape_keys))
    print("preferred budget level :", dist(all_budget))


# --------------------------------------------------------------------------- #
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--num-budgets", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    datasets = load_datasets()
    if args.analyze_only: # 如果之前运行过了，使用这个命令就无需再次运行了
        cache = _load_cache()
    else:
        cache = await run_all(datasets, args.limit, args.num_budgets, args.concurrency)

    per_domain = analyze(cache, datasets, args.limit, args.num_budgets)
    print_report(per_domain, args.num_budgets)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    json.dump(per_domain, open(REPORT_PATH, "w"), indent=2)
    print("\nsaved report ->", REPORT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
