"""
Run a test-time scaling (TTS) collaboration graph on a single query.

`run_tts_graph(models, budget, graph, query, domain)` builds the TTS graph by
REUSING the GPTSwarm `Swarm` (agents + FinalDecision) and its `realize_*`
topology builders, computes the graph's normalized FLOPs budget with the
`experiments_query.budget` package, runs the query through the graph on the
OpenRouter backbone, and returns the result.

Inputs
------
models : str | list[str]
    One OpenRouter model id applied to every node, or a per-node list (its
    length sets the number of agent nodes).
budget : float
    Budget in normalized units (1 unit = one inference of the smallest model in
    the pool). Reported and compared against the graph's computed cost; the
    graph is still run if it exceeds the budget (a warning is logged).
graph : str | dict
    Topology. A string archetype -- "chain" | "star" | "tree" | "parallel" |
    "full" | "direct" -- or a dict {"topology": "custom", "edges": [(i, j), ...]}
    with agent-index edges (sinks are auto-connected to the FinalDecision node).
query : str
    The task/question text.
domain : str
    Prompt/answer domain (e.g. "math", "gsm8k", "mmlu").

Returns a dict with the answer, the absolute FLOPs, the normalized budget units,
whether it is within `budget`, and a per-node (model, in-degree, tokens) list.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

from swarm.graph.swarm import Swarm
from swarm.llm import LLMRegistry
from swarm.environment.operations.final_decision import MergingStrategy

from experiments_query.budget import (
    MODEL_SPECS, Task, graph_cost, unit_cost,
)

ARCHETYPES = ("chain", "star", "tree", "parallel", "full", "direct")


# --------------------------------------------------------------------------- #
# Graph construction (reuses the swarm's realize_* builders)
# --------------------------------------------------------------------------- #
def _custom_graph(swarm: Swarm, edges: Sequence[Tuple[int, int]]):
    """Build a graph from agent-index edges; auto-wire sinks to FinalDecision."""
    g = deepcopy(swarm.composite_graph)
    nodes = [ag.output_nodes[0] for ag in g.graphs]   # agent nodes, in order
    final = g.decision_method
    for src, dst in edges:
        s, d = nodes[src], nodes[dst]
        if not g.check_cycle(d, {s}, set()):
            s.add_successor(d)                        # sets both links
    for n in nodes:                                    # sinks -> decision
        if len(n.successors) == 0 and not g.check_cycle(final, {n}, set()):
            n.add_successor(final)
    return g


def _build_graph(swarm: Swarm, topology: str, edges=None):
    cd = swarm.connection_dist
    base = swarm.composite_graph
    if topology in ("full", "full_connected"):
        return cd.realize_full(base)
    if topology == "chain":
        return cd.realize_chain(base)
    if topology == "star":
        return cd.realize_star(base)
    if topology == "tree":
        return cd.realize_tree(base)
    if topology in ("parallel", "direct"):
        return cd.realize_parallel(base)
    if topology == "custom":
        return _custom_graph(swarm, edges or [])
    raise ValueError(f"unknown topology {topology!r} (use one of {ARCHETYPES} or 'custom')")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
# The budget unit is ALWAYS one inference of LLaMA-3.2-1B (a single node).
UNIT_MODEL_NAME = "meta-llama/llama-3.2-1b-instruct"


def _pool(models) -> List[str]:
    return [models] if isinstance(models, str) else list(models)


def _agent_models(pool: List[str], n: int) -> List[str]:
    """Assign a model to each of n nodes by cycling the pool."""
    # 这里是有问题的，cycle the pool只能保证模型数量是一致的，但是不能保证模型大小被合理分配了
    return [pool[i % len(pool)] for i in range(n)]


def _topology_in_degrees(topology: str, n: int) -> Tuple[List[int], int]:
    """Analytical (agent in-degrees, decision in-degree) for an archetype at
    size n, matching the swarm's realize_* builders. Used to size the graph to
    the budget without repeatedly building swarms."""
    if n <= 1:
        return [0], 1
    if topology == "chain":
        return [0] + [1] * (n - 1), 1
    if topology in ("parallel", "direct"):
        return [0] * n, n
    if topology in ("full", "full_connected"):
        return list(range(n)), n
    if topology == "star":
        return [0] + [1] * (n - 1), n - 1
    if topology == "tree":                       # binary heap: parent i -> 2i+1, 2i+2
        agent = [0] + [1] * (n - 1)              # every non-root has exactly one parent
        leaves = sum(1 for i in range(n) if 2 * i + 1 >= n)
        return agent, leaves
    raise ValueError(f"cannot auto-size topology {topology!r}")


def _units_for_n(topology: str, n: int, pool: List[str],
                 decision_model: str, task: Task) -> float:
    """Normalized budget (unit = one LLaMA-1B node) of an n-node archetype."""
    agent_deg, dec_deg = _topology_in_degrees(topology, n)
    specs = [MODEL_SPECS[m] for m in _agent_models(pool, n)] + [MODEL_SPECS[decision_model]]
    degs = agent_deg + [dec_deg]
    return graph_cost(specs, degs, task) / unit_cost(MODEL_SPECS[UNIT_MODEL_NAME], task)


def _design_num_nodes(topology: str, pool: List[str], decision_model: str,
                      budget: float, task: Task, hard_cap: int = 128) -> Optional[int]:
    """Largest node count whose cost <= budget (closest to budget from below).

    Cost is monotonically increasing in n for every archetype, so we grow n
    until it would exceed the budget. Returns None if even one node exceeds it.
    """
    best = None
    cap = max(1, min(hard_cap, int(budget) + 8))
    for n in range(1, cap + 1):
        if _units_for_n(topology, n, pool, decision_model, task) <= budget:
            best = n
        else:
            break
    return best


def _extract_answer(raw, domain: str):
    """Best-effort final answer extraction for scoring-friendly domains."""
    if domain in ("math", "gsm8k"):
        try:
            from experiments.evaluator.datasets.math_dataset import extract_boxed_answers
            boxed = extract_boxed_answers(raw if isinstance(raw, str) else str(raw))
            if boxed:
                return boxed[-1]
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
async def run_tts_graph(
    models: Union[str, Sequence[str]],
    budget: float,
    graph: Union[str, Dict],
    query: Optional[str] = None,
    domain: str = "math",
    *,
    inputs: Optional[Dict] = None,
    decision_model: Optional[str] = None,
    num_nodes: Optional[int] = None,
    task_lengths: Optional[Tuple[float, float]] = None,
    strategy: MergingStrategy = MergingStrategy.SelectBest,
) -> Dict:
    # Accept a raw query string OR a full inputs dict (HumanEval needs
    # {"task": ..., "tests": ...}). The query text only feeds the task-length
    # estimate when task_lengths is not given.
    if inputs is None:
        inputs = {"task": query if query is not None else ""}
    if query is None:
        query = str(inputs.get("task", ""))

    agent_models: Sequence[str]
    pool = _pool(models)
    decision_model = decision_model or pool[0]
    topology = graph if isinstance(graph, str) else graph.get("topology", "custom")
    edges = None if isinstance(graph, str) else graph.get("edges")

    # Task-average lengths drive the (analytical) FLOPs budget.
    if task_lengths is None:
        task = Task(avg_input_len=max(1, len(query) // 4), avg_output_len=256)  # 输出长度根据经验自动调整
    else:
        task = Task(*task_lengths)  # 或者这里可以手动输入

    # --- decide the number of agent nodes ---------------------------------- #
    # By default the graph is SIZED TO THE BUDGET: pick the largest archetype
    # whose cost <= budget (closest to the budget without exceeding it). An
    # explicit num_nodes or a custom edge list fixes the size instead.
    designed = False
    if topology == "custom":
        n = (1 + max(max(i, j) for i, j in edges)) if edges else 1
    elif topology == "direct":
        n = 1
    elif num_nodes is not None:
        n = int(num_nodes)
    else:
        missing = [m for m in set(pool + [decision_model]) if m not in MODEL_SPECS]
        if missing:
            raise ValueError(
                f"budget-driven design needs FLOPs specs for all models; missing {missing}")
        n = _design_num_nodes(topology, pool, decision_model, budget, task)
        designed = True
        if n is None:
            print(f"[warn] even one node exceeds budget {budget}; using 1 node")
            n = 1
    agent_models = _agent_models(pool, n)
    num_agents = n

    # 1) Reuse the swarm to hold the agents + FinalDecision node.
    model_names = list(agent_models) + [decision_model]
    swarm = Swarm(
        num_agents * ["IO"], domain,
        model_names=model_names,
        final_node_class="FinalDecision",
        final_node_kwargs=dict(strategy=strategy, use_verifier=False),
        edge_optimize=True, models_cost=None, use_verifier=False,
    )

    # 2) Realize the requested topology.
    g = _build_graph(swarm, topology, edges)

    # FinalDecision hardcodes an ollama model; point it at the chosen backbone.
    g.decision_method.llm = LLMRegistry.get(decision_model)
    g.decision_method.model_name = decision_model

    # 3) FLOPs budget from the BUILT graph (unit = LLaMA-1B, fixed).
    specs, degs, nodes_info, unknown = [], [], [], []
    for nid, node in g.nodes.items():
        mname = getattr(node, "model_name", None)
        indeg = len(node.predecessors)
        info = {"id": nid, "model": mname, "in_degree": indeg,
                "is_decision": node is g.decision_method}
        nodes_info.append(info)
        spec = MODEL_SPECS.get(mname)
        if spec is None:
            unknown.append(mname)
            continue
        specs.append(spec)
        degs.append(indeg)

    flops = graph_cost(specs, degs, task)
    units = flops / unit_cost(MODEL_SPECS[UNIT_MODEL_NAME], task)
    within = bool(units <= budget)
    utilization = (units / budget) if budget else float("nan")
    if not within:
        print(f"[warn] graph budget {units:.3f} exceeds limit {budget} (running anyway)")
    if unknown:
        print(f"[warn] no FLOPs spec for models {sorted(set(unknown))}; excluded from cost")

    # 4) Run the graph on the query.
    res = await g.run(inputs)
    answers = res[0] if isinstance(res, tuple) and len(res) == 2 else res
    answer = answers[-1] if isinstance(answers, list) and answers else answers

    # attach actual token counts per node
    for info in nodes_info:
        node = g.nodes.get(info["id"])
        outs = getattr(node, "outputs", None)
        if outs and isinstance(outs[-1], dict) and outs[-1].get("cost"):
            p, c, _ = outs[-1]["cost"]
            info["prompt_tokens"], info["completion_tokens"] = p, c

    return {
        "answer": answer,
        "extracted": _extract_answer(answer, domain),
        "flops": flops,
        "budget_units": units,
        "budget_limit": budget,
        "budget_utilization": utilization,
        "within_budget": within,
        "designed": designed,
        "unit_model": UNIT_MODEL_NAME,
        "num_agents": num_agents,
        "topology": topology,
        "nodes": nodes_info,
        "unknown_models": sorted(set(unknown)),
    }


def run_tts_graph_sync(*args, **kwargs) -> Dict:
    return asyncio.run(run_tts_graph(*args, **kwargs))


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #
def _demo():
    parser = argparse.ArgumentParser(description="Run a TTS graph on one query.")
    parser.add_argument("--models", type=str, default="qwen/qwen-2.5-7b-instruct",
                        help="comma-separated model ids (or a single id)")
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--graph", type=str, default="chain",
                        help="chain|star|tree|parallel|full|direct")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="fix the node count; omit to size the graph to the budget")
    parser.add_argument("--domain", type=str, default="math")
    parser.add_argument("--query", type=str,
                        default="What is the smallest value of $x$ such that "
                                "$|5x-1|=|3x+2|$? Express your answer as a common fraction.")
    args = parser.parse_args()

    models = args.models.split(",") if "," in args.models else args.models
    out = run_tts_graph_sync(
        models, args.budget, args.graph, args.query, args.domain,
        num_nodes=args.num_nodes)

    print("\n=== TTS run ===")
    print("topology     :", out["topology"],
          f"({out['num_agents']} agents, {'designed' if out['designed'] else 'fixed'})")
    print("FLOPs        :", f"{out['flops']:.3e}")
    print("budget units :", f"{out['budget_units']:.3f} / {out['budget_limit']} "
                            f"({out['budget_utilization']*100:.1f}% used, "
                            f"within={out['within_budget']}, unit={out['unit_model']})")
    print("answer       :", out["answer"])
    if out["extracted"] is not None:
        print("extracted    :", out["extracted"])
    print("nodes:")
    for nd in out["nodes"]:
        tag = "decision" if nd["is_decision"] else "agent"
        toks = f"  tokens in/out={nd.get('prompt_tokens','?')}/{nd.get('completion_tokens','?')}"
        print(f"  [{tag:8}] d={nd['in_degree']} {nd['model']}{toks}")


if __name__ == "__main__":
    _demo()
