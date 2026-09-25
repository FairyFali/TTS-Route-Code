"""
Tests for the FLOPs cost/budget package (experiments_query/budget).

Run:
    python experiments_query/budget/test.py
"""

import math
import os
import sys

# repo root = .../experiments_query/budget/test.py -> up two levels
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from experiments_query.budget import (
    ModelSpec, Task,
    cost_prefill, cost_decode, node_cost,
    effective_lengths, node_coeffs, node_cost_by_degree, node_cost_quadratic,
    in_degrees_from_edges, graph_cost, graph_cost_from_edges,
    unit_cost, budget_units, normalized_budget,
    MODEL_SPECS, smallest_model,
)

# A toy model + task with round numbers so costs are hand-checkable.
TOY = ModelSpec("toy", params=100, hidden=10, layers=2)   # M=100, D=10, L=2, LD=20
TASK = Task(avg_input_len=4, avg_output_len=3)             # A=4, B=3


def test_node_cost_hand_values():
    # in-degree 2 -> Np = 4 + 2*3 = 10, Nd = 3
    n_p, n_d = effective_lengths(TASK, 2)
    assert (n_p, n_d) == (10, 3)
    # prefill: 2*100*10 + 2*2*10*10*11 = 2000 + 4400 = 6400
    assert cost_prefill(n_p, TOY) == 6400
    # decode: 2*100*3 + 2*2*10*3*(2*10+3+1) = 600 + 2880 = 3480
    assert cost_decode(n_p, n_d, TOY) == 3480
    assert node_cost(n_p, n_d, TOY) == 9880
    print("PASS test_node_cost_hand_values")


def test_direct_equals_quadratic():
    # The core invariant: the direct prefill+decode formula equals the
    # alpha*d^2 + beta*d + gamma quadratic, for every model and in-degree.
    models = [TOY] + list(MODEL_SPECS.values())
    tasks = [TASK, Task(500, 200), Task(1024, 64), Task(1, 1)]
    for m in models:
        for t in tasks:
            for d in range(0, 12):
                direct = node_cost_by_degree(m, t, d)
                quad = node_cost_quadratic(m, t, d)
                assert math.isclose(direct, quad, rel_tol=1e-9, abs_tol=1e-3), \
                    (m.name, t, d, direct, quad)
    # spot-check the coefficients against the toy hand values
    alpha, beta, gamma = node_coeffs(TOY, TASK)
    assert (alpha, beta, gamma) == (360, 2400, 3640)
    print("PASS test_direct_equals_quadratic")


def test_effective_lengths_and_topk():
    # d = 0 -> Np = A (no predecessors concatenated)
    assert effective_lengths(TASK, 0) == (4, 3)
    # top-k caps the concatenated predecessors: min(5, 2) = 2
    assert node_cost_by_degree(TOY, TASK, 5, top_k=2) == node_cost_by_degree(TOY, TASK, 2)
    # top_k >= in_degree is a no-op
    assert node_cost_by_degree(TOY, TASK, 2, top_k=5) == node_cost_by_degree(TOY, TASK, 2)
    print("PASS test_effective_lengths_and_topk")


def test_graph_cost():
    # in-degrees from an edge list: 0->2, 1->2  =>  degrees [0, 0, 2]
    degs = in_degrees_from_edges(3, [(0, 2), (1, 2)])
    assert degs == [0, 0, 2]
    models = [TOY, TOY, TOY]
    # gamma (d=0) = 3640 for two leaves; d=2 node = 9880
    expected = 3640 + 3640 + 9880
    assert graph_cost(models, degs, TASK) == expected
    assert graph_cost_from_edges(models, [(0, 2), (1, 2)], TASK) == expected
    print("PASS test_graph_cost")


def test_budget_normalization():
    # one smallest node = exactly 1 unit
    unit = unit_cost(TOY, TASK)
    assert unit == 3640
    assert budget_units(unit, TOY, TASK) == 1.0
    # two independent smallest nodes = 2 units
    two = graph_cost([TOY, TOY], [0, 0], TASK)
    assert math.isclose(budget_units(two, TOY, TASK), 2.0)
    # a bigger / higher in-degree graph costs more units
    b = normalized_budget([TOY, TOY, TOY], [0, 0, 2], TASK, smallest_model=TOY)
    assert b > 3.0
    print("PASS test_budget_normalization")


def test_alternative_metric_hook():
    # remark (ii): swap the FLOPs node cost for any surrogate; aggregation stays.
    const_one = lambda model, task, d, k: 1.0
    assert graph_cost([TOY, TOY, TOY], [0, 1, 2], TASK, node_cost_fn=const_one) == 3.0
    print("PASS test_alternative_metric_hook")


def test_registry_smallest():
    s = smallest_model()
    assert s.name == "meta-llama/llama-3.2-1b-instruct"
    print(f"PASS test_registry_smallest (smallest = {s.name})")


def demo():
    """Illustrative budget numbers with real specs on a representative task."""
    task = Task(avg_input_len=1000, avg_output_len=1000)
    small = smallest_model()  # llama-3.2-1b
    q7 = MODEL_SPECS["qwen/qwen-2.5-7b-instruct"]
    l70 = MODEL_SPECS["meta-llama/llama-3.1-70b-instruct"]

    print("\n--- demo: normalized budget (unit = one llama-3.2-1b node) ---")
    print(f"task: avg_input={task.avg_input_len}, avg_output={task.avg_output_len}")

    # single-node graphs
    for m in (small, q7, l70):
        b = normalized_budget([m], [0], task, small)
        print(f"  1x {m.name:35} -> B = {b:8.3f}")

    # a 3-node cascade of 1b models (in-degrees 0,1,2) + comparison to one 70b
    cascade = normalized_budget([small, small, small], [0, 1, 2], task, small)
    one_big = normalized_budget([l70], [0], task, small)
    print(f"  cascade 3x llama-1b (deg 0,1,2)          -> B = {cascade:8.3f}")
    print(f"  1x llama-70b                             -> B = {one_big:8.3f}")
    print("  (this is the 'more small nodes' vs 'one big node' trade-off the budget equates)")


if __name__ == "__main__":
    test_node_cost_hand_values()
    test_direct_equals_quadratic()
    test_effective_lengths_and_topk()
    test_graph_cost()
    test_budget_normalization()
    test_alternative_metric_hook()
    test_registry_smallest()
    demo()
    print("\nALL TESTS PASSED")
