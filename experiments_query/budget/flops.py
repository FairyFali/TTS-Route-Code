"""
FLOPs-based cost and budget for multi-LLM collaboration graphs.

Implements the cost function from the paper appendix (FLOPs metric). Conventions:
one multiply-add = 2 FLOPs; causal self-attention reuses cached keys/values
during decode.

Per node with non-embedding parameters ``M`` (params), hidden size ``D``
(hidden), and layers ``L`` (layers), and effective prefill/decode lengths
``Np``/``Nd``:

    cost_prefill(Np)      = 2*M*Np + 2*L*D*Np*(Np + 1)
    cost_decode(Np, Nd)   = 2*M*Nd + 2*L*D*Nd*(2*Np + Nd + 1)
    node_cost             = cost_prefill + cost_decode

In a graph, the effective lengths for a node with in-degree ``d`` on task
``T = (A, B)`` (average input A = N_p^T, average output B = N_d^T) are

    Np = A + d*B      # task input + one output per predecessor, concatenated
    Nd = B

(optionally ``d -> min(d, k)`` when a fuser keeps only its top-k predecessors).

The graph cost is the sum over nodes; equivalently each node is quadratic in d:

    node_cost = alpha*d^2 + beta*d + gamma
    alpha = 2*L*D*B^2
    beta  = 2*M*B + 2*L*D*B*(2A + 2B + 1)
    gamma = 2*(M + L*D)*(A + B) + 2*L*D*(A + B)^2

Budget normalization: one unit = one full inference of the smallest model as a
single node (in-degree 0). The normalized budget of G is
``cost(G) / cost(single smallest node)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Architecture of the model running at a node.

    params: M, non-embedding parameter count.
    hidden: D, hidden size.
    layers: L, number of transformer layers.
    """
    name: str
    params: float
    hidden: float
    layers: float


@dataclass(frozen=True)
class Task:
    """Average input/output lengths of a task, T = (N_p^T, N_d^T)."""
    avg_input_len: float   # A
    avg_output_len: float  # B


# --------------------------------------------------------------------------- #
# Node-level FLOPs
# --------------------------------------------------------------------------- #
def cost_prefill(n_p: float, model: ModelSpec) -> float:
    """FLOPs to prefill n_p tokens: projection/MLP (2*M*n_p) + attention."""
    return (2.0 * model.params * n_p
            + 2.0 * model.layers * model.hidden * n_p * (n_p + 1.0))


def cost_decode(n_p: float, n_d: float, model: ModelSpec) -> float:
    """FLOPs to decode n_d tokens after a prefill of n_p (KV cache reused)."""
    return (2.0 * model.params * n_d
            + 2.0 * model.layers * model.hidden * n_d * (2.0 * n_p + n_d + 1.0))


def node_cost(n_p: float, n_d: float, model: ModelSpec) -> float:
    """Total prefill + decode FLOPs for a node with lengths (n_p, n_d)."""
    return cost_prefill(n_p, model) + cost_decode(n_p, n_d, model)


# --------------------------------------------------------------------------- #
# Effective lengths / quadratic form in the in-degree
# --------------------------------------------------------------------------- #
def effective_degree(in_degree: int, top_k: Optional[int] = None) -> int:
    """Effective in-degree; a top-k fuser caps concatenated predecessors at k."""
    return in_degree if top_k is None else min(in_degree, top_k)


def effective_lengths(task: Task, in_degree: int,
                      top_k: Optional[int] = None) -> Tuple[float, float]:
    """(Np, Nd) for a node: Np = A + d*B, Nd = B, with d = min(in_degree, k)."""
    d = effective_degree(in_degree, top_k)
    n_p = task.avg_input_len + d * task.avg_output_len
    n_d = task.avg_output_len
    return n_p, n_d


def node_coeffs(model: ModelSpec, task: Task) -> Tuple[float, float, float]:
    """(alpha, beta, gamma): the node cost as a quadratic in the in-degree d."""
    A, B = task.avg_input_len, task.avg_output_len
    M = model.params
    LD = model.layers * model.hidden
    alpha = 2.0 * LD * B * B
    beta = 2.0 * M * B + 2.0 * LD * B * (2.0 * A + 2.0 * B + 1.0)
    gamma = 2.0 * (M + LD) * (A + B) + 2.0 * LD * (A + B) ** 2
    return alpha, beta, gamma


def node_cost_by_degree(model: ModelSpec, task: Task, in_degree: int,
                        top_k: Optional[int] = None) -> float:
    """Node FLOPs via the direct (prefill+decode) formula from the in-degree."""
    n_p, n_d = effective_lengths(task, in_degree, top_k)
    return node_cost(n_p, n_d, model)


def node_cost_quadratic(model: ModelSpec, task: Task, in_degree: int,
                        top_k: Optional[int] = None) -> float:
    """Node FLOPs via the quadratic alpha*d^2 + beta*d + gamma (== direct form)."""
    d = effective_degree(in_degree, top_k)
    alpha, beta, gamma = node_coeffs(model, task)
    return alpha * d * d + beta * d + gamma


# Signature of a pluggable per-node cost (remark ii: alternative metrics).
NodeCostFn = Callable[[ModelSpec, Task, int, Optional[int]], float]


# --------------------------------------------------------------------------- #
# Graph-level cost and budget
# --------------------------------------------------------------------------- #
def in_degrees_from_edges(num_nodes: int,
                          edges: Sequence[Tuple[int, int]]) -> List[int]:
    """In-degree per node index from an edge list of (src, dst) index pairs."""
    deg = [0] * num_nodes
    for src, dst in edges:
        deg[dst] += 1
    return deg


def graph_cost(models: Sequence[ModelSpec],
               in_degrees: Sequence[int],
               task: Task,
               top_ks: Optional[Sequence[Optional[int]]] = None,
               node_cost_fn: Optional[NodeCostFn] = None) -> float:
    """Total cost = sum of per-node cost.

    node_cost_fn overrides the FLOPs node cost with a calibrated surrogate
    (remark ii); graph aggregation is identical either way.
    """
    fn = node_cost_fn or node_cost_by_degree
    total = 0.0
    for i, (model, d) in enumerate(zip(models, in_degrees)):
        k = None if top_ks is None else top_ks[i]
        total += fn(model, task, d, k)
    return total


def graph_cost_from_edges(models: Sequence[ModelSpec],
                          edges: Sequence[Tuple[int, int]],
                          task: Task,
                          top_ks: Optional[Sequence[Optional[int]]] = None,
                          node_cost_fn: Optional[NodeCostFn] = None) -> float:
    """graph_cost with in-degrees derived from an edge list."""
    deg = in_degrees_from_edges(len(models), edges)
    return graph_cost(models, deg, task, top_ks, node_cost_fn)


def unit_cost(smallest_model: ModelSpec, task: Task) -> float:
    """Cost of one unit budget: a single smallest-model node (in-degree 0)."""
    return node_cost_by_degree(smallest_model, task, 0)


def budget_units(cost: float, smallest_model: ModelSpec, task: Task) -> float:
    """Normalize an absolute FLOPs cost into unit-budgets (smallest node = 1)."""
    return cost / unit_cost(smallest_model, task)


def normalized_budget(models: Sequence[ModelSpec],
                      in_degrees: Sequence[int],
                      task: Task,
                      smallest_model: ModelSpec,
                      top_ks: Optional[Sequence[Optional[int]]] = None,
                      node_cost_fn: Optional[NodeCostFn] = None) -> float:
    """Normalized budget B = cost(G) / cost(single smallest node)."""
    cost = graph_cost(models, in_degrees, task, top_ks, node_cost_fn)
    return budget_units(cost, smallest_model, task)
