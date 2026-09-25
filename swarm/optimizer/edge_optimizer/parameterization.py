#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Categorical
from copy import deepcopy
from typing import Tuple
from typing import Optional
from typing import List,Dict,Set
import random
import math
from functools import lru_cache
from collections import defaultdict
import re,ast

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph
from swarm.optimizer.role_optimizer.role import Role
from swarm.environment.operations.final_decision import FinalDecision
from experiments.evaluator.llm_optimizer import LLM_optimizer
from swarm.llm import LLMRegistry

class DFSLimitExceeded(Exception):
    pass

class ConnectDistribution(nn.Module):
    def __init__(self, potential_connections):
        super().__init__()
        self.potential_connections = potential_connections

    def realize(self, graph):
        raise NotImplemented


class MRFDist(ConnectDistribution):
    pass


class EdgeWiseDistribution(ConnectDistribution):
    def __init__(self,
                 potential_connections,
                 initial_probability: float = 0.5,
                 models_cost: dict = None,
                 roles: List[str] = ("IO", "Reviewer"),
                 models: List[str] = ("llama3.2-1b-longcontext:latest","llama3.2-3b-longcontext:latest","llama3.1-8b-longcontext:latest","llama3.1-70b-longcontext:latest","gemma3-1b-longcontext:latest","gemma1-2b-longcontext:latest","gemma1-7b-longcontext:latest","qwen2.5-1.5b-longcontext:latest","qwen2.5-3b-longcontext:latest","qwen2.5-7b-longcontext:latest")
                 ):
        super().__init__(potential_connections)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        self.roles = list(roles)
        self.role_logits = nn.Parameter(    # shape = [N, R]
            torch.zeros(len(node_ids), len(self.roles))
        )
        self.models = list(models)
        self.model_logits = nn.Parameter(                        # [N_node, 3]
            torch.zeros(len(node_ids), len(self.models))
        )
        self.models_cost = models_cost #{"OFF":0, "1b":1, "3b":3, "8b":16, "70b":110}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = torch.nn.Parameter(order_tensor)

    def get_hard_roles(self, graph: CompositeGraph) -> dict[int, int]:
        """
        返回 {node_id: role_idx} ，直接取 argmax(prob) 作为确定角色
        （跳过 FinalDecision）
        """
        hard_roles = {}
        final_id   = graph.decision_method.id
        for node in graph.nodes.values():
            if node.id == final_id:
                continue
            logits = self.role_logits[self.node_id2idx[node.id]]
            role_idx = torch.argmax(logits).item()
            hard_roles[node.id] = role_idx
        return hard_roles

    def sample_roles(self, graph: CompositeGraph,kept_ids,
                     hard: bool = False) -> Tuple[Dict[int, int], torch.Tensor]:
        
        def is_useful(node):
            if node in graph.output_nodes:
                return True
            
            for successor in node.successors:
                if is_useful(successor):
                    return True
            return False
        
        useful = {nid for nid,n in graph.nodes.items() if is_useful(n) and nid in kept_ids}

        # 3) 计算每个有用节点的入度，初始化队列
        in_deg = {nid: len(graph.nodes[nid].predecessors) for nid in useful}
        queue  = [nid for nid,deg in in_deg.items() if deg==0]
        final_node_id = graph.decision_method.id
        roles  = {}
        log_ps = []

        while queue:
            nid = queue.pop(0)
            node = graph.nodes[nid]

            # 决策节点 / FinalDecision 强制 IO，不计梯度
            if nid==final_node_id or isinstance(node, FinalDecision):
                roles[nid] = Role.IO
                log_ps.append(torch.tensor(0.0, requires_grad=False))
            else:
                # 看它所有前驱是不是 IO
                pred_ids = [p.id for p in node.predecessors]
                pred_ids = [pid for pid in pred_ids if pid in useful]
                all_pred_io = (
                    len(pred_ids) > 0
                    and all(roles.get(pid) == Role.IO for pid in pred_ids)
                )
                rev_pred_io = (len(pred_ids)>0 and any(roles.get(pid) == Role.REV for pid in pred_ids))
                if all_pred_io:
                    # 先从 learned logits 里算出原始 softmax
                    logits = self.role_logits[self.node_id2idx[nid]]
                    probs0 = torch.softmax(logits, dim=0)   # shape [2]
                    # reviewer 概率 +0.5
                    p_rev = min(1.0, float(probs0[Role.REV]) + 0.5)
                    p_io  = 1.0 - p_rev
                    prior = torch.tensor([p_io, p_rev])
                    dist  = torch.distributions.Categorical(probs=prior)
                elif rev_pred_io:
                    p_io = float(1.0)
                    p_rev = 0
                    prior = torch.tensor([p_io, p_rev])
                    dist  = torch.distributions.Categorical(probs=prior)
                else:
                    # 否则用学习到的 logits
                    logits = self.role_logits[self.node_id2idx[nid]]
                    dist   = Categorical(logits=logits)

                r = dist.sample()
                if hard:
                    # 如果是 hard 模式，就直接取最高项
                    if hasattr(dist, 'logits'):
                        r = torch.argmax(dist.logits)
                    else:
                        r = torch.argmax(prior)
                roles[nid]    = int(r.item())
                log_ps.append(dist.log_prob(r))

            for succ in node.successors:
                if succ.id in in_deg:
                    in_deg[succ.id] -= 1
                    if in_deg[succ.id]==0:
                        queue.append(succ.id)
        
        return roles, torch.sum(torch.stack(log_ps))

    def greedy_models_in_budget(
        self,
        graph: CompositeGraph,
        *,
        ranks: dict[str, int],    # topo 序
        budget_left: float,       # 以 3B=1 为单位
) -> tuple[dict[int, int], set[int]]:

        costs = [self.models_cost[m] for m in self.models]

        chosen   : dict[int, int] = {}
        kept_ids : set[int]       = set()

        for nid, _ in sorted(ranks.items(), key=lambda kv: kv[1]):
            feasible = [i for i, c in enumerate(costs) if c <= budget_left]
            if not feasible:
                break                                    # 后续节点全跳过

            logits = self.model_logits[self.node_id2idx[nid]][feasible]
            best_i = feasible[int(torch.argmax(logits))]  # 概率最大的一个

            chosen[nid]  = best_i
            kept_ids.add(nid)
            budget_left -= costs[best_i]
            if budget_left <= 0:
                break

        return chosen, kept_ids
    
    def sample_models(
    self,
    graph: CompositeGraph,
    *,
    ranks: dict[str, int],        # Topo 排序
    budget_left: float,           # 以 3B=1 为单位
    hard: bool = False,
) -> tuple[dict[int, int], torch.Tensor, set[int]]:
        """
        topo 顺序采模型；超预算后 **停止采样**，其余节点留给上层去“跳过”。
        """
        useful = sorted(ranks.items(), key=lambda kv: kv[1])  # [(nid, rank)…]

        chosen   : dict[int, int] = {}     # nid -> model idx
        logps    : list[torch.Tensor] = []
        kept_ids : set[int] = set()        # 参与本轮采样的节点

        costs = [self.models_cost[m] for m in self.models]

        for nid, _ in useful:
            feasible = [i for i, c in enumerate(costs) if c <= budget_left]
            if not feasible:          # 预算不够 → 提前结束；后续节点视为“本轮跳过”
                break

            logits_row = self.model_logits[self.node_id2idx[nid]][feasible]

            if hard:
                idx_local = torch.argmax(logits_row)
            else:
                idx_local = Categorical(logits=logits_row).sample()

            idx = feasible[int(idx_local)]
            logps.append(torch.log_softmax(logits_row, 0)[idx_local])

            chosen[nid] = idx
            kept_ids.add(nid)
            budget_left -= costs[idx]

        joint_lp = torch.sum(torch.stack(logps)) if logps else torch.tensor(0.0)
        return chosen, joint_lp, kept_ids

    def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
        _graph = deepcopy(graph)
        while True:
            if _graph.num_edges >= num_edges:
                break
            potential_connection = random.sample(self.potential_connections, 1)[0]
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    """def realize_ranks(self, graph, use_max: bool = False):
        log_probs = []
        ranks = {}
        in_degrees = {node.id: len(node.predecessors) for node in graph.nodes.values()}
        for i in range(len(self.order_params)):
            avaliable_nodes = [node for node in graph.nodes if in_degrees[node] == 0]
            logits = []
            for node in avaliable_nodes:
                logits.append(self.order_params[self.node_id2idx[node]])
            logits = torch.stack(logits).reshape(-1)
            if use_max:
                idx = torch.argmax(logits)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            log_probs.append(torch.log_softmax(logits, dim=0)[idx])

            ranks[avaliable_nodes[idx]] = i
            in_degrees[avaliable_nodes[idx]] = -1
            for successor in graph.nodes[avaliable_nodes[idx]].successors:
                in_degrees[successor.id] -= 1
        return ranks, torch.sum(torch.stack(log_probs))"""
        

    def realize_ranks(self, graph):
        node_ids = [
            nid for nid in graph.nodes
            if nid != graph.decision_method.id
        ]
        random.shuffle(node_ids)
        return {nid: idx for idx, nid in enumerate(node_ids)}
    
    def realize_bayes_models(self,
                graph: CompositeGraph,
                model_selection
                ):
        _graph = deepcopy(graph)
        model_pool: List[str] = []
        for m, cnt in model_selection.items():
            model_pool += [int(m)] * int(cnt)

        ranks = self.realize_ranks(_graph)
        models_count = len(model_pool)
        ranks = sorted(ranks.items(), key=lambda kv: kv[1])[:models_count]

        kept_ids = [node[0] for node in ranks]
        kept_ids.append(_graph.decision_method.id)
        ids = [node for node in _graph.nodes]
        useless_ids = [nid for nid in ids if nid not in kept_ids]
        for nid in useless_ids:
            _graph.nodes.pop(nid, None)

        for nid, midx in zip(kept_ids[:-1],model_pool):
            node = _graph.find_node(nid)
            # node.model_idx  = midx
            # node.model_name = self.models[midx+1]
            node.model_idx  = midx+1
            node.model_name = self.models[midx+1]
            if hasattr(node, "llm"):
                node.llm = LLMRegistry.get(node.model_name)
        return _graph

    def realize_bayes(
        self,
        graph          : CompositeGraph,
        *,
        edges_sample   : dict[str, int] | None = None,
        roles_sample   : dict[str, int] | None = None,
    ):
        """
        根据 edges_sample / roles_sample 构图，保证无环。
        若 sample 为空，则退化为随机连边 + 随机角色。
        返回 (graph, dummy_log_prob) —— 第二项给调用方占位即可。
        """
        # 2.0 base_graph 已含模型分配；这里再复制一次以免污染
        g        = deepcopy(graph)
        fd_id    = g.decision_method.id
        kept_ids = list(g.nodes.keys())

        # ============ 2.1 角色分配 =============
        for nid in kept_ids:
            if nid == fd_id:
                g.nodes[nid].role = Role.IO          # 决策节点固定 IO
                continue

            # 若有给定 roles_sample，就用它；否则随机
            if roles_sample is not None:
                key = f"r_{nid}"
                r   = roles_sample.get(key, 0)
            else:
                r   = random.randint(0, 1)
            g.nodes[nid].behavior_fn = g.nodes[nid].io_behavior  if r == 0 else g.nodes[nid].review_behavior
            g.nodes[nid].role_name = "IO" if r == 0 else "REV"
            g.nodes[nid].role_idx = 0 if r == 0 else 1

        # ============ 2.2 连边（保证 DAG） =========
        # 默认全清空旧边
        for i in range(2000):
            for node in g.nodes.values():
                node.successors.clear()
                node.predecessors.clear()

            for (src, dst) in self.potential_connections:
                if src not in kept_ids or dst not in kept_ids:
                    continue

                # 决定这条边是否保留
                if edges_sample is not None:
                    flag = edges_sample.get(f"e_{src}_{dst}", 0)
                else:
                    flag = random.randint(0, 1)

                if flag == 0:
                    continue

                s_node = g.find_node(src)
                d_node = g.find_node(dst)
                # 避免造成环
                if not g.check_cycle(d_node, {s_node}, set()):
                    s_node.add_successor(d_node)
                    d_node.add_predecessor(s_node)

            if len(g.decision_method.predecessors) == 0:
                continue

            return g
        
    def realize_random(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        
        max_try = 2000                       # 防止死循环
        for _ in range(max_try):
            if use_learned_order:
                ranks, log_prob = self.realize_ranks(graph, threshold is not None)
                log_probs = [log_prob]
            else:
                log_probs = [torch.tensor(0.0, requires_grad=True)]
            _graph = deepcopy(graph)
            for potential_connection, edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                out_node = _graph.find_node(potential_connection[0])
                in_node = _graph.find_node(potential_connection[1])

                if not out_node or not in_node:
                    continue
                
                addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
                addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
                if addable_if_not_used_learned_order or addable_if_use_learned_order:
                    edge_prob = torch.sigmoid(edge_logit - 2 / temperature)
                    if threshold:
                        edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                    if torch.rand(1) < edge_prob:
                        out_node.add_successor(in_node)
                        in_node.add_predecessor(out_node)
                        log_probs.append(torch.log(edge_prob))
                    else:
                        log_probs.append(torch.log(1 - edge_prob))

            final_node = _graph.decision_method
            if len(final_node.predecessors) < 1:
                continue

            log_prob_edges = torch.sum(torch.stack(log_probs))
        
        roles = {nid: Role.IO for nid in graph.nodes}
        _graph = deepcopy(graph)
        _graph.set_roles(roles)
        joint_log_prob = torch.tensor(0.0, requires_grad=True) + log_prob_edges

        return _graph, joint_log_prob

    def realize(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                budget: bool = False
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        
        max_try = 2000
        _graph = deepcopy(graph)
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(_graph, threshold is not None)
            log_ord = log_prob
        else:
            log_ord = torch.tensor(0.0, requires_grad=True)

        if budget:
            ranks = self.realize_ranks(_graph)
            
            models, log_mod, kept_ids = self.sample_models(
                _graph,
                ranks=ranks,
                budget_left=float(budget),
                hard=False,
            )
            kept_ids.add(_graph.decision_method.id)
            # 把选中模型写回 node（方便后面执行 / 可视化）
            for nid, midx in models.items():
                node = _graph.find_node(nid)
                node.model_idx  = midx
                node.model_name = self.models[midx]
                if hasattr(node, "llm"):
                    node.llm = LLMRegistry.get(node.model_name)
        else:
            # 不启用预算：全部节点都参与后续步骤
            kept_ids = set(_graph.nodes.keys())
            log_mod  = torch.tensor(0.0, requires_grad=False)    
                                   
        for _ in range(max_try):
            
            log_list = [log_ord, log_mod]
            for (src_id, dst_id), edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                if src_id not in kept_ids or dst_id not in kept_ids:
                    continue

                src = _graph.find_node(src_id)
                dst = _graph.find_node(dst_id)
                if not src or not dst:
                    continue
                
                addable_if_use_learned_order = use_learned_order and ranks[src_id] < ranks[dst_id]
                addable_if_not_used_learned_order = (not use_learned_order) and \
                           (not _graph.check_cycle(dst, {src}, set()))
                if addable_if_not_used_learned_order or addable_if_use_learned_order:
                    edge_prob = torch.sigmoid(edge_logit - 2 / temperature)
                    if threshold is not None:
                        edge_prob = torch.tensor(1.0) if edge_prob > threshold else torch.tensor(0.0)

                    if torch.rand(1) < edge_prob:
                        src.add_successor(dst)
                        dst.add_predecessor(src)
                        log_list.append(torch.log(edge_prob))
                    else:
                        log_list.append(torch.log(1 - edge_prob))

            final_node = _graph.decision_method
            if len(final_node.predecessors) < 1:
                continue

            log_prob_edges = torch.sum(torch.stack(log_list))

            roles, log_prob_roles = self.sample_roles(_graph,kept_ids)
            if _graph.check_role_constraints(
                    roles,
                    io_idx = Role.IO,
                    reviewer_idx = Role.REV):
                _graph.set_roles(roles)
                joint_log_prob = log_prob_roles + log_prob_edges
                return _graph, joint_log_prob
        
            # 连续 max_try 次都没采到合法配置 —— 直接回退为全 IO
        roles = {nid: Role.IO for nid in _graph.nodes}
        _graph = deepcopy(graph)
        _graph.set_roles(roles)
        joint_log_prob = torch.tensor(0.0, requires_grad=True) + log_prob_edges

        return _graph, joint_log_prob
    
    def realize_insight(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                model_selection = None
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        
        max_try = 2000
        _graph = deepcopy(graph)
        node_ids = [
            nid for nid in _graph.nodes
            if nid != _graph.decision_method.id
        ]
        ranks = {nid: idx for idx, nid in enumerate(node_ids)}

        model_pool: List[str] = []
        for m, cnt in model_selection.items():
            model_pool += [int(m)] * int(cnt)

        models_count = len(model_pool)
        ranks = sorted(ranks.items(), key=lambda kv: kv[1])[:models_count]

        kept_ids = [node[0] for node in ranks]
        kept_ids.append(_graph.decision_method.id)
        ids = [node for node in _graph.nodes]
        useless_ids = [nid for nid in ids if nid not in kept_ids]
        for nid in useless_ids:
            _graph.nodes.pop(nid, None)

        for nid, midx in zip(kept_ids[:-1],model_pool):
            node = _graph.find_node(nid)
            node.model_idx  = midx
            node.model_name = self.models[midx]
            if hasattr(node, "llm"):
                node.llm = LLMRegistry.get(node.model_name)   

        log_mod = torch.tensor(0.0, requires_grad=False)


        if use_learned_order:
            ranks, log_prob = self.realize_ranks(_graph, threshold is not None)
            log_ord = log_prob
        else:
            log_ord = torch.tensor(0.0, requires_grad=True)

        for _ in range(max_try):
            
            log_list = [log_ord, log_mod]
            for (src_id, dst_id), edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                if src_id not in kept_ids or dst_id not in kept_ids:
                    continue

                src = _graph.find_node(src_id)
                dst = _graph.find_node(dst_id)
                if not src or not dst:
                    continue
                
                addable_if_use_learned_order = use_learned_order and ranks[src_id] < ranks[dst_id]
                addable_if_not_used_learned_order = (not use_learned_order) and \
                           (not _graph.check_cycle(dst, {src}, set()))
                if addable_if_not_used_learned_order or addable_if_use_learned_order:
                    edge_prob = torch.sigmoid(edge_logit - 2 / temperature)
                    if threshold is not None:
                        edge_prob = torch.tensor(1.0) if edge_prob > threshold else torch.tensor(0.0)

                    if torch.rand(1) < edge_prob:
                        src.add_successor(dst)
                        dst.add_predecessor(src)
                        log_list.append(torch.log(edge_prob))
                    else:
                        log_list.append(torch.log(1 - edge_prob))

            final_node = _graph.decision_method
            if len(final_node.predecessors) < 1:
                continue

            log_prob_edges = torch.sum(torch.stack(log_list))

            roles, log_prob_roles = self.sample_roles(_graph,kept_ids)
            if _graph.check_role_constraints(
                    roles,
                    io_idx = Role.IO,
                    reviewer_idx = Role.REV):
                _graph.set_roles(roles)
                joint_log_prob = log_prob_roles + log_prob_edges
                return _graph, joint_log_prob
        
            # 连续 max_try 次都没采到合法配置 —— 直接回退为全 IO
        roles = {nid: Role.IO for nid in _graph.nodes}
        _graph = deepcopy(graph)
        _graph.set_roles(roles)
        joint_log_prob = torch.tensor(0.0, requires_grad=True) + log_prob_edges

        return _graph, joint_log_prob
    
    def initialize_realize(self,
                graph: CompositeGraph,
                model_selection,
                temperature: float = 1.0,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        
        max_try = 2000
        _graph = deepcopy(graph)
        model_pool: List[str] = []
        for m, cnt in model_selection.items():
            model_pool += [int(m)] * int(cnt)

        ranks = self.realize_ranks(_graph)
        models_count = len(model_pool)
        ranks = sorted(ranks.items(), key=lambda kv: kv[1])[:models_count]

        kept_ids = [node[0] for node in ranks]
        kept_ids.append(_graph.decision_method.id)
        ids = [node for node in _graph.nodes]
        useless_ids = [nid for nid in ids if nid not in kept_ids]
        for nid in useless_ids:
            _graph.nodes.pop(nid, None)

        for nid, midx in zip(kept_ids[:-1],model_pool):
            node = _graph.find_node(nid)
            # node.model_idx  = midx
            # node.model_name = self.models[midx]
            node.model_idx  = midx+8
            node.model_name = self.models[midx+8]
            if hasattr(node, "llm"):
                node.llm = LLMRegistry.get(node.model_name)    

        log_list: List[torch.Tensor] = []                       
        for _ in range(max_try):
            
            for (src_id, dst_id), edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                if src_id not in kept_ids or dst_id not in kept_ids:
                    continue

                src = _graph.find_node(src_id)
                dst = _graph.find_node(dst_id)
                if not src or not dst:
                    continue
                no_cycle = not _graph.check_cycle(dst, {src}, set())
                if no_cycle:
                    edge_prob = torch.sigmoid(edge_logit - 2 / temperature)

                    if torch.rand(1) < edge_prob:
                        src.add_successor(dst)
                        dst.add_predecessor(src)
                        log_list.append(torch.log(edge_prob))
                    else:
                        log_list.append(torch.log(1 - edge_prob))

            final_node = _graph.decision_method
            if len(final_node.predecessors) < 1:
                continue

            log_prob_edges = torch.sum(torch.stack(log_list))

            roles, log_prob_roles = self.sample_roles(_graph,kept_ids)
            if _graph.check_role_constraints(
                    roles,
                    io_idx = Role.IO,
                    reviewer_idx = Role.REV):
                _graph.set_roles(roles)
                joint_log_prob = log_prob_roles + log_prob_edges
                return _graph, kept_ids
        
            # 连续 max_try 次都没采到合法配置 —— 直接回退为全 IO
        roles = {nid: Role.IO for nid in _graph.nodes}
        _graph = deepcopy(graph)
        _graph.set_roles(roles)
        joint_log_prob = torch.tensor(0.0, requires_grad=True) + log_prob_edges

        return _graph, kept_ids
    
    def pretest_realize(self,
                graph: CompositeGraph,
                model_selection,):
        _graph = deepcopy(graph)
        max_try = 2000
        model_pool: List[str] = []
        for m, cnt in model_selection.items():
            model_pool += [int(m)] * int(cnt)

        ranks = self.realize_ranks(_graph)
        models_count = len(model_pool)
        ranks = sorted(ranks.items(), key=lambda kv: kv[1])[:models_count]
        useful_nodes = [k for k, _ in ranks]
        useful_nodes.append(_graph.decision_method.id)
        for nid, midx in zip(useful_nodes[:-1],model_pool):
            node = _graph.find_node(nid)
            node.model_idx  = midx   # +1
            node.model_name = self.models[midx]
            if hasattr(node, "llm"):
                node.llm = LLMRegistry.get(node.model_name)
        for _ in range(max_try):
            for (src_id, dst_id), edge_logit in zip(
                    self.potential_connections, self.edge_logits):
                if src_id not in useful_nodes or dst_id not in useful_nodes:
                    continue

                src = _graph.find_node(src_id)
                dst = _graph.find_node(dst_id)
                if not src or not dst:
                    continue
                no_cycle = not _graph.check_cycle(dst, {src}, set())
                if no_cycle:
                    edge_prob = torch.sigmoid(edge_logit)

                    if torch.rand(1) < edge_prob:
                        src.add_successor(dst)
                        dst.add_predecessor(src)

            final_node = _graph.decision_method
            if len(final_node.predecessors) >= 1:
                break
        return _graph

    def model_pretest_initialize(self,graph,model_selection_list,max_return=2):
        order = ["0", "1", "2", "3"]  # weakest → strongest
        all_graphs = []
        all_selections = []

        for model_selection in model_selection_list:
            selections = [model_selection.copy()]  # 包含原始组合
            cur_sel = model_selection.copy()

            for key in order:
                while cur_sel[key] > 0 and len(selections) < max_return:
                    cur_sel = cur_sel.copy()
                    cur_sel[key] -= 1
                    if sum(cur_sel.values()) <= 0:
                        continue
                    selections.append(cur_sel.copy())

                if len(selections) >= max_return:
                    break

            for sel in selections:
                graph_out = self.pretest_realize(
                    graph=graph,
                    model_selection=sel,
                )
                all_graphs.append(graph_out)
                all_selections.append(sel)

        return all_graphs, all_selections
    
    def llm_realize(self,
        graph: CompositeGraph,
        model_selection,
        his_graph,
        acc_now,
        edge_probs_now,
        his_response,
        optimizer):
        _graph = deepcopy(graph)
        _graph, edge_prob, response, price = optimizer.llm_forward(
            prev_graph=his_graph,
            acc=acc_now,
            edge_probs=edge_probs_now,
            his_response=his_response,
            model_selection=model_selection,
            graph = _graph,
            models = self.models
        )
        return _graph, edge_prob, response, price
    
    def llm_ablation_role_realize(self,
        graph: CompositeGraph,
        model_selection,
        his_graph,
        acc_now,
        edge_probs_now,
        his_response,
        optimizer):
        _graph = deepcopy(graph)
        _graph, edge_prob, response = optimizer.llm_forward_ablation_role(
            prev_graph=his_graph,
            acc=acc_now,
            edge_probs=edge_probs_now,
            his_response=his_response,
            model_selection=model_selection,
            graph = _graph,
            models = self.models
        )
        return _graph, edge_prob, response
    
    def latency_realize(self,
        graph: CompositeGraph,
        model_selection,
        his_graph,
        acc_now,
        edge_probs_now,
        his_response,
        optimizer,
        inference_list):
        _graph = deepcopy(graph)
        _graph, edge_prob, response = optimizer.llm_forward_latency(
            prev_graph=his_graph,
            acc=acc_now,
            edge_probs=edge_probs_now,
            his_response=his_response,
            model_selection=model_selection,
            graph = _graph,
            models = self.models,
            latency_list=inference_list
        )
        return _graph, edge_prob, response
    
    def textgrad_realize(self,
        graph: CompositeGraph,
        model_selection,
        his_graph,
        acc_now,
        edge_probs_now,
        his_response,
        optimizer):
        _graph = deepcopy(graph)
        _graph, edge_prob, response, price = optimizer.llm_forward_textgrad(
            prev_graph=his_graph,
            acc=acc_now,
            edge_probs=edge_probs_now,
            his_response=his_response,
            model_selection=model_selection,
            graph = _graph,
            models = self.models
        )
        return _graph, edge_prob, response, price

    def entropy_regularizer(self, lambda_Q=0.05, eps=1e-8):
        edge_probs = torch.sigmoid(self.edge_logits)
        dst_index = torch.tensor(
            [dst for (_, dst) in self.potential_connections],
            device=edge_probs.device
        )
        num_nodes = int(dst_index.max().item()) + 1

        S = torch.zeros(num_nodes, device=edge_probs.device).scatter_add_(
            0, dst_index, edge_probs
        )
        q_tilde = edge_probs / (S[dst_index] + eps)
        ent_per_edge = - q_tilde * torch.log(q_tilde + eps)

        H_cols = torch.zeros(num_nodes, device=edge_probs.device).scatter_add_(
            0, dst_index, ent_per_edge
        )
        R_entropy = H_cols.sum()
        return lambda_Q * R_entropy

    def edge_count_regularizer(self, lambda_E=0.05, target_edges=None, squared=False):
        edge_probs = torch.sigmoid(self.edge_logits)
        expected_num_edges = edge_probs.sum()

        if target_edges is None:
            penalty = expected_num_edges
        else:
            excess_edges = torch.relu(expected_num_edges - float(target_edges))
            penalty = excess_edges.pow(2) if squared else excess_edges

        return lambda_E * penalty
    
    
    def make_all_nodes_useful(self,
                              graph: CompositeGraph,
                              prob_thresh: float = 0.2,
                              max_calls = 3) -> CompositeGraph:
        """
        对 graph 做原地修改；确保所有节点到 final 有路径。
        prob_thresh: 可选：过滤极低概率边
        """

        def is_useful(node):
            if node in graph.output_nodes:
                return True
            
            for successor in node.successors:
                if is_useful(successor):
                    return True
            return False

        final_id = graph.decision_method.id

        # 1) 先算现有 useful
        useful = {nid for nid,n in graph.nodes.items() if is_useful(n)}

        # 2) 概率字典（所有潜在边）
        with torch.no_grad():
            probs = torch.sigmoid(self.edge_logits).tolist()
        prob_dict = {
            conn: p for conn, p in zip(self.potential_connections, probs)
            if p >= prob_thresh
        }

        for _ in range(max_calls):
            changed = False
            
            # 3) 遍历所有无用节点，连最大概率路径
            for nid, node in graph.nodes.items():
                if nid in useful or nid == final_id:
                    continue
                ok = self._fallback_connect_to_best_useful(
                    nid=nid,
                    useful_ids=useful,
                    prob_dict=prob_dict,
                    graph=graph
                )
                if ok:
                    # 新连上后马上标记为 useful（可选：也可等下一轮统一重算）
                    useful.add(nid)
                    changed = True
            if not changed:
                break

        return graph

    def make_all_kept_nodes_useful(self,
                              graph: CompositeGraph,
                              kept_ids,
                              prob_thresh: float = 0.2,
                              max_calls = 3) -> CompositeGraph:
        """
        对 graph 做原地修改；确保所有节点到 final 有路径。
        prob_thresh: 可选：过滤极低概率边
        """

        def is_useful(node):
            if node in graph.output_nodes:
                return True
            
            for successor in node.successors:
                if is_useful(successor):
                    return True
            return False

        final_id = graph.decision_method.id

        # 1) 先算现有 useful
        useful = {nid for nid,n in graph.nodes.items() if is_useful(n) and (nid in kept_ids)}

        # 2) 概率字典（所有潜在边）
        with torch.no_grad():
            probs = torch.sigmoid(self.edge_logits).tolist()
        prob_dict = {
            conn: p for conn, p in zip(self.potential_connections, probs)
            if p >= prob_thresh
            and conn[0] in kept_ids and conn[1] in kept_ids
        }

        for _ in range(max_calls):
            changed = False
            
            # 3) 遍历所有无用节点，连最大概率路径
            for nid, node in graph.nodes.items():
                if nid in useful or nid == final_id or nid not in kept_ids:
                    continue
                ok = self._fallback_connect_to_best_useful(
                    nid=nid,
                    useful_ids=useful,
                    prob_dict=prob_dict,
                    graph=graph
                )
                if ok:
                    # 新连上后马上标记为 useful（可选：也可等下一轮统一重算）
                    useful.add(nid)
                    changed = True
            if not changed:
                break

        return graph
        
        '''try:
                max_logp, path = self._best_path_to_final(nid, final_id, prob_dict, graph)
                if path and len(path) > 1:
                    # 把路径中缺失的边加进去
                    for u, v in zip(path[:-1], path[1:]):
                        out_node = graph.find_node(u)
                        in_node  = graph.find_node(v)
                        # 再次检查是否成环 & 是否已存在
                        if in_node in out_node.successors:
                            continue
                        if not graph.check_cycle(in_node, {out_node}, set()):
                            out_node.add_successor(in_node)
                            in_node.add_predecessor(out_node)
                else:
                    self._fallback_connect_to_best_useful(nid, useful, prob_dict, graph)
            except DFSLimitExceeded:
                self._fallback_connect_to_best_useful(nid, useful, prob_dict, graph)

        return graph'''
    
    def _best_path_to_final(self,
                        start_id: int,
                        final_id: int,
                        prob_dict: dict[tuple, float],
                        graph,
                        max_calls = 3):
        """
        返回 (max_log_prob, path_ids[list])；若无路则 (-inf, [])
        在潜在边上做 DFS：防环 + 记忆化
        """
        # 1. 建邻接表
        adj = {}
        for (u, v), p in prob_dict.items():
            adj.setdefault(u, []).append((v, p))

        calls = 0

        # 2. DFS + 防环 + 记忆化
        @lru_cache(None)
        def dfs(u, stack_tuple=()):
            nonlocal calls
            calls += 1
            # stack_tuple：用 tuple 存当前路径上的节点
            if u == final_id:
                return 0.0, [u]

            # 当前路径上成环，剪枝
            if u in stack_tuple:
                return -math.inf, []

            best_log, best_path = -math.inf, []
            new_stack = stack_tuple + (u,)  

            for v, p in adj.get(u, []):
                if graph.check_cycle(graph.find_node(v), {graph.find_node(u)}, set()):
                     continue

                logp = math.log(max(p, 1e-12))
                sub_log, sub_path = dfs(v, new_stack)
                if sub_path:  
                    tot = logp + sub_log
                    if tot > best_log:
                        best_log = tot
                        best_path = [u] + sub_path

            return best_log, best_path

        return dfs(start_id)

    def _fallback_connect_to_best_useful(self,
                                     nid: int,
                                     useful_ids: set[int],
                                     prob_dict: dict[tuple, float],
                                     graph) -> bool:
        """
        为节点 nid 选择一个“有用节点” v（能到达 final），在潜在边里 (nid, v) 概率最大，
        且加入后不成环，则添加该边并返回 True；否则返回 False。
        """
        # 候选为所有 (nid, v) 且 v 在 useful 集合内的潜在边
        cands = [ (v, p) for (u, v), p in prob_dict.items() if u == nid and v in useful_ids ]
        if not cands:
            return False

        # 概率最大优先
        v_best, _ = max(cands, key=lambda x: x[1])

        out_node = graph.find_node(nid)
        in_node  = graph.find_node(v_best)

        # 已存在或会成环则拒绝
        if in_node in out_node.successors:
            return True  # 已连上，相当于成功
        if graph.check_cycle(in_node, {out_node}, set()):
            # 若最大概率这个会成环，可按概率从高到低再试其它候选
            for v, _ in sorted(cands, key=lambda x: -x[1]):
                in_node_alt = graph.find_node(v)
                if in_node_alt in out_node.successors:
                    return True
                if not graph.check_cycle(in_node_alt, {out_node}, set()):
                    out_node.add_successor(in_node_alt)
                    in_node_alt.add_predecessor(out_node)
                    return True
            return False

        # 安全则加边
        out_node.add_successor(in_node)
        in_node.add_predecessor(out_node)
        return True

    
    """def realize(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = [torch.tensor(0.0, requires_grad=True)]
        _graph = deepcopy(graph)
        for potential_connection, edge_logit in zip(
                self.potential_connections, self.edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    # in_node.add_predecessor(out_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob"""

    def realize_full(self, graph: CompositeGraph) -> CompositeGraph:
        _graph = deepcopy(graph)
        for i, potential_connection in enumerate(self.potential_connections):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
        """
        在 edge_mask 的基础上硬采样，同时保证每个 out_node 的出度 <= 2
        """
        _graph = deepcopy(graph)

        # 1) 挑出被选中的边并按概率降序
        cand = []
        for i, ((u, v), is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
            if not is_edge:
                continue
            prob = torch.sigmoid(self.edge_logits[i]).item()
            cand.append((prob, i, u, v))
        cand.sort(reverse=True, key=lambda x: x[0])

        # 2) 记录每个出点已连出的边数
        out_degree_cnt = {}

        for _, _, u, v in cand:
            out_node = _graph.find_node(u)
            in_node  = _graph.find_node(v)
            if not out_node or not in_node:
                continue

            # 出度约束：最多 2
            if out_degree_cnt.get(u, 0) >= 2:
                continue

            # 成环检测
            if _graph.check_cycle(in_node, {out_node}, set()):
                continue

            # 加边
            out_node.add_successor(in_node)
            in_node.add_predecessor(out_node)
            out_degree_cnt[u] = out_degree_cnt.get(u, 0) + 1

        return _graph
    
    def realize_chain(self, graph: CompositeGraph) -> CompositeGraph:
        _g = deepcopy(graph)
        agents = _g.graphs

        for i in range(len(agents) - 1):
            src_agent = agents[i]
            dst_agent = agents[i + 1]
            for out_node in src_agent.output_nodes:
                for in_node in dst_agent.input_nodes:
                    out_node.add_successor(in_node)

        # 收尾：链尾 -> FinalDecision
        final_node = _g.decision_method
        for out_node in agents[-1].output_nodes:
            if not _g.check_cycle(final_node, {out_node}, set()):
                out_node.add_successor(final_node)
                final_node.add_predecessor(out_node)

        return _g
    
    def realize_star(
        self,
        graph: "CompositeGraph",
        center_idx: int = 0,           # 可显式指定中心下标
    ) -> CompositeGraph:

        _g = deepcopy(graph)
        agents = _g.graphs
        n = len(agents)

        if n == 1:
            final_node = _g.decision_method
            for out_node in agents[0].output_nodes:
                if not _g.check_cycle(final_node, {out_node}, set()):
                    out_node.add_successor(final_node)
                    final_node.add_predecessor(out_node)
            return _g

        # -------- 1. 连接中心 → 叶节点 --------
        center_agent = agents[center_idx]
        for idx, leaf_agent in enumerate(agents):
            if idx == center_idx:
                continue
            for out_node in center_agent.output_nodes:
                for in_node in leaf_agent.input_nodes:
                    if not _g.check_cycle(in_node, {out_node}, set()):
                        out_node.add_successor(in_node)

        # -------- 2. 叶节点 → FinalDecision --------
        final_node = _g.decision_method
        for idx, leaf_agent in enumerate(agents):
            if idx == center_idx:
                continue
            for out_node in leaf_agent.output_nodes:
                if not _g.check_cycle(final_node, {out_node}, set()):
                    out_node.add_successor(final_node)
                    final_node.add_predecessor(out_node)

        return _g

    def realize_parallel(
        self,
        graph: "CompositeGraph",
    ) -> "CompositeGraph":
        """
        构造“并行 → 汇总”的拓扑：
        - 所有子图(agents)之间不互连
        - 每个子图的所有输出节点都直接指向 FinalDecision
        """
        _g = deepcopy(graph)

        # 处理只有一个 agent 的情况（也同样连接其输出到 FinalDecision）
        agents = getattr(_g, "graphs", [])
        final_node = _g.decision_method

        # 逐个 agent，把其 output_nodes 指向 FinalDecision
        for agent in agents:
            for out_node in agent.output_nodes:
                # 避免成环：final_node 作为入点检查是否会形成环
                if not _g.check_cycle(final_node, {out_node}, set()):
                    out_node.add_successor(final_node)
                    final_node.add_predecessor(out_node)

        return _g
    
    def realize_depth_parallel(self,
    graph: "CompositeGraph",
    depth: int,
) -> "CompositeGraph":

        if depth <= 0:
            raise ValueError("depth must be a positive integer")

        g = deepcopy(graph)

        final_node = g.decision_method
        all_nodes = [
            n.output_nodes[0] for n in g.graphs if n is not g.decision_method
        ]

        branches = [
            all_nodes[i : i + depth] for i in range(0, len(all_nodes), depth)
        ]

        for chain in branches:
            for a, b in zip(chain[:-1], chain[1:]):  # pairwise
                # safety: avoid accidental cycle (shouldn't happen after clearing)
                if not g.check_cycle(b, {a}, set()):
                    a.add_successor(b)
                    b.add_predecessor(a)

            tail = chain[-1]
            if not g.check_cycle(final_node, {tail}, set()):
                tail.add_successor(final_node)
                final_node.add_predecessor(tail)

        return g
    
    def realize_width_chain(
    self,
    graph: "CompositeGraph",
    width: int,
) -> "CompositeGraph":

        if width <= 0:
            raise ValueError("width must be a positive integer")

        g = deepcopy(graph)

        final_node = g.decision_method
        all_nodes = [
            n.output_nodes[0] for n in g.graphs if n is not g.decision_method
        ]

        # 按给定宽度切成若干链
        chains = [all_nodes[i : i + width] for i in range(0, len(all_nodes), width)]

        chains = [[] for _ in range(width)]

        for idx, node in enumerate(all_nodes):
            chains[idx % width].append(node)

        for chain in chains:
            if not chain:                       
                continue
            # 链内 pairwise
            for a, b in zip(chain[:-1], chain[1:]):
                if not g.check_cycle(b, {a}, set()):
                    a.add_successor(b)
                    b.add_predecessor(a)

            # 尾节点连 FinalDecision
            tail = chain[-1]
            if not g.check_cycle(final_node, {tail}, set()):
                tail.add_successor(final_node)
                final_node.add_predecessor(tail)

        return g
    
    def realize_tree(self, graph: CompositeGraph) -> CompositeGraph:
        _g: CompositeGraph = deepcopy(graph)
        agents = _g.graphs      # 按添加顺序视作层序顺序
        n = len(agents)

        # ---------- 1. 逐层连边 ----------
        for parent_idx in range(n):
            left_idx  = 2 * parent_idx + 1
            right_idx = 2 * parent_idx + 2
            parent_agent = agents[parent_idx]

            # 遍历左右孩子（若存在）
            for child_idx in (left_idx, right_idx):
                if child_idx >= n:           # 节点数不够，break
                    continue
                child_agent = agents[child_idx]

                # 父的每个 output_node → 子的每个 input_node
                for out_node in parent_agent.output_nodes:
                    for in_node in child_agent.input_nodes:
                        out_node.add_successor(in_node)

        # ---------- 2. 叶节点 → FinalDecision ----------
        final_node = _g.decision_method

        # 叶节点：没有任何孩子的 agent
        leaf_indices = [i for i in range(n) if 2 * i + 1 >= n]   # 简单层序判断
        for idx in leaf_indices:
            leaf_agent = agents[idx]
            for out_node in leaf_agent.output_nodes:
                # 防止形成环
                if not _g.check_cycle(final_node, {out_node}, set()):
                    out_node.add_successor(final_node)
                    final_node.add_predecessor(out_node)

        return _g
    
    def realize_custom_topology(
    self,
    graph: "CompositeGraph",
    models_left:  list[str] = None,   # e.g. ['llama3.2-1b-longcontext:latest'] * L
    models_mid:   list[str] = None,   # e.g. ['llama3.2-1b-longcontext:latest'] * M
    models_right: list[str] = None,   # e.g. ['llama3.2-3b-longcontext:latest'] * R
) -> "CompositeGraph":
        """
        构造与截图相同的“三链汇总”结构：
        • 左链  →  主链  →  FinalDecision
        • 右链 ↗︎
        右链第 2 个节点 role=REV，其余均 IO
        """
        models_left  = ['llama3.2-1b-longcontext:latest'] * 2  # 自行调整数量
        models_mid   = ['llama3.2-3b-longcontext:latest']
        models_right = ['llama3.2-3b-longcontext:latest','llama3.2-3b-longcontext:latest','llama3.2-1b-longcontext:latest']

        g = deepcopy(graph)

        # ------- 0. 拆分三条链 --------------------------------------------------
        left_chain, mid_chain, right_chain = [], [], []
        # 按添加顺序切分；也可传入 indices
        # 这里假设已保证 len(...) 正确
        n_left   = len(models_left)
        n_mid    = len(models_mid)
        n_right  = len(models_right)
        agents   = g.graphs

        left_chain  = agents[0:n_left]
        mid_chain   = agents[n_left:n_left + n_mid]
        right_chain = agents[n_left + n_mid : n_left + n_mid + n_right]

        final_node = g.decision_method

        # ------- 1. 模型 / 角色分配 ---------------------------------------------
        for agent, model in zip(left_chain,  models_left):
            agent.output_nodes[0].model_name = model
            agent.output_nodes[0].role_name       = "IO"

        for idx, (agent, model) in enumerate(zip(right_chain, models_right), start=1):
            agent.output_nodes[0].model_name = model
            agent.output_nodes[0].role_name       = "REV" if idx == 2 else "IO"   # 右链第二个节点

        for agent, model in zip(mid_chain,   models_mid):
            agent.output_nodes[0].model_name = model
            agent.output_nodes[0].role_name       = "IO"

        # ------- 2. 连边：左 → 主 → FinalDecision ，右 ↗︎ 主 ---------------------
        def link_chain(chain):
            for a, b in zip(chain[:-1], chain[1:]):         # pair-wise
                a.output_nodes[0].add_successor(b.input_nodes[0])

        link_chain(left_chain)
        link_chain(mid_chain)
        link_chain(right_chain)

        # 左/右链尾与主链首相连
        if left_chain:
            left_chain[-1].output_nodes[0].add_successor(final_node)
        if right_chain:
            right_chain[-1].output_nodes[0].add_successor(final_node)

        # 主链尾接 FinalDecision
        mid_chain[-1].output_nodes[0].add_successor(final_node)
        mid_chain[-1].output_nodes[0].add_successor(left_chain[-1].input_nodes[0])

        return g
    
    def realize_fixed_chain(self,
                        graph: CompositeGraph,
                        edge_probs: torch.Tensor) -> CompositeGraph:
        """
        按 edge_probs 在当前图上构造一条“最大概率链式结构”，
        要求以 decision_method 结尾，至少有一条边指向 decision_method。
        """
        # 复制一份图，避免就地修改原图
        edge_probs = self.edgeprob_string_to_list(edge_probs)

        _graph = deepcopy(graph)

        # 1) 清空所有 IO 节点之间的边（保留节点本身）
        for node in _graph.nodes.values():
            node.predecessors = []
            node.successors = []

        # 2) 构建邻接表：node_id -> List[(next_node_id, prob)]
        adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        nodes: Set[str] = set()

        for (src_id, dst_id), p in zip(self.potential_connections, edge_probs):
            p_val = float(p)
            if p_val <= 0.0:
                continue
            if src_id not in _graph.nodes or dst_id not in _graph.nodes:
                continue
            adj[src_id].append((dst_id, p_val))
            nodes.add(src_id)
            nodes.add(dst_id)

        final_id = _graph.decision_method.id

        best_path: Optional[List[str]] = None
        best_logp: float = -float("inf")

        def dfs(cur: str, path: List[str], logp: float, visited: Set[str]):
            nonlocal best_path, best_logp
            # 以 FinalDecision 结尾且至少有一条边（path 长度 >= 2）
            if cur == final_id and len(path) >= 2:
                if logp > best_logp:
                    best_logp = logp
                    best_path = path[:]

            for nxt, prob in adj.get(cur, []):
                if prob <= 0.0:
                    continue
                if nxt in visited:
                    continue  # 防止环
                dfs(
                    nxt,
                    path + [nxt],
                    logp + math.log(prob),
                    visited | {nxt}
                )

        # 3) 从所有非 FinalDecision 的节点尝试出发
        for start in nodes:
            if start == final_id:
                continue
            dfs(start, [start], 0.0, {start})

        # 4) 如果没有找到合法路径，则至少连一条到 FinalDecision 的最大概率边
        if best_path is None:
            best_edge = None
            best_p = 0.0
            for (src_id, dst_id), p in zip(self.potential_connections, edge_probs):
                if dst_id != final_id:
                    continue
                p_val = float(p)
                if p_val > best_p and src_id in _graph.nodes:
                    best_p = p_val
                    best_edge = (src_id, dst_id)
            if best_edge is not None:
                src = _graph.find_node(best_edge[0])
                dst = _graph.find_node(best_edge[1])
                src.add_successor(dst)
                dst.add_predecessor(src)
            # 没有任何边能连到 FinalDecision，那就直接返回“只有节点没边”的图
            return _graph

        # 5) 按 best_path 实际连边
        for u, v in zip(best_path[:-1], best_path[1:]):
            src = _graph.find_node(u)
            dst = _graph.find_node(v)
            if src is None or dst is None:
                continue
            src.add_successor(dst)
            dst.add_predecessor(src)

        return _graph
    
    def realize_fixed_tree(self,
                       graph: CompositeGraph,
                       edge_probs: torch.Tensor) -> CompositeGraph:
        """
        使用 edge_probs 构造一张“树状”的最大概率子图：
        - 每个节点最多有一个父节点（除了根）；
        - 不允许环；
        - 按边概率从大到小贪心选边；
        - 至少有一条边指向 decision_method。
        """
        edge_probs = self.edgeprob_string_to_list(edge_probs)

        _graph = deepcopy(graph)

        # 1) 清空原有边
        for node in _graph.nodes.values():
            node.predecessors = []
            node.successors = []

        final_id = _graph.decision_method.id

        # 2) 收集正概率边并按 prob 降序排序
        pos_edges: List[Tuple[str, str, float]] = []
        for (src_id, dst_id), p in zip(self.potential_connections, edge_probs):
            p_val = float(p)
            if p_val <= 0.0:
                continue
            if src_id not in _graph.nodes or dst_id not in _graph.nodes:
                continue
            pos_edges.append((src_id, dst_id, p_val))

        pos_edges.sort(key=lambda x: x[2], reverse=True)

        parent: Dict[str, str] = {}  # child -> parent

        def would_create_cycle(src_id: str, dst_id: str) -> bool:
            """
            若在当前 parent 结构下加入 src->dst 会形成环，则返回 True。
            思路：查看 src 的祖先链上是否能访问到 dst。
            """
            cur = src_id
            visited = set()
            while cur in parent:
                if cur == dst_id:
                    return True
                if cur in visited:
                    # 理论上不会出现，防御性写法
                    return True
                visited.add(cur)
                cur = parent[cur]
            return False

        selected_edges: List[Tuple[str, str, float]] = []

        # 3) 贪心加边
        for src_id, dst_id, p_val in pos_edges:
            # 限制：每个节点最多一个父
            if dst_id in parent:
                continue
            if would_create_cycle(src_id, dst_id):
                continue
            parent[dst_id] = src_id
            selected_edges.append((src_id, dst_id, p_val))

        # 4) 确保至少有一条边指向 FinalDecision
        if not any(dst_id == final_id for (_, dst_id, _) in selected_edges):
            # 在所有候选边中找一条指向 FinalDecision 的最大概率边
            best_edge = None
            best_p = 0.0
            for src_id, dst_id, p_val in pos_edges:
                if dst_id != final_id:
                    continue
                if p_val > best_p and dst_id not in parent and not would_create_cycle(src_id, dst_id):
                    best_p = p_val
                    best_edge = (src_id, dst_id, p_val)
            if best_edge is not None:
                src_id, dst_id, p_val = best_edge
                parent[dst_id] = src_id
                selected_edges.append(best_edge)

        # 5) 按 selected_edges 在图上真正加边
        for src_id, dst_id, _ in selected_edges:
            src = _graph.find_node(src_id)
            dst = _graph.find_node(dst_id)
            if src is None or dst is None:
                continue
            src.add_successor(dst)
            dst.add_predecessor(src)

        return _graph


    def parse_edgeprob_string_to_map(self,raw: str) -> Dict[Tuple[str, str], float]:
        # 先把 "['...', '...']" 变成 list[str]
        EDGE_LINE_RE = re.compile(
    r"(?P<idx>\d+): src=(?P<src>[^,]+), dst=(?P<dst>[^,]+), prob=(?P<prob>[0-9.]+)"
)
        NODE_ID_RE = re.compile(r"\(([^)]+)\)")  # 匹配括号里的短 id，例如 (5CW2)

        edge_map: Dict[Tuple[str, str], float] = {}

        for line in raw.splitlines():
            line = line.strip()
            m = EDGE_LINE_RE.match(line)
            if not m:
                continue

            src_full = m.group("src").strip()  # e.g. "DirectAnswer(5CW2)"
            dst_full = m.group("dst").strip()  # e.g. "FinalDecision(85oC)"
            prob = float(m.group("prob"))

            # 提取括号里的短 id
            src_match = NODE_ID_RE.search(src_full)
            dst_match = NODE_ID_RE.search(dst_full)
            if not src_match or not dst_match:
                continue

            src_id = src_match.group(1)
            dst_id = dst_match.group(1)  

            edge_map[(src_id, dst_id)] = prob

        return edge_map

    def edgeprob_string_to_list(self, raw: str):
        edge_map = self.parse_edgeprob_string_to_map(raw)

        probs = []
        for (src_id, dst_id) in self.potential_connections:
            prob = edge_map.get((src_id, dst_id), 0.0)
            probs.append(prob)
        return probs