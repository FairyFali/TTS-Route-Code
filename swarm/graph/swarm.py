#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any
import asyncio
import shortuuid
import numpy as np
import torch
import copy

from swarm.environment.operations.final_decision import FinalDecision, MergingStrategy
from swarm.optimizer.edge_optimizer.parameterization import EdgeWiseDistribution
from swarm.memory import GlobalMemory
from swarm.graph.composite_graph import CompositeGraph
from swarm.utils.log import logger
from swarm.environment.agents import AgentRegistry
from swarm.environment.operations.operation_registry import OperationRegistry


class Swarm:
    """
    A class representing a swarm in the GPTSwarm framework.

    Attributes:
    """

    def __init__(self, 
                 agent_names: List[str], # 比如IO
                 domain: str, # No default, we want the user to be aware of what domain they select.
                 model_names: Optional[list] = None, # None is mapped to "gpt-4-1106-preview".
                 open_graph_as_html: bool = False,
                 final_node_class: str = "FinalDecision",
                 final_node_kwargs: Dict[str, Any] = {'strategy': MergingStrategy.OutputsAsReferences},
                 edge_optimize: bool = False,
                 node_optimize: bool = False,
                 role_optimize: bool = False,
                 init_connection_probability: float = 0.5,
                 connect_output_nodes_to_final_node: bool = False,
                 include_inner_agent_connections: bool = True,
                 max_tokens = 4096,
                 models_cost: dict = None,
                 use_verifier: bool = False
                 ):
        
        self.id = shortuuid.ShortUUID().random(length=4)    
        self.agent_names = agent_names
        self.domain = domain
        self.model_names = model_names
        self.open_graph_as_html = open_graph_as_html
        self.memory = GlobalMemory.instance()
        self.final_node_class = final_node_class  # 使用默认
        self.final_node_kwargs = final_node_kwargs  # 默认输出方式是提供参考
        self.edge_optimize = edge_optimize  # 默认false
        self.node_optimize = node_optimize  # 默认false
        self.role_optimize = role_optimize
        self.init_connection_probability = init_connection_probability
        self.connect_output_nodes_to_final_node = connect_output_nodes_to_final_node
        self.use_verifier = use_verifier
        self.max_tokens = max_tokens
        self.models_cost = models_cost
        # 调用这个函数是初始化的关键
        self.organize(include_inner_agent_connections)

    def organize(self, include_inner_agent_connections: bool = True):

        self.used_agents = []
        decision_model = self.model_names[-1]  # 最后一个模型用于决策
        decision_method = OperationRegistry.get(self.final_node_class, self.domain, decision_model, **self.final_node_kwargs)
        self.composite_graph = CompositeGraph(decision_method, self.domain, self.model_names)
        potential_connections = []
        index = 0
        for model_name, agent_name in zip(self.model_names[:-1], self.agent_names):
            if agent_name in AgentRegistry.registry:
                agent_instance = AgentRegistry.get(agent_name, self.domain, model_name, index=index, max_tokens= self.max_tokens, use_verifier=self.use_verifier) #对于每一个agent，实例化一个graph类，实际上只有一个节点在我们的设置里面
                if not include_inner_agent_connections:
                    for node in agent_instance.nodes:
                        for successor in agent_instance.nodes[node].successors:
                            potential_connections.append((node, successor.id))
                        agent_instance.nodes[node].successors = []
                self.composite_graph.add_graph(agent_instance)  # 这里会把加入的新agent当做input node，input node有多个
                self.used_agents.append(agent_instance)
            else:
                logger.error(f"Cannot find {agent_name} in the list of registered agents "
                             f"({list(AgentRegistry.keys())})")
            index += 1
        
        potential_connections = []
        if self.edge_optimize:  
            # Add bi-directional connections between all nodes of all agents (except for the decision nodes).
            for agent1 in self.used_agents:  # 注意 used_agents不包括decision_making node
                for agent2 in self.used_agents:
                    if agent1 != agent2:
                        for node1 in agent1.nodes:
                            for node2 in agent2.nodes:
                                potential_connections.append((node1, node2)) # (from, to) 连接所有nodes，双向连接

            # Add only forward connections from all agents' nodes to the final decision node.
            for agent in self.used_agents:
                for node in agent.nodes:
                    if (self.connect_output_nodes_to_final_node and
                            node in [output_node.id for output_node in agent.output_nodes]):
                        agent.nodes[node].add_successor(decision_method)
                    else:
                        potential_connections.append((node, decision_method.id)) # (from, to)
                        
        else:
            # Connect all output nodes to the decision method if edge optimization is not enabled
            # 如果不使用edge optimazation 方法，则所有的sub graph的输出节点都链向decision 节点
            for agent in self.used_agents:
                for node in agent.nodes:
                    if node in [output_node.id for output_node in agent.output_nodes]:
                        agent.nodes[node].add_successor(decision_method)
        # 边的分布
        self.connection_dist = EdgeWiseDistribution(potential_connections, self.init_connection_probability, self.models_cost)
        self.potential_connections = potential_connections

    def visualize_adj_matrix_distribution(self, logits):
        probs = torch.sigmoid(logits)
        matrix = np.zeros((self.composite_graph.num_nodes, self.composite_graph.num_nodes))
        num_nodes_per_agent = np.array([len(agent.nodes) for agent in self.used_agents])
        for i in range(len(num_nodes_per_agent)):
            matrix[num_nodes_per_agent[:i].sum():num_nodes_per_agent[:i+1].sum(), num_nodes_per_agent[:i].sum():num_nodes_per_agent[:i+1].sum()] \
                = self.used_agents[i].adj_matrix
        
        probs_idx = 0
        for i in range(len(self.used_agents)):
            for j in range(len(self.used_agents)):
                if i != j:
                    for k in range(num_nodes_per_agent[i]):
                        for l in range(num_nodes_per_agent[j]):
                            matrix[k + num_nodes_per_agent[:i].sum(), l + num_nodes_per_agent[:j].sum()] = probs[probs_idx]
                            probs_idx += 1

        node_idx = 0
        for agent in self.used_agents:
            for node in agent.nodes:
                if node in [output_node.id for output_node in agent.output_nodes] and self.connect_output_nodes_to_final_node:
                    matrix[node_idx, -1] = 1
                else:
                    matrix[node_idx, -1] = probs[probs_idx]
                    probs_idx += 1
                node_idx += 1

        return matrix

    def run(self,
            inputs: Dict[str, Any],
            realized_graph: Optional[CompositeGraph] = None,
            display: bool = False,  # swarm对象有一个display函数
            ):

        if realized_graph is None:
            _graph, _ = self.connection_dist.realize(self.composite_graph)
        else:
            _graph = copy.deepcopy(realized_graph)

        if display:
            _graph.display(draw=self.open_graph_as_html)
        # final_answer is a list， 但是因为输出节点是final_decision，专门设计为只输出一个
        final_answer = asyncio.run(_graph.run(inputs))

        return final_answer

    async def arun(self,
             inputs: Dict[str, Any],
             realized_graph: Optional[CompositeGraph] = None,
             ):

        if realized_graph is None:
            _graph, _ = self.connection_dist.realize(self.composite_graph)
        else:
            _graph = copy.deepcopy(realized_graph)

        _graph.display(draw=self.open_graph_as_html)

        final_answer = await _graph.run(inputs)

        return final_answer
    