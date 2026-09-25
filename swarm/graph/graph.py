#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import shortuuid
from typing import Any, List, Optional, Dict
from copy import deepcopy
from abc import ABC, abstractmethod
import async_timeout
import numpy as np
import random

from swarm.graph.visualize import GPTSwarmVis
from swarm.memory import GlobalMemory
from swarm.graph.node import Node


class Graph(ABC):
    """
    A framework for managing and executing a network of interconnected nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    Attributes:
        model (LLM): An instance of a language model used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.
        memory (Memory): A memory management system to store and retrieve data related to the graph.
        role (str): The role of the graph, defining its purpose within a larger system.
        constraint (str): Operational constraints that the graph adheres to.
        format (str): The format of responses or data processed by the graph.
        system_content (str): A formatted string that combines role, constraint, and format.
        is_aggregate (bool): Flag indicating whether the graph aggregates data from nodes.
        input_nodes (list): List of nodes designated as input points to the graph.
        output_node (Node): The node designated as the primary output point of the graph.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        display(draw=True): Displays a textual representation of the graph, with an option for a visual representation.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                model_name: Optional[str] = None,
                meta_prompt: bool = False,
                index: int = -1,
                max_tokens=4096,
                use_verifier = False
                ):

        self.id = shortuuid.ShortUUID().random(length=4)
        self.index = index
        self.max_tokens = max_tokens
        self.domain = domain
        self.model_name = model_name
        self.meta_prompt = meta_prompt
        self.nodes = {}
        self.memory = GlobalMemory.instance()
        self.is_aggregate = False
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.use_verifier = use_verifier
        self.build_graph()

    @property
    def adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    @abstractmethod
    def build_graph(self):
        """ To be overriden bu a descendant class """

    def add_node(self, node: Node):
        """
        Creates and adds a new node to the graph.
        If id is not provided, generates a unique id for the node.
        """
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=5)
        node.id = node_id

        self.nodes[node_id] = node
        return node   

    def display(self, draw=True, file_name=None):
        """
        Prints a simple textual representation of the graph.
        """
        # for node in self.nodes.values():
        #     print(f"Node ID: {node.id}, Type: {type(node).__name__}, "
        #           f"Predecessors: {[n.id for n in node.predecessors]}, "
        #           #f"Successors: {[n.id for n in node.successors]}"
        #           )
        if draw:
            # GPTSwarmVis(self, file_name=file_name)
            try:
                GPTSwarmVis(self, file_name=file_name)
            except:
                print('GPTSwarmVis failed')

    async def run(self, inputs: Dict[str, Any], 
                  max_tries: int = 3, 
                  max_time: int = 600, 
                  return_all_outputs: bool = False) -> List[Any]:
        
        total_cost = []
 
        def is_node_useful(node):
            if node in self.output_nodes:
                return True
            
            for successor in node.successors:
                if is_node_useful(successor):
                    return True
            return False
        
        if hasattr(self, "decision_method") and self.decision_method is not None:
            if not getattr(self.decision_method, "predecessors", None):
                pre = random.choice(self.graphs).output_nodes[0]
                self.decision_method.predecessors = [pre]
                pre.successors = [self.decision_method]
        
        useful_node_ids = [node_id for node_id, node in self.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(self.nodes[node_id].predecessors) for node_id in useful_node_ids}
        zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and node_id in useful_node_ids]

        for i, input_node in enumerate(self.input_nodes):
            node_input = deepcopy(inputs)
            input_node.inputs = [node_input]

        task = inputs['task']
        answer_score = []
        memory = []

        while zero_in_degree_queue:
            # --- 4.1 取出当前层所有节点 ---
            current_level = zero_in_degree_queue.copy()
            zero_in_degree_queue.clear()
            for nid in current_level:
                current_node = self.nodes[nid]
                tries = 0
                while tries < max_tries:
                    try:
                        # 这一步会更新node的outputs
                        await asyncio.wait_for(self.nodes[nid].execute(), timeout=max_time)
                        break
                    except asyncio.TimeoutError:
                        print(f"Node {nid} execution timed out, retrying {tries + 1} out of {max_tries}...")
                    except Exception as e:
                        print(f"Error during execution of node {nid}: {e}")
                        break
                    tries += 1

                answer = None
                if self.nodes[nid].outputs:
                    output_messages = self.nodes[nid].outputs
                    if len(output_messages) > 0:
                        answer = output_messages[-1].get("output", output_messages[-1])
                        verified = output_messages[-1].get("verified_answer", output_messages[-1])
                        cost = output_messages[-1].get("cost", output_messages[-1])
                        total_cost.append(cost)
                        # anwer_with_source_dst = (answer,self.nodes[nid].id,[node.id for node in self.nodes[nid].predecessors])
                        memory.append(answer)
                        if isinstance(verified, tuple) and len(verified) == 3:
                            score, _, _ = verified
                            answer_score.append((score, answer))
            
            for nid in current_level:
                for succ in self.nodes[nid].successors:
                    if succ.id in useful_node_ids:
                        in_degree[succ.id] -= 1
                        if in_degree[succ.id] == 0:
                            zero_in_degree_queue.append(succ.id)

        final_answers = []
        if answer_score and self.domain in ('math','gsm8k'):
           best_answer = max(answer_score, key=lambda t: t[0])[1]
           memory.append(best_answer)
           final_answers.append(best_answer)
           self.memory = memory
        else:
            for output_node in self.output_nodes:
                outputs = output_node.outputs
            ans = outputs[-1].get("output", outputs[-1])
            cost = outputs[-1].get("cost", output_messages[-1])
            total_cost.append(cost)
            memory.append(ans)
            final_answers.append(ans)
            self.memory = memory
            if cost:
                price = 0
                for node_cost in total_cost:
                    price += node_cost[-1]
                return final_answers,price

        """for output_node in self.output_nodes:
            outputs = output_node.outputs
            from swarm.environment.agents.io import IO
            if isinstance(self,IO):
                final_answers.append(answer)
            else:
                if outputs and not return_all_outputs:
                    ans = outputs[-1].get("output", outputs[-1])
                    final_answers.append(ans)
                else:
                    tier_answers = []
                    for msg in outputs:
                        ans = msg.get("output", msg)
                        tier_answers.append(ans)
                        final_answers.append(ans)"""

        if len(final_answers) == 0:
            final_answers.append("No answer since there are no inputs provided")
        return final_answers

    def find_node(self, id: str):
        for node in self.nodes.values():
            if node.id == id:
                return node
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
