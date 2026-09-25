#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import DirectAnswer
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('IO')
class IO(Graph):
    def build_graph(self):
        io = DirectAnswer(self.domain, self.model_name, max_token=self.max_tokens, use_constraint=True, use_reviewer=True, use_verifier=self.use_verifier)  # IO就相当于直接回答，一个subgraph里面只有一个节点
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]
