#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import review
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('reviewer')
class Reviewer(Graph):
    def build_graph(self):
        io = review(self.domain, self.model_name, max_token=self.max_tokens) 
        self.add_node(io)
        self.input_nodes = [io]
        self.output_nodes = [io]
