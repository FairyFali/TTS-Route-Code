#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer


class Reviewer(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Review previous answer",
                 max_token: int = 4096,
                 id=None,
                 use_constraint=False,
                 use_reviewer=False):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)  # prompt_set也是提前注册好的
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        role = self.role
        constraint = self.constraint
        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
            response = [output.get('output') for output in predecessor_outputs]
            prompt = self.prompt_set.get_reflect_prompt(question=task,answer=response)
            refine_message = [Message(role="system", content="You are a reflection agent in a multi-agent system. Your primary role is to carefully review and condense the reasoning and answers produced by previous agents."),
                            Message(role="user", content=prompt)]
            refine_response = await self.llm.agen(refine_message,  max_tokens=self.max_token)
            # 用正则匹配 <REFLECTED> 中的内容
            pattern = r"<REFLECTED>\s*(.*?)\s*</REFLECTED>"
            match = re.search(pattern, refine_response, re.DOTALL)

            if match:
                format_response = match.group(1).strip()
                print('### format message,', refine_response, '\n')
            else:
                format_response = response
           
            if self.domain == 'humaneval':
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": format_response,#format_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "tests": input["tests"]
                }
            else:
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": format_response,#format_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 