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


class DirectAnswer(Node):
    '''
    直接回答是Node的一种类型
    '''
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
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
        self.use_constraint = use_constraint
        self.use_reviewer = use_reviewer



    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role
            node_optmizer = MetaPromptOptimizer(self.domain, self.model_name, self.node_name)
            update_constraint = await node_optmizer.generate(init_prompt=task, init_role=role, init_constraint=constraint, tests=input)    #输入问题，role，constraint，input（测试样例）
            print('update_constraint',update_constraint)
            return update_role, update_constraint

        return role, constraint


    async def _execute(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []
        has_predecessor = True if len(predecessor_outputs) > 0 else False
        print(self.use_reviewer)
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
            previous_answers = [output.get('output') for output in predecessor_outputs]
            if self.use_constraint:
                try:
                    role, constraint = await self.node_optimize(input, meta_optmize=False)
                except Exception as e:
                    print('❌ node_optimize 出错了:', e)
                    exit()
            else:
                role, constraint = await self.node_optimize(input, meta_optmize=False)
                if has_predecessor: # 存在前置节点
                    constraint = ""
                else: # 没有前置节点
                    if self.domain == 'humaneval':
                        constraint = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
                    else:
                        constraint = "Given the question, solve it step by step."
            if has_predecessor:
                prompt = self.prompt_set.get_answer_prompt_refine_last_answers(task, previous_answers)
            else:
                prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [Message(role="system", content=f"You are {role}. {constraint}"),
                       Message(role="user", content=prompt)]
            # print('### Log, message,', message, '\n')
            response = await self.llm.agen(message, max_tokens=self.max_token)
            if self.use_reviewer:
                refine_prompt = self.prompt_set.get_reflect_prompt(question=task,answer=response)
                refine_message = [Message(role="system", content="You are a reflection agent in a multi-agent system. Your primary role is to carefully review and condense the reasoning and answers produced by previous agents."),
                                Message(role="user", content=refine_prompt)]
                refine_response = await self.llm.agen(refine_message,  max_tokens=self.max_token)
                # 用正则匹配 <REFLECTED> 中的内容
                pattern = r"<REFLECTED>\s*(.*?)\s*</REFLECTED>"
                match = re.search(pattern, refine_response, re.DOTALL)

                if match:
                    format_response = match.group(1).strip()
                    print('### format message,', message, '\n')
                else:
                    format_response = response
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