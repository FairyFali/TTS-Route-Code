#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Any, Optional
import random
import re
import asyncio

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry, PromptSet
from swarm.llm import LLMRegistry, LLM
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.operations.operation_registry import OperationRegistry
from swarm.environment.tools.coding.python_executor import PyExecutor

random.seed(0)

class MergingStrategy(Enum):
    OutputsAsReferences = 0
    MajorityVote = 1
    RandomChoice = 2
    SelfConsistency = 3
    SelectBest = 5
    Verifier = 6


@OperationRegistry.register("FinalDecision")
class FinalDecision(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],  # 没用上
                 strategy: MergingStrategy,
                 operation_description: str = "Refer to all answers and give a final answer.", 
                 id=None,
                 use_verifier=False):
        super().__init__(operation_description, id, True)
        self.strategy: MergingStrategy = strategy  # 比如选择最好的
        self.domain: str = domain
        self.llm: LLM = LLMRegistry.get(model_name)
        self.prompt_set: PromptSet = PromptSetRegistry.get(domain)
        self.role: str = self.prompt_set.get_role()
        self.constraint: str = self.prompt_set.get_constraint()
        self.use_verifier=use_verifier
        self.model_name = model_name

    @property
    def node_name(self):
        return self.__class__.__name__

    def meta_prompt(self, node_inputs, meta_init=False):

        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        self.materials = defaultdict(str)

        for input in node_inputs:
            operation = input.get('operation')
            if operation != "FileAnalyse":
                # 我们仅仅用这一行，将前面的所有节点的output拿过来
                self.materials[operation] += f'{input.get("output", "")}\n'
            else:
                self.materials["files"] = input.get("files") 
            self.materials["task"] = input.get('task') 

        question = self.prompt_set.get_combine_materials(self.materials)
        prompt = self.prompt_set.get_answer_prompt(question=question)    

        if meta_init:
            pass #TODO

        return role, constraint, prompt

    async def _execute(self, inputs: List[Any] = [], 
                       **kwargs) -> None:

        node_inputs = self.process_input(inputs)  # 将前置节点转换为list，如果是not none list，就什么都不做
        prompt = None
        response = None
        # 选择策略包括
        #   output as references: 基于之前的输出再输出一个新的版本
        #   majority vote: 投票
        #   random choice
        #   SelfConsistency
        #   SelectBest，默认
        if self.strategy == MergingStrategy.OutputsAsReferences:

            role, constraint, prompt = self.meta_prompt(node_inputs)
            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                    Message(role="user", content=prompt)]
        
            response = await self.llm.agen(message)

        elif self.strategy == MergingStrategy.MajorityVote:
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            answers = [input.get("output") for input in inputs]
            counter = Counter(answers)
            sorted_counter = counter.most_common()
            max_freq = sorted_counter[0][1]
            equally_frequent_answers = [ans for ans, freq in sorted_counter if freq == max_freq]
            response = random.choice(equally_frequent_answers)
            print(f"{answers=} {response=}")
            
        elif self.strategy == MergingStrategy.RandomChoice:
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for RandomChoice")
            answers = [input.get("output") for input in inputs]
            response = random.choice(answers)
            print(f"{answers=} {response=}")

        elif self.strategy == MergingStrategy.SelfConsistency:  
            # This is different from MajorityVote because it is prompt-based.
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            constraint = self.prompt_set.get_constraint()
            prompt = self.prompt_set.get_self_consistency(question=question, answers=answers, constraint=constraint)
            message = [Message(role="system", content=f"You are a {self.role}. {self.constraint}"),
                    Message(role="user", content=prompt)]
            response = await self.llm.agen(message)
            print(f"{answers=} {response=}")

        elif self.strategy == MergingStrategy.SelectBest:  
            # This is different from MajorityVote because it is prompt-based.
            if len(inputs) == 0:
                prompt = self.prompt_set.get_answer_prompt()
                raise Exception("No inputs is not supported for MajorityVote")
            
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            verified_answers = [input.get("verified_answer") for input in inputs]
            if self.use_verifier == False:
                # answers = [ans for ans in answers if len(ans)<1500]
                answers = [ans for ans in answers]
                prompt = self.prompt_set.get_select_best(question=question, solutions=answers)
            elif len(answers) > 3:
                if all(isinstance(ans, tuple) and len(ans) == 3 for ans in verified_answers):
                    verified_answers.sort(key=lambda x: -x[0])
                    top4 = verified_answers[:3]
                    verified_answers = [f"Answer: {ans}\nReason: {reason}" for score, reason, ans in top4]
                    verified_answers = [ans for ans in verified_answers if len(ans) < 1500]
                    prompt = self.prompt_set.get_select_best(question=question, solutions=verified_answers)
                else:
                    print("Invalid verified_answers format. Falling back to random selection.")
                    # verified_answers = random.sample(answers, k=4)
                    verified_answers = [ans for ans in answers if len(ans) < 1500] # [ans for ans in verified_answers if len(ans) < 1500]
                    prompt = self.prompt_set.get_select_best(question=question, solutions=verified_answers)
            else:
                # answers = [ans for ans in answers if len(ans) < 1500]
                answers = [ans for ans in answers]
                prompt = self.prompt_set.get_select_best(question=question, solutions=answers)
            message = [Message(role="system", content=f"You are a {self.role}. {self.constraint}"),
                    Message(role="user", content=prompt)]
            response,prompt_tokens,completion_tokens,price = await self.llm.agen(message)
            print(f"{len(answers)=}")

        elif self.strategy == MergingStrategy.Verifier:
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            verified_answers = [input.get("verified_answer") for input in inputs]
            valid_verified = [va for va in verified_answers if isinstance(va, tuple) and len(va) == 3]
    
            if not valid_verified:
                raise Exception("No valid verified answers found.")
            valid_verified.sort(key=lambda x: -x[0])
            top1 = valid_verified[0]
            response = top1[-1]


        else:
            logger.error(f"Error: does not support \"{self.strategy}\"!")

        executions = {"operation": self.node_name,
                            "task": inputs[0]["task"], 
                            "files": inputs[0]["files"],
                            "input": inputs, 
                            "subtask": prompt,
                            "output": response,
                            "format": "natural language",
                            "cost": (prompt_tokens,completion_tokens,price)}

        self.memory.add(self.id, executions)
        self.log()
        return executions
        
