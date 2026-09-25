#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import re
import random
import asyncio
import copy

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
from swarm.environment.operations.final_decision import FinalDecision

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
                 use_reviewer=False,
                 use_verifier=False):
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
        self.role_idx = 0
        self.role_name = ''
        self.behavior_fn = self.io_behavior
        self.original_fn = self.io_behavior
        if use_verifier:
            self.verifier = LLMRegistry.get('deepseek-v3')
        else:
            self.verifier = None
        # self.checker = LLMRegistry.get('llama3.2-3b-longcontext:latest')
        self.slm_rate = asyncio.Semaphore(1000000)
        self.llm_rate = asyncio.Semaphore(5)
        self.use_verifier = use_verifier
        self.use_prm = (domain in ("math", "gsm8k")) and self.use_verifier
        if self.use_prm:
            self.prm_llm = LLMRegistry.get("/home/irlab/whu_project/cjh_project/models/Qwen2.5-Math-PRM-7B")


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

    async def _execute(self,
                       inputs: List[Any] = [],
                       predecessor_outputs: List[Any] = [],
                       **kwargs):
        node_inputs = self.process_input(inputs)
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
        previous_answers = [output.get('output') for output in predecessor_outputs]
        if self.use_verifier == False:
            # previous_answers = [ans for ans in previous_answers if len(ans)<1500]
            previous_answers = [ans for ans in previous_answers]
            return await self.behavior_fn(inputs,
                                        previous_answers,
                                        )
        elif len(previous_answers) > 3:
            verified_answers = [output.get('verified_answer') for output in predecessor_outputs]
            if all(isinstance(ans, tuple) and len(ans) == 3 for ans in verified_answers):
                verified_answers.sort(key=lambda x: -x[0])
                top3 = verified_answers[:3]
                if self.domain in ('math','gsm8k'):
                    verified_answers = [f"Score:{score:.2f} Answer:{ans}" for score,reason,ans in top3 if score > 0.1 and len(ans) <1500]
                else:
                    verified_answers = [f"Score:{score} Answer: {ans}\nReason: {reason}" for score, reason, ans in top3 if len(ans) < 1500]
                if self.behavior_fn == self.review_behavior and len(verified_answers) == 0:
                    self.behavior_fn = self.io_behavior
                if self.original_fn == self.review_behavior and len(verified_answers) > 0:
                    self.behavior_fn = self.review_behavior
            else:
                print("Invalid verified_answers format. Falling back to random selection.")
                verified_answers = random.sample(previous_answers, k=3)
                verified_answers = [ans for ans in verified_answers if len(ans) < 1500]
            print('### Log, verified answers,', verified_answers, '\n')
            return await self.behavior_fn(inputs,
                                      verified_answers,
                                      )
        else:
            if self.domain in ('math','gsm8k'):
                complete_pre = []
                previous_score = [output.get('verified_answer')[0] for output in predecessor_outputs]
                for i in range(len(previous_score)):
                    answer = (previous_score[i],previous_answers[i])
                    complete_pre.append(answer)
                previous_answers = [f"Score:{score:.2f} Answer:{ans}" for score , ans in complete_pre if len(ans) < 1500 and score > 0.1]
                return await self.behavior_fn(inputs,
                                        previous_answers,
                                        )
            previous_answers = [ans for ans in previous_answers if len(ans) < 1500]
            return await self.behavior_fn(inputs,
                                        previous_answers,
                                        )
    
    async def io_behavior(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        has_predecessor = True if len(predecessor_outputs) > 0 else False
        # print(self.use_reviewer)
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
            previous_answers = predecessor_outputs
            if self.use_constraint:
                try:
                    role, constraint = await self.node_optimize(input, meta_optmize=False)
                except Exception as e:
                    print('❌ node_optimize 出错了:', e)
                    exit()
            else:
                role, constraint = await self.node_optimize(input, meta_optmize=False)
                if previous_answers: # 存在前置节点
                    if self.domain == 'math':
                        constraint = 'Make sure to put the answer (and only answer) inside \\boxed{}.'
                    else:
                        constraint = ""
                else: # 没有前置节点
                    if self.domain == 'humaneval':
                        constraint = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
                    if self.domain == 'math' or self.domain == 'gsm8k' and self.use_constraint == False:
                        constraint = """You must provide the separator 'The answer is: ' before your final answer.
            # Make sure to put the answer (and only answer) inside \\boxed{}."""
                    else:
                        constraint = constraint
            if previous_answers:
                prompt = self.prompt_set.get_answer_prompt_refine_last_answers(task, previous_answers)
            else:
                prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [Message(role="system", content=f"You are {role}. {constraint}"),
                       Message(role="user", content=prompt)]
            # print('### Log, message,', message, '\n')
            async with self.slm_rate:
                response, prompt_tokens, completion_tokens, price = await self.llm.agen(message, max_tokens=self.max_token)
            """summary_prompt = self.prompt_set.get_summary(response,task)
            summary_message = [Message(role='system', content="You are a summery agent in a multi-agent system. Your job is to clean and summarize the previous output."),
                               Message(role='user', content=summary_prompt)]
            async with self.slm_rate:
                summary_response = await self.checker.agen(summary_message, max_tokens=self.max_token)
            pattern = r"<OUTPUT>\s*(.*?)\s*</OUTPUT>"
            match = re.search(pattern, summary_response, re.DOTALL)

            if match:
                summary_response = match.group(1).strip()
            else:
                summary_response = summary_response"""
            if self.use_verifier == False or self.use_constraint == False: # self.use_constraint == False
                verified_answer = '' #not use verifier ()
            elif self.use_prm:
                async with self.llm_rate:
                    verified_answer = await self.verifier_eval(response,task)
            else:
                if any(isinstance(succ, FinalDecision) or 
                    (len(succ.predecessors) > 3) for succ in self.successors):
                    async with self.llm_rate:
                        verified_answer = await self.verifier_eval(response,task)
                else:
                    verified_answer = ''#use big model as verifier

            if self.domain == 'humaneval' or self.domain == 'livecodebench':
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "tests": input["tests"],
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens,price)
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
                    "output": response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens,price)
                }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 

    async def review_behavior(self, inputs: List[Any] = [], predecessor_outputs: List[Any] = [], **kwargs):
        role = self.role
        constraint = self.constraint
        node_inputs = self.process_input(inputs)
        outputs = []
        for input in node_inputs:  # 其实只有一个
            task = input["task"]
            response = predecessor_outputs
            prompt = self.prompt_set.get_reflect_prompt(question=task,answer_list=response)
            refine_message = [Message(role="system", content="You are a fusion agent in a multi-agent system. Your primary role is to carefully review , condense and synthesize the reasoning and answers produced by previous agents."),
                            Message(role="user", content=prompt)]
            async with self.slm_rate:
                refine_response, prompt_tokens, completion_tokens, price = await self.llm.agen(refine_message,  max_tokens=self.max_token)
            format_response = refine_response
            """summary_prompt = self.prompt_set.get_summary(format_response,task)
            summary_message = [Message(role='system', content="You are a summery agent in a multi-agent system. Your job is to clean and summarize the previous output."),
                               Message(role='user', content=summary_prompt)]
            async with self.slm_rate:
                summary_response = await self.checker.agen(summary_message, max_tokens=self.max_token)
            match1 = re.search(pattern, summary_response, re.DOTALL)

            if match1:
                summary_response = match1.group(1).strip()
            else:
                summary_response = summary_response"""
            """if any(isinstance(succ, FinalDecision) or 
                (len(succ.predecessors) > 3) for succ in self.successors):
                async with self.llm_rate:
                    verified_answer = await self.verifier_eval(summary_response,task)
            else:
                verified_answer = ''"""

            if self.use_verifier == False or self.use_constraint == False: # self.use_constraint == False
                verified_answer = '' #not use verifier ()
            elif self.use_prm:
                async with self.llm_rate:
                    verified_answer = await self.verifier_eval(response,task)
            else:
                if any(isinstance(succ, FinalDecision) or 
                    (len(succ.predecessors) > 3) for succ in self.successors):
                    async with self.llm_rate:
                        verified_answer = await self.verifier_eval(response,task)
                else:
                    verified_answer = ''#use big model as verifier
           
            if self.domain == 'humaneval' or self.domain == "livecodebench":
                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": format_response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "tests": input["tests"],
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens,price)
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
                    "output": format_response,#summary_response
                    "ground_truth": input.get("GT", []),
                    "format": "natural language",
                    "verified_answer": verified_answer,
                    "cost": (prompt_tokens,completion_tokens,price)
                }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 
    
    async def verifier_eval(self,
                              answer,
                              task
                              ):
        """
        调用 LLM，筛选出前 3 个最优答案并返回。
        """

        if self.domain in ('math', 'gsm8k'):
            score = self.prm_score_response(answer,task)
            reason = ""
            verified_answer = (score,reason,answer)
            return verified_answer

        prompt = self.prompt_set.get_verifier(answer, task)
        msgs = [
                Message(role="system", content=(
                    "You are a verifier agent in a multi-agent system. "
                    "Your job is to score an agent's answer and briefly explain the score by correctness and reasoning quality."
                )),
                Message(role="user", content=prompt)
            ]
        raw, prompt_tokens, completion_tokens = await self.verifier.agen(msgs)
        score_match = re.search(r"Score:\s*([0-9](?:\.?[0-9])?)", raw)
        reason_match = re.search(r"Reason:\s*(.+)", raw, re.DOTALL)
        if score_match and reason_match:
                score = float(score_match.group(1))
                reason = reason_match.group(1).strip()
                verified_answer = (score,reason,answer)
                return verified_answer
        else:
                print(f"Failed to parse score and reason from response: {raw}")
                return answer
        
    def prm_score_response(self,
    response: str,
    question: str,
    model_path: str = "/home/irlab/whu_project/cjh_project/models/Qwen2.5-Math-PRM-7B",
    device: str = "auto",
) -> float:
        
        def make_step_rewards(logits, token_masks):
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
            
            all_scores_res = []
            for i in range(probabilities.size(0)):
                sample = probabilities[i] # seq_len, num_labels
                positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
                non_zero_elements_list = positive_probs.cpu().tolist()
                all_scores_res.append(non_zero_elements_list)
            return all_scores_res
        
        tokenizer = self.prm_llm.tokenizer
        model = self.prm_llm.model
        # ---- 构造 PRM 会话 ----
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()
        system_prompt: str = f"You are {role}. {constraint}"
        answer = [response]
        messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
    {"role": "assistant", "content": "<extra_0>".join(answer) + "<extra_0>"},
]
        conversation_str = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)
        with torch.inference_mode():
            input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
            outputs = model(input_ids=input_ids)
        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        score = make_step_rewards(outputs[0], token_masks)
        return score[0][0]
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k in ("prm_llm", "llm", "checker", "verifier"):
                setattr(new, k, v)  # 复用同一实例
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new
            
        