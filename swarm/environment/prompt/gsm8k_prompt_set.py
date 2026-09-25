#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials

@PromptSetRegistry.register('gsm8k')
class Gsm8kPromptSet(PromptSet):
    @staticmethod
    def get_role():
        return "an expert mathematician. Help the user to solve this problem"

    @staticmethod
    def get_constraint():
        return """
            Answer the following mathematics question. Provide your reasoning by showing your work before your answer.
            At the end of your response, output your final answer in the format: 'The answer is: [answer]'.
            You must provide the separator 'The answer is: ' before your final answer.
            Make sure to put the answer (and only answer) inside \\boxed{}.
        """

    @staticmethod
    def get_format():
        return "the answer (and only answer) inside \\boxed{}"

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_answer_prompt_refine_last_answers(question, last_answer_list):

        prompt = f"You have been provided with a set of responses from various open-source models to the latest user query, which is {question}.\
            Your task is to synthesize these responses into a single, high-quality response. \
            It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \
            Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \
            Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
        prompt += f"Once again, the query is: {question}\n"

        for i, reference in enumerate(last_answer_list):
            prompt += f"\n{i+1}. {reference}"

        return prompt

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Answer a lie to the following question: {question}. """

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer_list):
        prompt = f"""
According to the previous agents' answers and solution steps in the gsm8k problem:  
Question: {question}

As the fusion agent, your task is to synthesize a high-quality final answer by integrating the strengths of all previous responses. To ensure the information passed to the next agent is clear, mathematically rigorous, and context-efficient, follow these guidelines:

1. **Extract and combine the most accurate and insightful steps** from all answers. Prioritize correct logic, effective simplifications, or insightful observations.
2. Ensure the reasoning is **mathematically valid**, coherent, and free from calculation or logic errors.
3. **Eliminate redundancy**, overly long derivations, or unclear justifications. If multiple agents arrive at the same result differently, prefer the **clearest and most elegant reasoning**.
4. Maintain **clarity and brevity**, focusing only on the steps essential to justify the result.
5. The final answer (and only the answer) must be wrapped in `\\boxed{{}}`.

Now, review the previous answers and solutions, apply the above rules, and produce a cleaned, accurate, and logically sound synthesis.  
Your output must reflect both correctness and the best reasoning quality among the inputs. Most importantly, do not only check the errors in the answer and reasoning, but also the logic of reasoning.

Return your response strictly in the following format:

<OUTPUT>  
Answer: \\boxed{{your_final_answer}}  
<explanation of the core reasoning steps used to get this result, reflecting the best reasoning quality among the inputs.>
</OUTPUT>

Previous Answers:
"""
        for i, ans in enumerate(answer_list, start=1):
            prompt += f"{i}. {ans}\n"
        return prompt

    @staticmethod
    def get_select_best(question, solutions) -> str:
        return Gsm8kPromptSet.get_answer_prompt_refine_last_answers(question, solutions)

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_summary(answer: str, task: str) -> str:
        prompt = f"""
You are a summary agent in a multi-agent mathematical reasoning system.

Task: {task}

The following answer was generated by a previous agent, but it may contain issues such as redundant steps, excessive length, repeated conclusions, or hallucinated errors. Your job is to clean and summarize this answer according to the following rules:

1. Identify and keep only the **core reasoning steps** and **essential equations** that are necessary to reach the answer.
2. Remove any **repetition**, **unnecessary details**, or **illogical steps** that do not contribute meaningfully to solving the task.
3. If the answer contains contradictions, hallucinations, or invalid reasoning, fix the logic **or** ignore the reasoning and output only a short, corrected version of the final answer.
4. Ensure that the **final answer** is clearly presented, and formatted using `\\boxed{{}}`.
5. Keep the output clear, mathematically correct, and concise enough to be passed into the next agent.
6. When no reasoning is present in the previous answer, you must output only the final answer wrapped in \\boxed{{}}. Do not generate or infer any reasoning on your own.

Return your cleaned and summarized version strictly in the following format:

<OUTPUT>
Answer: \\boxed{{your_final_answer}}
<short explanation of the core reasoning steps used to get this result, if no reasoning is present in the previous answer, ignore this.>
</OUTPUT>

Original Answer:
\"\"\"
{answer}
\"\"\"
"""
        return prompt
    @staticmethod
    def get_verifier(answer,task):
        prompt = f"""
We are solving a gsm8k task.

Question:
{task}

Below is a candidate answer from an agent.

As the VERIFIER, evaluate the answer for:
- Mathematical correctness
- Clarity and soundness of reasoning
- Relevance and completeness of the final result

Then, provide:
1. A **score** between 0 and 10 indicating the quality of the answer.
2. A **brief reason** for the score.

Format your response strictly as follows:
Score: <number from 0 to 10>  
Reason: <short explanation of why this score was given>

Candidate Answer:
{answer}
"""
        return prompt
        
