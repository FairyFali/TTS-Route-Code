#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials


@PromptSetRegistry.register('livecodebench')
class LiveCodeBenchPromptSet(PromptSet):

    @staticmethod
    def get_role():
        # 更贴近 LiveCodeBench：竞赛 / 在线评测题
        return "an AI that only responds with Python code to solve programming problems"

    @staticmethod
    def get_constraint():
        return (
" You will be given a competitive programming style problem statement"
" and optionally some starter code. "
" Write a full Python solution that solves the problem. "
" Your solution should be a complete, runnable Python program (or a function if the prompt explicitly asks so). "
" Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
" Provide your reasoning by showing your work after your answer."
" At the head of your response, output your final answer in the format: 'Answer: [answer]'. Make sure there is only Python code block in the final answer"
)
    @staticmethod
    def get_format():
        return "python code"

    @staticmethod
    def get_answer_prompt(question):
        # question 就是 LiveCodeBenchDataset 里构造好的 task（题面 + starter code 提示）
        return f"{question}"

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the following LiveCodeBench question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:
{feedback}

Rewrite the code based on the feedback and the original question.
Return ONLY a single Python code block with the corrected full solution.
"""

    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Code Problem Resolution\n\n"
"Evaluate if additional information is needed to solve the coding problem. "
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Problem:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify critical constraints, input/output formats, edge cases, and algorithmic hints that may be necessary to design a correct and efficient solution.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
"# File Analysis Task\n\n"
f"## 🔍 Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## 📄 File Under Analysis:\n---\n{file}\n---\n\n"
"## 📝 Instructions:\n"
"1. Identify the key sections in the file relevant to the query (e.g., input format, constraints, helper functions, existing code structure).\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify how the input is parsed and what the expected output format is.'"
        )

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Coding Problem: \n---\n{question}\n---\n\n"
            f"## 🔍 Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## 🌐 Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the algorithmic aspects of the problem. "
            "Each query should focus on key terms such as the problem type (e.g., 'shortest path', 'segment tree', 'DP on trees'), constraints, or known patterns.\n"
            "Format the output as a comma-separated list.\n"
            "For example: 'two pointers substring problem, sliding window distinct characters, codeforces abc1873 A solution idea'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )

    @staticmethod
    def get_adversarial_answer_prompt(question):
        # LiveCodeBench 一般不需要 adversarial 模式，可以留空或简单返回
        return f"Write an incorrect Python solution to the following coding problem (intentionally buggy):\n{question}"

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
"# Summarization of Search Results\n\n"
f"## Original coding problem: \n---\n{question}\n---\n\n"
f"## 🔍 Required Information for Summary:\n---\n{query}\n---\n\n"
f"## 🌐 Analyzed Search Results:\n---\n{results}\n---\n\n"
"## 📝 Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant algorithmic ideas or patterns that apply to this problem.\n"
"2. Extract and highlight the key findings, such as time complexity requirements, standard techniques (e.g., BFS/DFS, greedy, DP, binary search), or tricky edge cases.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"
        )

    @staticmethod
    def get_reflect_prompt(question, answer_list):

        prompt = f"""According to the previous agents' answers and reasoning in the **LiveCodeBench** task:
Problem:
{question}

As the fusion agent, your task is to synthesise a high-quality final solution by integrating the strengths of all previous code proposals. 
To ensure the information passed downstream is concise, focused, and context-efficient, follow these rules:

1. Answer (code)
• Provide the entire Python implementation to pass downstream (a full program or required function).  
• If one of the previous codes is already correct, copy it verbatim.  
• If you can fix obvious bugs with ≤ 3 lines of edits, supply the fixed version instead.  

2. Reflection
• 2–4 sentences.  
• State whether the code meets the problem specification and handles the documented input/output format and constraints.  
• Point out any remaining edge cases or potential pitfalls if they exist.

Now, review all previous answers, check the rules above one by one, and propose a cleaned and precise final solution + reflection. 
Most importantly, do not only check the syntax, but also the algorithmic correctness and edge-case handling.

Return your summary strictly in the following format:
<OUTPUT>
Answer:
<full python code>

Reasoning: <2–4 concise sentences describing why this implementation is correct and robust enough for the LiveCodeBench problem.>
</OUTPUT>
"""

        for i, ans in enumerate(answer_list, start=1):
            prompt += f"{i}. {ans}\n"
        return prompt

    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}:\n{answer}\n" for index, answer in enumerate(answers)])
        return (
"# Self-Consistency Evaluation Task (LiveCodeBench)\n\n"
f"## 🤔 Coding Problem:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Solutions:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Instructions:\n"
"1. Read each candidate solution and assess whether it correctly implements a Python program/function that solves the problem.\n"
"2. Compare the answers for their adherence to the problem statement, input/output format, and algorithmic correctness.\n"
"3. Ignore answers that do not contain valid Python code or that obviously do not attempt to solve the task.\n"
"4. Select the solution that is **most likely** to be correct, efficient, and robust to edge cases.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to fulfilling them."
        )

    @staticmethod
    def get_select_best(question: str, solutions: list) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}:\n{answer}\n" for index, answer in enumerate(solutions)])
        return (
"# Best Answer Evaluation Task (LiveCodeBench)\n\n"
f"## 🤔 Problem:\n---\n{question}\n---\n\n"
f"## 💡 Candidate Solutions for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## 📋 Evaluation Instructions:\n"
"1. Examine the problem carefully to understand the required behavior, input format, and output format.\n"
"2. Read each candidate Python solution and assess its algorithmic correctness, complexity, and edge-case handling.\n"
"3. Choose the answer that most accurately and completely solves the problem.\n"
"4. Ignore any surrounding non-code text (e.g., 'as an AI...' or explanations) when deciding, and focus on the code itself.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
"Note: If none of the answers fully meet the problem's criteria, select the one that is closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

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
    def get_summary(answer: str, task: str) -> str:
        prompt = f"""
    You are a **summary agent** in a multi-agent LiveCodeBench system.

    Problem:
    {task}

    The following answer was generated by a previous agent. It may contain redundant comments,
    extra explanations, or even small bugs. Your goals:

    1. **Answer (code)**
       • Keep ONLY the complete, runnable Python implementation (program or required function).  
       • If the code is clearly correct, copy it verbatim.  
       • If you can fix obvious bugs with **≤ 3 lines of edits**, output the fixed version instead.  
       • Remove superfluous comments / debug prints that do not affect correctness.

    2. **Reasoning (optional but concise)**
       • ≤ 3 concise sentences explaining the core algorithm / approach (e.g., greedy, DP, two pointers).  
       • Do NOT invent new problem requirements; just explain how the code satisfies the given one.  
       • If the original answer contains no reasoning and the code is clearly correct, you may omit the reasoning line.

    3. The output must follow **exactly** the format below — nothing more, nothing less.

    Return your cleaned summary strictly in the following format:

    <OUTPUT>
    Answer:
    <full python code block>

    Reasoning: <concise explanation of the core logic, 1–3 sentences, if no reasoning is present in the previous answer, ignore this.>
    </OUTPUT>

    Original Answer:
    \"\"\"
    {answer}
    \"\"\"
    """
        return prompt

    @staticmethod
    def get_verifier(answer, task):
        prompt = f"""
    We are solving a **LiveCodeBench programming task**.

    Problem Specification:
    {task}

    Below is a **candidate Python solution** written by an agent.

    As the **VERIFIER**, carefully evaluate this code on:

    - **Correctness** — Does it implement the required behavior for all typical and edge-case inputs?
    - **Completeness** — Does it handle the specified input/output format and constraints?
    - **Efficiency** — Is the time and space complexity appropriate for typical competitive-programming limits?
    - **Code Quality** — Is it readable and logically structured (even if not perfectly styled)?

    Then, provide:
    1. A **score** between 0 and 10 indicating the overall quality of the solution.
    2. A **brief reason** for the score — mention correctness, logic, edge cases, or improvements needed.

    Format your response **strictly** as follows:

    Score: <number from 0 to 10>  
    Reason: <short explanation of why this score was given>

    Candidate Answer:
    {answer}
    """
        return prompt

    @staticmethod
    def get_model_initialize(model_combo):
        prompt = f"""
You are a researcher specializing in multi-agent systems (MAS).  
Your current task is **model initialization**: under a fixed computational **budget** you must choose an initial set of language-model agents (each model = one node) for a MAS that will later be optimized into a DAG. An edge means the previous agent’s output is the next agent’s input.

================  TASK  =================
1. Examine the **candidate model combinations** listed at the end of this message.  
2. Using the insights and data below, pick **two** combination that will give the best expected performance on the **Math** dataset.  
3. Return **only** two JSON dictionaries with four integer keys:  
   - `"0"` = number of 1 B models  
   - `"1"` = number of 3 B models  
   - `"2"` = number of 8 B models  
   - `"3"` = number of 70 B models  

No extra text, explanations, or formatting—just the dictionary.

===============  INSIGHTS  ===============
(1) Well-designed MASes usually improve as the number of nodes increases, **but**  
    • very long contexts fed to a weak model can hurt accuracy, and  
    • overly deep DAGs may let later agents overwrite correct answers.
    • both depth and width have an optimal point—beyond that, adding more layers or parallel branches starts to decrease overall performance.
    • Too many weak models may decrease the performance instead of increase.
(2) Stronger models (better performance) tolerate longer context and are less likely to corrupt correct answers.  
(3) You must trade off “more nodes help” vs. “too many weak models hurt”. Choose the mix that best balances these forces.
(4) You may refer to the 'Data' section showing the performance of different model combinations under the same budget on this dataset to help you decide which two model selection are the best candidates among the current model selection options.

===============  DATA  ===================
● **Single-model accuracy on livecodebench (higher is better)**  
 1 B = 8  3 B = 20  8 B = 31  70 B = 51
● Random-graph pre-experiments (equal budget):
9X1 B → 18   2X3B + 3X1B → 20  3X3b → 22  1X8B → 31
  

===============  CANDIDATES  =============
Choose only **one** from this list (each already fits the budget):

{model_combo}

=========================================

Respond with the dictionary **only**. Example format (do NOT copy):  
```json
{{"0":0,"1":4,"2":0,"3":0}}
```json
{{"0":0,"1":1,"2":1,"3":0}}
"""
        return prompt


    @staticmethod
    def get_llm_forward(prev_graph,acc,edge_probs,model_selection):

        prompt = f"""
You are a professional **Multi-Agent-System (MAS) optimizer**.  
Your task is an iterative self-RL refinement of a MAS that solves the **livecodebench** dataset.

────────────────────────────────────
TASK CONTEXT
────────────────────────────────────
• A MAS is represented as a **directed acyclic graph (DAG)**.  
  - Each **node** = one language-model agent.  
  - Each **directed edge** = “the source agent's output is appended to the destination agent's context”.  
• For the current budget we have a fixed **model-selection requirement**:  
  {model_selection}
• You will see the **last-round graph**, its **batch accuracy**, and the **full table of edge-selection probabilities**.  
• Your job: **propose the next-round graph** (same format) **and the updated probability table** (same order & format), applying * RL-style* probability nudges.
• The graph you receive in this iteration has been expanded outward from the FinalDecision node, gradually increasing in both depth and breadth. The edge-probabilities starts with all edge probabilities set to zero, and through multiple sampling rounds, probabilities are raised only for edges that prove useful.

────────────────────────────────────
HISTORICAL SNAPSHOT
────────────────────────────────────
Last-round accuracy ( livecodebench-dev batch ) : **{acc:.3%}**  
Last-round graph:  
{prev_graph}
Last-round edge-probabilities:
{edge_probs}

────────────────────────────────────
OPTIMIZATION RULES
────────────────────────────────────
R-1  Model counts must exactly match model_selection after you assign models to all nodes.
R-2  A node's role is either "IO" (generates new answer) or "REV" (reviews & picks best).
R-3  Return values must keep the identical schema / key order as the inputs — only the values may change.
R-4  Increase an edge probability **only if it was sampled in the last-round graph AND proved useful**.  
Always start expansion from FinalDecision's incoming edges, then its parents' incoming edges, and so on.
↑ increase edges used by high-accuracy graphs, ↓ decrease edges from poor graphs.
R-5  Keep the graph acyclic; avoid too much in-degree to prevent context explosion; avoid very deep chains to prevent “answer corruption”.
R-6 If a node appears with model = FinalDecision (this is the single output node of the MAS), do not modify its model or role.
Your optimization may only update the set/probabilities of its incoming edges — that is, adjust which predecessors feed it and with what likelihood—but the node itself must stay unchanged.

────────────────────────────────────
DATA and Insight
────────────────────────────────────
• Model accuracy on livecodebench (single-agent):
1 B = 8  3 B = 20  8 B = 31  70 B = 51
• Larger models tolerate longer context and are harder to corrupt.
• Larger models outperform smaller models when assigned the nodes with more predecessors.
• For nodes with multiple incoming edges, assigning diverse models to their predecessors often yields better performance than using a single repeated model.
• The optimal depth is conditioned by current width, and vice-versa: wider graphs shift the depth sweet-spot downward, while deeper graphs reduce the optimal width.
• You should expand the architecture outward from the FinalDecision node, gradually adding depth and breadth.
• Different tasks favor different graph topologies depending on the model mix. Certain model configurations benefit more from greater depth, while others perform best with greater width. With the current model selection, optimize toward the topology style that this task prefers.
• For this round, increase probabilities only for nodes that boost MAS accuracy, and lower those that harm it.

────────────────────────────────────
WHAT TO RETURN
────────────────────────────────────
Return ONLY two blocks, nothing else.
	1.	graph   - the next-round DAG, same schema as last-round graph.
	2.	edge_probs - the updated probability table, same schema and order as last-round edge-probabilities.
IMPORTANT: The "Graph:" block must list exactly the same nodes as in the last-round graph. Do NOT create any new node lines. Do NOT change any node ID token.

Example output format (do NOT add comments):
Graph:
"Node 3CoH | model=llama3.2-3b-longcontext:latest | role=IO | preds=['4Dhq'] | succs=[]\nNode 4Dhq | model=llama3.2-1b-longcontext:latest | role=IO | preds=[] | succs=['3CoH', 'cFSM']\nNode 5XF3 | model=llama3.2-1b-longcontext:latest | role=IO | preds=['cFSM'] | succs=[]\nNode 6S5Q | model=FinalDecision | role=IO | preds=['cFSM'] | succs=[]\nNode cFSM | model=llama3.2-3b-longcontext:latest | role=REV | preds=['4Dhq'] | succs=['5XF3', '6S5Q']"
Edge-probabilities:
['0: src=DirectAnswer(5XF3), dst=DirectAnswer(3CoH), prob=0.000\n', '1: src=DirectAnswer(5XF3), dst=DirectAnswer(4Dhq), prob=0.000\n', '2: src=DirectAnswer(5XF3), dst=DirectAnswer(cFSM), prob=0.000\n', '3: src=DirectAnswer(3CoH), dst=DirectAnswer(5XF3), prob=0.000\n', '4: src=DirectAnswer(3CoH), dst=DirectAnswer(4Dhq), prob=0.000\n', '5: src=DirectAnswer(3CoH), dst=DirectAnswer(cFSM), prob=0.000\n', '6: src=DirectAnswer(4Dhq), dst=DirectAnswer(5XF3), prob=0.000\n', '7: src=DirectAnswer(4Dhq), dst=DirectAnswer(3CoH), prob=0.100\n', '8: src=DirectAnswer(4Dhq), dst=DirectAnswer(cFSM), prob=0.150\n', '9: src=DirectAnswer(cFSM), dst=DirectAnswer(5XF3), prob=0.00\n', '10: src=DirectAnswer(cFSM), dst=DirectAnswer(3CoH), prob=0.000\n', '11: src=DirectAnswer(cFSM), dst=DirectAnswer(4Dhq), prob=0.000\n', '12: src=DirectAnswer(5XF3), dst=FinalDecision(6S5Q), prob=0.000\n', '13: src=DirectAnswer(3CoH), dst=FinalDecision(6S5Q), prob=0.000\n', '14: src=DirectAnswer(4Dhq), dst=FinalDecision(6S5Q), prob=0.000\n', '15: src=DirectAnswer(cFSM), dst=FinalDecision(6S5Q), prob=0.100\n']

Now think step-by-step with the rules and insights above and return the Graph and Edge-probabilities two blocks only.
Disallow the following symbol-sequence pattern: a single space, then several arbitrary tokens, followed by a placeholder or ellipsis.
"""
        return prompt
    
    @staticmethod
    def get_model_initialize_textgrad(model_combo):
        prompt = f"""
You are a researcher specializing in multi-agent systems (MAS).  
Your current task is **model initialization**: under a fixed computational **budget** you must choose an initial set of language-model agents (each model = one node) for a MAS that will later be optimized into a DAG. An edge means the previous agent’s output is the next agent’s input.

================  TASK  =================
1. Examine the **candidate model combinations** listed at the end of this message.  
2. Based on your knowledge of test-time scaling, pick **one** combination that will give the best expected performance on the **livecodebench** dataset.  
3. Return **only** a JSON dictionary with four integer keys:  
   - `"0"` = number of 1 B models  
   - `"1"` = number of 3 B models  
   - `"2"` = number of 8 B models  
   - `"3"` = number of 70 B models  

No extra text, explanations, or formatting—just the dictionary.

===============  CANDIDATES  =============
Choose only **one** from this list (each already fits the budget):

{model_combo}

=========================================

Respond with the dictionary **only**. Example format (do NOT copy):  
```json
{{"0":7,"1":3,"2":0,"3":0}}
"""
        return prompt


    @staticmethod
    def get_llm_forward_textgrad(prev_graph,acc,edge_probs,model_selection):

        prompt = f"""
You are a professional **Multi-Agent-System (MAS) optimizer**.  
Your task is an iterative self-RL refinement of a MAS that solves the **LiveCodeBench** dataset.

────────────────────────────────────
TASK CONTEXT
────────────────────────────────────
• A MAS is represented as a **directed acyclic graph (DAG)**.  
  - Each **node** = one language-model agent.  
  - Each **directed edge** = “the source agent's output is appended to the destination agent's context”.  
• For the current budget we have a fixed **model-selection requirement**:  
  {model_selection}
• You will see the **last-round graph**, its **batch accuracy**, and the **full table of edge-selection probabilities**.  
• Your job: **propose the next-round graph** (same format) **and the updated probability table** (same order & format), applying * RL-style* probability nudges.

────────────────────────────────────
HISTORICAL SNAPSHOT
────────────────────────────────────
Last-round accuracy ( LiveCodeBench-dev batch ) : **{acc:.3%}**  
Last-round graph:  
{prev_graph}
Last-round edge-probabilities:
{edge_probs}

────────────────────────────────────
OPTIMIZATION RULES
────────────────────────────────────
R-1  Model counts must exactly match model_selection after you assign models to all nodes.
R-2  A node's role is either "IO" (generates new answer) or "REV" (reviews & picks best).
R-3  Return values must keep the identical schema / key order as the inputs — only the values may change.
R-4  Keep the graph acyclic.
R-5  If a node appears with model = FinalDecision (this is the single output node of the MAS), do not modify its model or role.
Your optimization may only update the set/probabilities of its incoming edges — that is, adjust which predecessors feed it and with what likelihood—but the node itself must stay unchanged.

────────────────────────────────────
WHAT TO RETURN
────────────────────────────────────
Return ONLY two blocks, nothing else.
	1.	graph   - the next-round DAG, same schema as last-round graph.
	2.	edge_probs - the updated probability table, same schema and order as last-round edge-probabilities.
IMPORTANT: The "Graph:" block must list exactly the same nodes as in the last-round graph. Do NOT create any new node lines. Do NOT change any node ID token.

Example output format (do NOT add comments):
Graph:
"Node 3CoH | model=llama3.2-3b-longcontext:latest | role=IO | preds=['4Dhq'] | succs=[]\nNode 4Dhq | model=gemma3-1b-longcontext:latest | role=IO | preds=[] | succs=['3CoH', 'cFSM']\nNode 5XF3 | model=gemma1-7b-longcontext:latest | role=IO | preds=['cFSM'] | succs=[]\nNode 6S5Q | model=FinalDecision | role=IO | preds=['cFSM'] | succs=[]\nNode cFSM | model=qwen2.5-3b-longcontext:latest | role=REV | preds=['4Dhq'] | succs=['5XF3', '6S5Q']"
Edge-probabilities:
['0: src=DirectAnswer(5XF3), dst=DirectAnswer(3CoH), prob=0.000\n', '1: src=DirectAnswer(5XF3), dst=DirectAnswer(4Dhq), prob=0.000\n', '2: src=DirectAnswer(5XF3), dst=DirectAnswer(cFSM), prob=0.000\n', '3: src=DirectAnswer(3CoH), dst=DirectAnswer(5XF3), prob=0.000\n', '4: src=DirectAnswer(3CoH), dst=DirectAnswer(4Dhq), prob=0.000\n', '5: src=DirectAnswer(3CoH), dst=DirectAnswer(cFSM), prob=0.000\n', '6: src=DirectAnswer(4Dhq), dst=DirectAnswer(5XF3), prob=0.000\n', '7: src=DirectAnswer(4Dhq), dst=DirectAnswer(3CoH), prob=0.100\n', '8: src=DirectAnswer(4Dhq), dst=DirectAnswer(cFSM), prob=0.150\n', '9: src=DirectAnswer(cFSM), dst=DirectAnswer(5XF3), prob=0.00\n', '10: src=DirectAnswer(cFSM), dst=DirectAnswer(3CoH), prob=0.000\n', '11: src=DirectAnswer(cFSM), dst=DirectAnswer(4Dhq), prob=0.000\n', '12: src=DirectAnswer(5XF3), dst=FinalDecision(6S5Q), prob=0.000\n', '13: src=DirectAnswer(3CoH), dst=FinalDecision(6S5Q), prob=0.000\n', '14: src=DirectAnswer(4Dhq), dst=FinalDecision(6S5Q), prob=0.000\n', '15: src=DirectAnswer(cFSM), dst=FinalDecision(6S5Q), prob=0.100\n']

Now think step-by-step with the rules above and return the Graph and Edge-probabilities two blocks only.
Disallow the following symbol-sequence pattern: a single space, then several arbitrary tokens, followed by a placeholder or ellipsis.
"""
        return prompt

    @staticmethod
    def get_llm_forward_maao(prev_graph,acc,edge_probs,role_probs):

        prompt = f"""
You are a professional **Multi-Agent-System (MAS) optimizer**.  
Your task is an iterative self-RL refinement of a MAS that solves the **livecodebench** dataset.

────────────────────────────────────
TASK CONTEXT
────────────────────────────────────
• A MAS is represented as a **directed acyclic graph (DAG)**.  
  - Each **node** = one language-model agent.  
  - Each **directed edge** = “the source agent's output is appended to the destination agent's context”.  
• You will see the **last-round graph**, its **batch accuracy**, and the **full table of edge-selection probabilities,role-selection probabilities**.  
• Your job: updated all probability table based on the accuracy of the previous graph ** (same order & format), applying * RL-style* probability nudges.

────────────────────────────────────
HISTORICAL SNAPSHOT
────────────────────────────────────
Last-round accuracy ( livecodebench-dev batch ) : **{acc:.3%}**  
Last-round graph:  
{prev_graph}
Last-round edge-probabilities:
{edge_probs}
Last-round role-probabilities:
{role_probs}

────────────────────────────────────
OPTIMIZATION RULES
────────────────────────────────────
R-1  Return values must keep the identical schema / key order as the inputs — only the values may change.
R-2  If a node appears with model = FinalDecision (this is the single output node of the MAS), do not modify its role's probabilities.
Your optimization may only update the set/probabilities of its incoming edges — that is, adjust which predecessors feed it and with what likelihood—but the node itself must stay unchanged.

────────────────────────────────────
WHAT TO RETURN
────────────────────────────────────
Return ONLY the following **two blocks**, in the order shown below — nothing else:
	1. Edge_probs - the updated probability table, same schema and order as last-round edge-probabilities.
    2. Role-probabilities — updated role assignment probabilities. Must preserve:
   • Same node list as input  
   • Same order  
   • Only values may change
    
Example output format (do NOT add comments):
Edge-probabilities:
['0: src=DirectAnswer(6bfQ), dst=DirectAnswer(jCyH), prob=0.475\n', '1: src=DirectAnswer(jCyH), dst=DirectAnswer(6bfQ), prob=0.475\n', '2: src=DirectAnswer(6bfQ), dst=FinalDecision(oznx), prob=0.475\n', '3: src=DirectAnswer(jCyH), dst=FinalDecision(oznx), prob=0.525\n']
Role-probabilities:
['0: node=FinalDecision(oznx), IO=0.500, REV=0.500\n', '1: node=DirectAnswer(6bfQ), IO=0.550, REV=0.450\n', '2: node=DirectAnswer(jCyH), IO=0.550, REV=0.450\n']

Now think step-by-step with the rules above and return the two probabilities blocks only.
Disallow the following symbol-sequence pattern: a single space, then several arbitrary tokens, followed by a placeholder or ellipsis.
"""
        return prompt

    @staticmethod
    def get_llm_forward_latency(prev_graph,acc,edge_probs,model_selection,latency_list):

        prompt = f"""
You are a professional **Multi-Agent-System (MAS) optimizer**.  
Your task is an iterative self-RL refinement of a MAS that solves the **Math** dataset.

────────────────────────────────────
TASK CONTEXT
────────────────────────────────────
• A MAS is represented as a **directed acyclic graph (DAG)**.  
  - Each **node** = one language-model agent.  
  - Each **directed edge** = “the source agent's output is appended to the destination agent's context”.  
• For the current budget we have a fixed **model-selection requirement**:  
  {model_selection}
• You will see the **last-round graph**, its **batch accuracy**, and the **full table of edge-selection probabilities**.  
• Your job: **propose the next-round graph** (same format) **and the updated probability table** (same order & format), applying * RL-style* probability nudges.
• The graph you receive in this iteration has been expanded outward from the FinalDecision node, gradually increasing in both depth and breadth. The edge-probabilities starts with all edge probabilities set to zero, and through multiple sampling rounds, probabilities are raised only for edges that prove useful.

────────────────────────────────────
HISTORICAL SNAPSHOT
────────────────────────────────────
Last-round accuracy ( Math-dev batch ) : **{acc:.3%}**  
Last-round graph:  
{prev_graph}
Last-round edge-probabilities:
{edge_probs}

────────────────────────────────────
OPTIMIZATION RULES
────────────────────────────────────
R-1  Model counts must exactly match model_selection after you assign models to all nodes.
R-2  A node's role is either "IO" (generates new answer) or "REV" (reviews & picks best).
R-3  Return values must keep the identical schema / key order as the inputs — only the values may change.
R-4  Increase an edge probability **only if it was sampled in the last-round graph AND proved useful**.  
Always start expansion from FinalDecision's incoming edges, then its parents' incoming edges, and so on.
↑ increase edges used by high-accuracy graphs, ↓ decrease edges from poor graphs.
R-5  Keep the graph acyclic; avoid too much in-degree to prevent context explosion; avoid very deep chains to prevent “answer corruption”.
R-6 If a node appears with model = FinalDecision (this is the single output node of the MAS), do not modify its model or role.
Your optimization may only update the set/probabilities of its incoming edges — that is, adjust which predecessors feed it and with what likelihood—but the node itself must stay unchanged.

────────────────────────────────────
DATA and Insight
────────────────────────────────────
• Model accuracy on Math (single-agent):
1 B → 31   3 B → 40   8 B → 48   70 B → 68

• Larger models tolerate longer context and are harder to corrupt.
• Larger models outperform smaller models when assigned the nodes with more predecessors.
• For nodes with multiple incoming edges, assigning diverse models to their predecessors often yields better performance than using a single repeated model.
• The optimal depth is conditioned by current width, and vice-versa: wider graphs shift the depth sweet-spot downward, while deeper graphs reduce the optimal width.
• You should expand the architecture outward from the FinalDecision node, gradually adding depth and width.
• Different tasks favor different graph topologies depending on the model mix. Certain model configurations benefit more from greater depth, while others perform best with greater width. With the current model selection, optimize toward the topology style that this task prefers.
• For this round, increase probabilities only for nodes that boost MAS accuracy, and lower those that harm it.

────────────────────────────────────
WHAT TO RETURN
────────────────────────────────────
Return ONLY two blocks, nothing else.
	1.	graph - the next-round DAG, same schema as last-round graph.
	2.	edge_probs - the updated probability table, same schema and order as last-round edge-probabilities.
IMPORTANT: The "Graph:" block must list exactly the same nodes as in the last-round graph. Do NOT create any new node lines. Do NOT change any node ID token.

Example output format (do NOT add comments):
Graph:
"Node 3CoH | model=llama3.2-3b-longcontext:latest | role=IO | preds=['4Dhq'] | succs=[]\nNode 4Dhq | model=llama3.2-1b-longcontext:latest | role=IO | preds=[] | succs=['3CoH', 'cFSM']\nNode 5XF3 | model=llama3.2-1b-longcontext:latest | role=IO | preds=['cFSM'] | succs=[]\nNode 6S5Q | model=FinalDecision | role=IO | preds=['cFSM'] | succs=[]\nNode cFSM | model=llama3.2-3b-longcontext:latest | role=REV | preds=['4Dhq'] | succs=['5XF3', '6S5Q']"
Edge-probabilities:
['0: src=DirectAnswer(5XF3), dst=DirectAnswer(3CoH), prob=0.000\n', '1: src=DirectAnswer(5XF3), dst=DirectAnswer(4Dhq), prob=0.000\n', '2: src=DirectAnswer(5XF3), dst=DirectAnswer(cFSM), prob=0.000\n', '3: src=DirectAnswer(3CoH), dst=DirectAnswer(5XF3), prob=0.000\n', '4: src=DirectAnswer(3CoH), dst=DirectAnswer(4Dhq), prob=0.000\n', '5: src=DirectAnswer(3CoH), dst=DirectAnswer(cFSM), prob=0.000\n', '6: src=DirectAnswer(4Dhq), dst=DirectAnswer(5XF3), prob=0.000\n', '7: src=DirectAnswer(4Dhq), dst=DirectAnswer(3CoH), prob=0.100\n', '8: src=DirectAnswer(4Dhq), dst=DirectAnswer(cFSM), prob=0.150\n', '9: src=DirectAnswer(cFSM), dst=DirectAnswer(5XF3), prob=0.00\n', '10: src=DirectAnswer(cFSM), dst=DirectAnswer(3CoH), prob=0.000\n', '11: src=DirectAnswer(cFSM), dst=DirectAnswer(4Dhq), prob=0.000\n', '12: src=DirectAnswer(5XF3), dst=FinalDecision(6S5Q), prob=0.000\n', '13: src=DirectAnswer(3CoH), dst=FinalDecision(6S5Q), prob=0.000\n', '14: src=DirectAnswer(4Dhq), dst=FinalDecision(6S5Q), prob=0.000\n', '15: src=DirectAnswer(cFSM), dst=FinalDecision(6S5Q), prob=0.100\n']

Now think step-by-step with the rules and insights above and return the Graph and Edge-probabilities two blocks only.
Disallow the following symbol-sequence pattern: a single space, then several arbitrary tokens, followed by a placeholder or ellipsis.
"""
        return prompt
    
    @staticmethod
    def get_llm_forward_ablation_role(prev_graph,acc,edge_probs,model_selection):

        prompt = f"""
You are a professional **Multi-Agent-System (MAS) optimizer**.  
Your task is an iterative self-RL refinement of a MAS that solves the **livecodebench** dataset.

────────────────────────────────────
TASK CONTEXT
────────────────────────────────────
• A MAS is represented as a **directed acyclic graph (DAG)**.  
  - Each **node** = one language-model agent.  
  - Each **directed edge** = “the source agent's output is appended to the destination agent's context”.  
• For the current budget we have a fixed **model-selection requirement**:  
  {model_selection}
• You will see the **last-round graph**, its **batch accuracy**, and the **full table of edge-selection probabilities**.  
• Your job: **propose the next-round graph** (same format) **and the updated probability table** (same order & format), applying * RL-style* probability nudges.
• The graph you receive in this iteration has been expanded outward from the FinalDecision node, gradually increasing in both depth and breadth. The edge-probabilities starts with all edge probabilities set to zero, and through multiple sampling rounds, probabilities are raised only for edges that prove useful.

────────────────────────────────────
HISTORICAL SNAPSHOT
────────────────────────────────────
Last-round accuracy ( livecodebench-dev batch ) : **{acc:.3%}**  
Last-round graph:  
{prev_graph}
Last-round edge-probabilities:
{edge_probs}

────────────────────────────────────
OPTIMIZATION RULES
────────────────────────────────────
R-1  Model counts must exactly match model_selection after you assign models to all nodes.
R-2  A node's role is either "IO" (generates new answer) or "REV" (reviews & picks best).
R-3  Return values must keep the identical schema / key order as the inputs — only the values may change.
R-4  Increase an edge probability **only if it was sampled in the last-round graph AND proved useful**.  
Always start expansion from FinalDecision's incoming edges, then its parents' incoming edges, and so on.
↑ increase edges used by high-accuracy graphs, ↓ decrease edges from poor graphs.
R-5  Keep the graph acyclic; avoid too much in-degree to prevent context explosion; avoid very deep chains to prevent “answer corruption”.
R-6 If a node appears with model = FinalDecision (this is the single output node of the MAS), do not modify its model or role.
Your optimization may only update the set/probabilities of its incoming edges — that is, adjust which predecessors feed it and with what likelihood—but the node itself must stay unchanged.

────────────────────────────────────
DATA and Insight
────────────────────────────────────
• Model accuracy on livecodebench (single-agent):
1 B → 31   3 B → 50   8 B → 60   70 B → 68
• Larger models tolerate longer context and are harder to corrupt.
• Larger models outperform smaller models when assigned the nodes with more predecessors.
• For nodes with multiple incoming edges, assigning diverse models to their predecessors often yields better performance than using a single repeated model.
• The optimal depth is conditioned by current width, and vice-versa: wider graphs shift the depth sweet-spot downward, while deeper graphs reduce the optimal width.
• You should expand the architecture outward from the FinalDecision node, gradually adding depth and breadth.
• Different tasks favor different graph topologies depending on the model mix. Certain model configurations benefit more from greater depth, while others perform best with greater width. With the current model selection, optimize toward the topology style that this task prefers.
• For this round, increase probabilities only for nodes that boost MAS accuracy, and lower those that harm it.

────────────────────────────────────
WHAT TO RETURN
────────────────────────────────────
Return ONLY two blocks, nothing else.
	1.	graph   - the next-round DAG, same schema as last-round graph.
	2.	edge_probs - the updated probability table, same schema and order as last-round edge-probabilities.
IMPORTANT: The "Graph:" block must list exactly the same nodes as in the last-round graph. Do NOT create any new node lines. Do NOT change any node ID token.

Example output format (do NOT add comments):
Graph:
"Node 3CoH | model=llama3.2-3b-longcontext:latest | role=IO | preds=['4Dhq'] | succs=[]\nNode 4Dhq | model=llama3.2-1b-longcontext:latest | role=IO | preds=[] | succs=['3CoH', 'cFSM']\nNode 5XF3 | model=llama3.2-1b-longcontext:latest | role=IO | preds=['cFSM'] | succs=[]\nNode 6S5Q | model=FinalDecision | role=IO | preds=['cFSM'] | succs=[]\nNode cFSM | model=llama3.2-3b-longcontext:latest | role=IO | preds=['4Dhq'] | succs=['5XF3', '6S5Q']"
Edge-probabilities:
['0: src=DirectAnswer(5XF3), dst=DirectAnswer(3CoH), prob=0.000\n', '1: src=DirectAnswer(5XF3), dst=DirectAnswer(4Dhq), prob=0.000\n', '2: src=DirectAnswer(5XF3), dst=DirectAnswer(cFSM), prob=0.000\n', '3: src=DirectAnswer(3CoH), dst=DirectAnswer(5XF3), prob=0.000\n', '4: src=DirectAnswer(3CoH), dst=DirectAnswer(4Dhq), prob=0.000\n', '5: src=DirectAnswer(3CoH), dst=DirectAnswer(cFSM), prob=0.000\n', '6: src=DirectAnswer(4Dhq), dst=DirectAnswer(5XF3), prob=0.000\n', '7: src=DirectAnswer(4Dhq), dst=DirectAnswer(3CoH), prob=0.100\n', '8: src=DirectAnswer(4Dhq), dst=DirectAnswer(cFSM), prob=0.150\n', '9: src=DirectAnswer(cFSM), dst=DirectAnswer(5XF3), prob=0.00\n', '10: src=DirectAnswer(cFSM), dst=DirectAnswer(3CoH), prob=0.000\n', '11: src=DirectAnswer(cFSM), dst=DirectAnswer(4Dhq), prob=0.000\n', '12: src=DirectAnswer(5XF3), dst=FinalDecision(6S5Q), prob=0.000\n', '13: src=DirectAnswer(3CoH), dst=FinalDecision(6S5Q), prob=0.000\n', '14: src=DirectAnswer(4Dhq), dst=FinalDecision(6S5Q), prob=0.000\n', '15: src=DirectAnswer(cFSM), dst=FinalDecision(6S5Q), prob=0.100\n']

Now think step-by-step with the rules and insights above and return the Graph and Edge-probabilities two blocks only.
Disallow the following symbol-sequence pattern: a single space, then several arbitrary tokens, followed by a placeholder or ellipsis.
"""
        return prompt
    
    @staticmethod
    def get_model_initialize_latency(model_combo):
        prompt = f"""
You are a researcher specializing in multi-agent systems (MAS).  
Your current task is **model initialization**: under a fixed computational **budget** you must choose an initial set of language-model agents (each model = one node) for a MAS that will later be optimized into a DAG. An edge means the previous agent’s output is the next agent’s input.

================  TASK  =================
1. Examine the **candidate model combinations** listed at the end of this message.  
2. Using the insights and data below, pick **two** combination that achieves the highest efficiency under acceptable performance on the Math dataset.  
3. Return **only** two JSON dictionaries with four integer keys:  
   - `"0"` = number of 1 B models  
   - `"1"` = number of 3 B models  
   - `"2"` = number of 8 B models  
   - `"3"` = number of 70 B models  

No extra text, explanations, or formatting—just the dictionary.

===============  INSIGHTS  ===============
(1) Well-designed MASes usually improve as the number of nodes increases, **but**  
    • very long contexts fed to a weak model can hurt accuracy, and  
    • overly deep DAGs may let later agents overwrite correct answers.
    • both depth and width have an optimal point—beyond that, adding more layers or parallel branches starts to decrease overall performance.
    • Too many weak models may decrease the performance instead of increase.
(2) Stronger models (with higher standalone performance) can handle longer contexts and are less likely to corrupt correct answers, but they are also slower and more costly.
(3) Efficiency should be prioritized:
• Systems with more small models generally have higher inference efficiency (lower latency, lower cost).
• When choosing among combinations, prefer those that use small or efficient models even if their performance is slightly lower—as long as the overall performance remains acceptable.
• The goal is to maximize efficiency under acceptable performance, not to maximize raw accuracy.
(4) You may refer to the 'Data' section showing the performance of different model combinations under the same budget on this dataset to help you decide which two model selection are the best candidates among the current model selection options.

===============  DATA  ===================
● **Single-model accuracy on Math (higher is better)**  
1 B = 31  3 B = 40  8 B = 49  70 B = 68
● Random-graph pre-experiments (equal budget):
9X1 B → 39   2X3B + 3X1B → 48  1X3B + 6X1B → 47  3X3B → 54  1X8B → 49


===============  CANDIDATES  =============
Choose only **two** from this list (each already fits the budget):

{model_combo}

=========================================

Respond with the dictionary **only**. Example format (do NOT copy):  
```json
{{"0":0,"1":4,"2":0,"3":0}}
```json
{{"0":0,"1":1,"2":1,"3":0}}
"""
        return prompt
