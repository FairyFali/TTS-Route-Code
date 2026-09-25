from typing import Dict, List, Any, Tuple
import json
import re, ast
import random
import time

from swarm.llm import LLMRegistry
from swarm.llm.format import Message
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry

class LLM_optimizer():
    def __init__(self, 
                 domain: str,
                 model_name = 'deepseek-r1'): # deepseek-r1
        self.domain = domain
        self.model_name = model_name  # DeepSeek-R1 by default
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)  # prompt_set也是提前注册好的
    
    def enumerate_model_combos(self, budget: int,
                           models_cost: Dict[str, int]) -> List[Dict[str, int]]:
        def _find_cost(sub: str) -> int:
            """根据子串(1b/3b/8b/70b)在 models_cost 中找到对应 cost。"""
            for k, v in models_cost.items():
                if sub in k.lower():
                    return int(v)
            raise ValueError(f"models_cost 里找不到包含 “{sub}” 的键")

        c1, c3, c8, c70 = (
            _find_cost("1b"),
            _find_cost("3b"),
            _find_cost("8b"),
            _find_cost("70b")
        )

        best_leftover = budget                
        combos: List[Dict[str, int]] = []     

        for n70 in range(budget // c70 + 1):
            used70 = n70 * c70
            rem70  = budget - used70

            for n8 in range(rem70 // c8 + 1):
                used8 = n8 * c8
                rem8  = rem70 - used8

                for n3 in range(rem8 // c3 + 1):
                    used3 = n3 * c3
                    rem3  = rem8 - used3
                    n1 = rem3 // c1
                    spend    = used70 + used8 + used3 + n1 * c1
                    leftover = budget - spend

                    combo = {"0": n1, "1": n3, "2": n8, "3": n70}

                    if leftover < best_leftover:
                        best_leftover = leftover
                        combos = [combo]          
                    elif leftover == best_leftover:
                        combos.append(combo)    

        uniq, seen = [], set()
        for c in combos:
            key = (c["0"], c["1"], c["2"], c["3"])
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq
    
    def enumerate_model_combos_multi(
        self,
        budget: int,
        models_cost: Dict[str, int],
        model_order = [
            'llama3.2-1b-longcontext:latest',
            'llama3.2-3b-longcontext:latest',
            'llama3.1-8b-longcontext:latest',
            'llama3.1-70b-longcontext:latest',
            'gemma3-1b-longcontext:latest',
            'gemma1-2b-longcontext:latest',
            'gemma1-7b-longcontext:latest',
            'qwen2.5-1.5b-longcontext:latest',
            'qwen2.5-3b-longcontext:latest',
            'qwen2.5-7b-longcontext:latest',
        ]
) -> List[Dict[str, int]]:

        idx2info = [(i, m, models_cost[m])
                for i, m in enumerate(model_order) if models_cost[m] > 0]

        best_left = budget
        raw_combos: List[Dict[str, int]] = []

        def dfs(pos: int, used: int, cur: Dict[str, int]):
            nonlocal best_left, raw_combos
            if used > budget:
                return
            if pos == len(idx2info):
                left = budget - used
                if left < best_left:
                    best_left = left
                    raw_combos = [cur.copy()]
                elif left == best_left:
                    raw_combos.append(cur.copy())
                return

            idx, _, cost = idx2info[pos]
            max_n = (budget - used) // cost
            for n in range(max_n + 1):
                key = str(idx)
                if n:
                    cur[key] = n
                elif key in cur:
                    del cur[key]
                dfs(pos + 1, used + n * cost, cur)

            # 回溯
            cur.pop(str(idx), None)

        dfs(0, 0, {})

        full_combos: List[Dict[str, int]] = []
        all_keys = [str(i) for i in range(len(model_order))]
        for c in raw_combos:
            full = {k: c.get(k, 0) for k in all_keys}
            full_combos.append(full)

        return full_combos
   
    def model_initialize(self,budget,model_cost):
        '''
        enumerates combos, builds a prompt via prompt_set.get_model_initialize(...), asks
          the LLM to choose, parses JSON out of the reply (inner extract_json_dict: strips ```json fences, regex-finds {...},
          json.loads).
          Computes agents_num = largest agent count among returned combos. Returns (model_selection, agents_num, cost).
        Args:
            budget:
            model_cost:

        Returns:

        '''
        """def extract_json_dict(text: str):
            # 去掉前后空格
            text = text.strip()

            # 如果是 markdown 格式 ```json ... ```
            if text.startswith("```"):
                # 去掉三引号包裹
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)

            # 用正则提取 { ... } 部分
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text}")

            return json.loads(match.group())"""

        def extract_json_dict(text: str):
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)

            # 提取所有 JSON 对象
            matches = re.findall(r"\{.*?\}", text, re.DOTALL)
            if not matches:
                raise ValueError(f"No JSON object found in response: {text}")

            json_objects = []
            for json_str in matches:
                try:
                    json_objects.append(json.loads(json_str))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON found: {json_str}\nError: {e}")

            # 如果只有一个对象，返回它本身；否则返回列表
            return json_objects
        
        model_combo = self.enumerate_model_combos(budget,model_cost)
        prompt = self.prompt_set.get_model_initialize(model_combo)
        # 这里model combo是尽可能用完预算的模型组合，初始化调用LLM的目的是为了挑选最好的两个model combo来进行初始化（这一步没有必要）。
        message = [Message(role="system", content=f"You are a researcher specializing in multi-agent systems (MAS)."),Message(role="user", content=prompt)]
        if "llama" in self.model_name:
            response = self.llm.gen(message,16384)
            price = 0
        else:
            start_time = time.time()
            response,price = self.llm.gen(message,16384)
            time_cost = time.time() - start_time
            cost = (price,time_cost)
        model_selection = extract_json_dict(response)  # model_selection 有一个还是两个？？？
        agents_num = 0
        for dict_m in model_selection:
            agents_count = sum(dict_m.values())
            if agents_count > agents_num:
                agents_num = agents_count
        # 返回初始化模型组合，组合里面的agent/LLM数量，搜索cost
        return model_selection, agents_num, cost
    
    def llm_forward(self,
        prev_graph,
        acc: float,
        edge_probs,
        his_response: list,
        model_selection: dict,
        graph,
        models):


        def convert_model_dict(idx_dict: Dict[Any, int]) -> Dict[str, int]:
            k0 = idx_dict.get("0", idx_dict.get(0, 0))
            k1 = idx_dict.get("1", idx_dict.get(1, 0))
            k2 = idx_dict.get("2", idx_dict.get(2, 0))
            k3 = idx_dict.get("3", idx_dict.get(3, 0))

            """return {
                "1b":  int(k0),
                "3b":  int(k1),
                "8b":  int(k2),
                "70b": int(k3),
            }"""
            return {
                "1b":  int(k3),
                "3b":  int(k0),
                "8b":  int(k1),
                "70b": int(k3),
            }
        
        def extract_graph_and_edges(response: str) -> Tuple[str, str]:
            m_graph = re.search(
                r"Graph:\s*(.*?)\s*Edge-probabilities:",
                response,
                flags=re.S
            )
            if not m_graph:
                raise ValueError("Cannot find 'Graph:' section in LLM response")
            graph_raw = m_graph.group(1).strip()

            # 去掉首尾包裹的引号（LLM 常用 "..." 返回）
            if (graph_raw.startswith('"') and graph_raw.endswith('"')) or \
            (graph_raw.startswith("'") and graph_raw.endswith("'")):
                graph_raw = graph_raw[1:-1]

            graph_str = graph_raw.replace("\\n", "\n")   # 若 LLM 转义了换行符

            # -------- 2. Edge-probabilities 段 --------
            m_edge = re.search(r"Edge-probabilities:\s*(.*)", response, flags=re.S)
            if not m_edge:
                raise ValueError("Cannot find 'Edge-probabilities:' section")
            edge_raw = m_edge.group(1).strip()

            try:
                edge_list = ast.literal_eval(edge_raw)
                # edge_list 可能是 list[str] 或 tuple[str]
                if isinstance(edge_list, (list, tuple)):
                    edge_probs_str = "".join(edge_list).replace("\\n", "\n")
                else:  # 解析不到 list，就保持原样
                    edge_probs_str = str(edge_list)
            except Exception:
                # 解析失败就直接返回原字符串
                edge_probs_str = edge_raw.replace("\\n", "\n")

            return graph_str.strip(), edge_probs_str.strip()
        
        model_selection = convert_model_dict(model_selection)
        total_model = sum(model_selection.values())
        message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS).",)]
        if his_response:                                     # 至少完成过 1 轮
            len_resp = len(his_response)                    # 已完成轮数
            len_hist = min(1, len_resp)                     # 最多取 3 轮

            # 2-A. 最近 ≤3 轮（从旧到新）user / assistant 对
            start_idx = len_resp - len_hist                 # 起始索引
            for i in range(start_idx, len_resp):
                user_prompt = self.prompt_set.get_llm_forward(
                    prev_graph[i], acc[i], edge_probs[i], model_selection
                )
                message.append(Message(role="user",      content=user_prompt))
                message.append(Message(role="assistant", content=his_response[i]))

            # 2-B. 本轮 user prompt（graph 比 his_response 长 1）
            curr_idx = len(prev_graph) - 1
            curr_prompt = self.prompt_set.get_llm_forward(
                prev_graph[curr_idx], acc[curr_idx], edge_probs[curr_idx], model_selection
            )
            message.append(Message(role="user", content=curr_prompt))

        else:
            prompt = self.prompt_set.get_llm_forward(prev_graph[-1],acc[-1],edge_probs[-1],model_selection)
            message.append(Message(role="user", content=prompt))
        max_retries = 3
        CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")
        price = 0        
        for attempt in range(max_retries):
            if total_model > 5:
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
            if "llama" in self.model_name:
                response = self.llm.gen(message,16384)
            else:
                response,price_i = self.llm.gen(message, 16384)
                price += price_i
            graph_str, edge_probs = extract_graph_and_edges(response)

            # 检查 node IDs 合法性
            valid_ids = {n for n in graph.nodes}
            found_ids = re.findall(r'Node\s+([A-Za-z0-9]+)\b', graph_str)
            bad_ids = [nid for nid in found_ids if nid not in valid_ids]

            if CHINESE_RE.search(graph_str) or CHINESE_RE.search(edge_probs):
                print(f"[Retry {attempt}/{max_retries}] Chinese characters detected, regenerating...")
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
                continue

            if not bad_ids:
                # 校验通过，退出循环
                break
            else:
                print(f"[Warning] Attempt {attempt+1}/{max_retries}: Invalid node IDs {bad_ids}, retrying...")

        else:
            # 如果循环结束还没 break，说明三次都失败
            raise ValueError(f"LLM produced invalid node IDs after {max_retries} attempts: {bad_ids}")

        
        graph.apply_graph_string(graph,graph_str,models)
        return graph, edge_probs, response, price

    def model_initialize_textgrad(self,budget,model_cost):
        model_combo = self.enumerate_model_combos_multi(budget,model_cost)
        model_selection = random.choice(model_combo)
        agents_num = sum(model_selection.values())

        return model_selection, agents_num
    
    def model_initialize_ablation_random(self,budget,model_cost):
        model_combo = self.enumerate_model_combos(budget,model_cost)
        model_selection = random.choice(model_combo)
        agents_num = sum(model_selection.values())

        return model_selection, agents_num
    
    def model_initialize_ablation(self,budget,model_cost):
        def extract_json_dict(text: str):
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)

            # 提取所有 JSON 对象
            matches = re.findall(r"\{.*?\}", text, re.DOTALL)
            if not matches:
                raise ValueError(f"No JSON object found in response: {text}")

            json_objects = []
            for json_str in matches:
                try:
                    json_objects.append(json.loads(json_str))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON found: {json_str}\nError: {e}")

            # 如果只有一个对象，返回它本身；否则返回列表
            return json_objects
        
        model_combo = self.enumerate_model_combos(budget,model_cost)
        prompt = self.prompt_set.get_model_initialize_textgrad(model_combo)
        message = [Message(role="system", content=f"You are a researcher specializing in multi-agent systems (MAS)."),Message(role="user", content=prompt)]
        response = self.llm.gen(message,16384)
        if "llama" in self.model_name:
            response = self.llm.gen(message,16384)
            price = 0
        else:
            start_time = time.time()
            response,price = self.llm.gen(message,16384)
            time_cost = time.time() - start_time
            cost = (price,time_cost)
        model_selection = extract_json_dict(response)
        agents_num = 0
        for dict_m in model_selection:
            agents_count = sum(dict_m.values())
            if agents_count > agents_num:
                agents_num = agents_count

        return model_selection[0], agents_num
    
    def llm_forward_textgrad(self,
        prev_graph,
        acc: float,
        edge_probs,
        his_response: list,
        model_selection: dict,
        graph,
        models):


        def convert_model_dict(idx_dict: Dict[Any, int]) -> Dict[str, int]:
            k0 = idx_dict.get("0", idx_dict.get(0, 0))
            k1 = idx_dict.get("1", idx_dict.get(1, 0))
            k2 = idx_dict.get("2", idx_dict.get(2, 0))
            k3 = idx_dict.get("3", idx_dict.get(3, 0))
            k4 = idx_dict.get("4", idx_dict.get(4, 0))
            k5 = idx_dict.get("5", idx_dict.get(5, 0))
            k6 = idx_dict.get("6", idx_dict.get(6, 0))
            k7 = idx_dict.get("7", idx_dict.get(7, 0))
            k8 = idx_dict.get("8", idx_dict.get(8, 0))
            k9 = idx_dict.get("9", idx_dict.get(9, 0))

            return {
                "llama:1b":  int(k0),
                "llama:3b":  int(k1),
                "llama:8b":  int(k2),
                "llama:70b": int(k3),
                "gemma:1b": int(k4),
                "gemma:2b": int(k5),
                "gemma:7b": int(k6),
                "qwen:1.5b": int(k7),
                "qwen:3b": int(k8),
                "qwen:7b": int(k9),
            }
        
        """def convert_model_dict(idx_dict: Dict[Any, int]) -> Dict[str, int]:
            k0 = idx_dict.get("0", idx_dict.get(0, 0))
            k1 = idx_dict.get("1", idx_dict.get(1, 0))
            k2 = idx_dict.get("2", idx_dict.get(2, 0))
            k3 = idx_dict.get("3", idx_dict.get(3, 0))
            k4 = idx_dict.get("4", idx_dict.get(4, 0))
            k5 = idx_dict.get("5", idx_dict.get(5, 0))
            k6 = idx_dict.get("6", idx_dict.get(6, 0))
            k7 = idx_dict.get("7", idx_dict.get(7, 0))
            k8 = idx_dict.get("8", idx_dict.get(8, 0))
            k9 = idx_dict.get("9", idx_dict.get(9, 0))

            return {
                "llama:1b":  int(k0),
                "llama:3b":  int(k1),
                "llama:8b":  int(k2),
                "llama:70b": int(k3),
                "gemma:1b": int(k4),
                "gemma:2b": int(k5),
                "gemma:7b": int(k6),
                "qwen:1.5b": int(k7),
                "qwen:3b": int(k8),
                "qwen:7b": int(k9),
            }"""
        
        def extract_graph_and_edges(response: str) -> Tuple[str, str]:
            m_graph = re.search(
                r"Graph:\s*(.*?)\s*Edge-probabilities:",
                response,
                flags=re.S
            )
            if not m_graph:
                raise ValueError("Cannot find 'Graph:' section in LLM response")
            graph_raw = m_graph.group(1).strip()

            # 去掉首尾包裹的引号（LLM 常用 "..." 返回）
            if (graph_raw.startswith('"') and graph_raw.endswith('"')) or \
            (graph_raw.startswith("'") and graph_raw.endswith("'")):
                graph_raw = graph_raw[1:-1]

            graph_str = graph_raw.replace("\\n", "\n")   # 若 LLM 转义了换行符

            # -------- 2. Edge-probabilities 段 --------
            m_edge = re.search(r"Edge-probabilities:\s*(.*)", response, flags=re.S)
            if not m_edge:
                raise ValueError("Cannot find 'Edge-probabilities:' section")
            edge_raw = m_edge.group(1).strip()

            try:
                edge_list = ast.literal_eval(edge_raw)
                # edge_list 可能是 list[str] 或 tuple[str]
                if isinstance(edge_list, (list, tuple)):
                    edge_probs_str = "".join(edge_list).replace("\\n", "\n")
                else:  # 解析不到 list，就保持原样
                    edge_probs_str = str(edge_list)
            except Exception:
                # 解析失败就直接返回原字符串
                edge_probs_str = edge_raw.replace("\\n", "\n")

            return graph_str.strip(), edge_probs_str.strip()
        
        model_selection = convert_model_dict(model_selection)
        total_model = sum(model_selection.values())
        message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS).",)]
        if his_response:                                     # 至少完成过 1 轮
            len_resp = len(his_response)                    # 已完成轮数
            len_hist = min(1, len_resp)                     # 最多取 3 轮

            # 2-A. 最近 ≤3 轮（从旧到新）user / assistant 对
            start_idx = len_resp - len_hist                 # 起始索引
            for i in range(start_idx, len_resp):
                user_prompt = self.prompt_set.get_llm_forward_textgrad(
                    prev_graph[i], acc[i], edge_probs[i], model_selection
                )
                message.append(Message(role="user",      content=user_prompt))
                message.append(Message(role="assistant", content=his_response[i]))

            # 2-B. 本轮 user prompt（graph 比 his_response 长 1）
            curr_idx = len(prev_graph) - 1
            curr_prompt = self.prompt_set.get_llm_forward_textgrad(
                prev_graph[curr_idx], acc[curr_idx], edge_probs[curr_idx], model_selection
            )
            message.append(Message(role="user", content=curr_prompt))

        else:
            prompt = self.prompt_set.get_llm_forward_textgrad(prev_graph[-1],acc[-1],edge_probs[-1],model_selection)
            message.append(Message(role="user", content=prompt))
        max_retries = 3
        price = 0  
        CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")        
        for attempt in range(max_retries):
            if total_model > 5:
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
            if "llama" in self.model_name:
                response = self.llm.gen(message,16384)
            else:
                response,price_i = self.llm.gen(message, 16384)
                price += price_i
            graph_str, edge_probs = extract_graph_and_edges(response)

            # 检查 node IDs 合法性
            valid_ids = {n for n in graph.nodes}
            found_ids = re.findall(r'Node\s+([A-Za-z0-9]+)\b', graph_str)
            bad_ids = [nid for nid in found_ids if nid not in valid_ids]

            if CHINESE_RE.search(graph_str) or CHINESE_RE.search(edge_probs):
                print(f"[Retry {attempt}/{max_retries}] Chinese characters detected, regenerating...")
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
                continue

            if not bad_ids:
                # 校验通过，退出循环
                break
            else:
                print(f"[Warning] Attempt {attempt+1}/{max_retries}: Invalid node IDs {bad_ids}, retrying...")

        else:
            # 如果循环结束还没 break，说明三次都失败
            raise ValueError(f"LLM produced invalid node IDs after {max_retries} attempts: {bad_ids}")

        
        graph.apply_graph_string(graph,graph_str,models)
        return graph, edge_probs, response, price
    
    def maao_forward(self,
        prev_graph,
        acc: float,
        edge_probs,
        role_probs):

        prompt = self.prompt_set.get_llm_forward_maao(prev_graph,acc,edge_probs,role_probs)
        message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=prompt)]
        CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")
        max_retries=3
        for attempt in range(max_retries):
            response = self.llm.gen(message, 16384)

            cost = None
            if isinstance(response, tuple):
                response, cost = response

            if CHINESE_RE.search(response):
                print(f"[Retry {attempt}/{max_retries}] Chinese characters detected, regenerating...")
                continue
            else:
                break

        return response
        
    def llm_forward_latency(self,
        prev_graph,
        acc: float,
        edge_probs,
        his_response: list,
        model_selection: dict,
        graph,
        models,
        latency_list):


        def convert_model_dict(idx_dict: Dict[Any, int]) -> Dict[str, int]:
            k0 = idx_dict.get("0", idx_dict.get(0, 0))
            k1 = idx_dict.get("1", idx_dict.get(1, 0))
            k2 = idx_dict.get("2", idx_dict.get(2, 0))
            k3 = idx_dict.get("3", idx_dict.get(3, 0))

            return {
                "1b":  int(k0),
                "3b":  int(k1),
                "8b":  int(k2),
                "70b": int(k3),
            }
        
        def extract_graph_and_edges(response: str) -> Tuple[str, str]:
            m_graph = re.search(
                r"Graph:\s*(.*?)\s*Edge-probabilities:",
                response,
                flags=re.S
            )
            if not m_graph:
                raise ValueError("Cannot find 'Graph:' section in LLM response")
            graph_raw = m_graph.group(1).strip()

            # 去掉首尾包裹的引号（LLM 常用 "..." 返回）
            if (graph_raw.startswith('"') and graph_raw.endswith('"')) or \
            (graph_raw.startswith("'") and graph_raw.endswith("'")):
                graph_raw = graph_raw[1:-1]

            graph_str = graph_raw.replace("\\n", "\n")   # 若 LLM 转义了换行符

            # -------- 2. Edge-probabilities 段 --------
            m_edge = re.search(r"Edge-probabilities:\s*(.*)", response, flags=re.S)
            if not m_edge:
                raise ValueError("Cannot find 'Edge-probabilities:' section")
            edge_raw = m_edge.group(1).strip()

            try:
                edge_list = ast.literal_eval(edge_raw)
                # edge_list 可能是 list[str] 或 tuple[str]
                if isinstance(edge_list, (list, tuple)):
                    edge_probs_str = "".join(edge_list).replace("\\n", "\n")
                else:  # 解析不到 list，就保持原样
                    edge_probs_str = str(edge_list)
            except Exception:
                # 解析失败就直接返回原字符串
                edge_probs_str = edge_raw.replace("\\n", "\n")

            return graph_str.strip(), edge_probs_str.strip()
        
        model_selection = convert_model_dict(model_selection)
        total_model = sum(model_selection.values())
        message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS).",)]
        if his_response:                                     # 至少完成过 1 轮
            len_resp = len(his_response)                    # 已完成轮数
            len_hist = min(1, len_resp)                     # 最多取 3 轮

            # 2-A. 最近 ≤3 轮（从旧到新）user / assistant 对
            start_idx = len_resp - len_hist                 # 起始索引
            for i in range(start_idx, len_resp):
                user_prompt = self.prompt_set.get_llm_forward_latency(
                    prev_graph[i], acc[i], edge_probs[i], model_selection,latency_list[i]
                )
                message.append(Message(role="user",      content=user_prompt))
                message.append(Message(role="assistant", content=his_response[i]))

            # 2-B. 本轮 user prompt（graph 比 his_response 长 1）
            curr_idx = len(prev_graph) - 1
            curr_prompt = self.prompt_set.get_llm_forward_latency(
                prev_graph[curr_idx], acc[curr_idx], edge_probs[curr_idx], model_selection,latency_list[curr_idx]
            )
            message.append(Message(role="user", content=curr_prompt))

        else:
            prompt = self.prompt_set.get_llm_forward_latency(prev_graph[-1],acc[-1],edge_probs[-1],model_selection,latency_list[-1])
            message.append(Message(role="user", content=prompt))
        max_retries = 3
        CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")        
        for attempt in range(max_retries):
            if total_model > 5:
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
            response = self.llm.gen(message, 16384)
            graph_str, edge_probs = extract_graph_and_edges(response)

            # 检查 node IDs 合法性
            valid_ids = {n for n in graph.nodes}
            found_ids = re.findall(r'Node\s+([A-Za-z0-9]+)\b', graph_str)
            bad_ids = [nid for nid in found_ids if nid not in valid_ids]

            if CHINESE_RE.search(graph_str) or CHINESE_RE.search(edge_probs):
                print(f"[Retry {attempt}/{max_retries}] Chinese characters detected, regenerating...")
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
                continue

            if not bad_ids:
                # 校验通过，退出循环
                break
            else:
                print(f"[Warning] Attempt {attempt+1}/{max_retries}: Invalid node IDs {bad_ids}, retrying...")

        else:
            # 如果循环结束还没 break，说明三次都失败
            raise ValueError(f"LLM produced invalid node IDs after {max_retries} attempts: {bad_ids}")

        
        graph.apply_graph_string(graph,graph_str,models)
        return graph, edge_probs, response
    
    def llm_forward_ablation_role(self,
        prev_graph,
        acc: float,
        edge_probs,
        his_response: list,
        model_selection: dict,
        graph,
        models):


        def convert_model_dict(idx_dict: Dict[Any, int]) -> Dict[str, int]:
            k0 = idx_dict.get("0", idx_dict.get(0, 0))
            k1 = idx_dict.get("1", idx_dict.get(1, 0))
            k2 = idx_dict.get("2", idx_dict.get(2, 0))
            k3 = idx_dict.get("3", idx_dict.get(3, 0))

            return {
                "1b":  int(k0),
                "3b":  int(k1),
                "8b":  int(k2),
                "70b": int(k3),
            }
        
        def extract_graph_and_edges(response: str) -> Tuple[str, str]:
            m_graph = re.search(
                r"Graph:\s*(.*?)\s*Edge-probabilities:",
                response,
                flags=re.S
            )
            if not m_graph:
                raise ValueError("Cannot find 'Graph:' section in LLM response")
            graph_raw = m_graph.group(1).strip()

            # 去掉首尾包裹的引号（LLM 常用 "..." 返回）
            if (graph_raw.startswith('"') and graph_raw.endswith('"')) or \
            (graph_raw.startswith("'") and graph_raw.endswith("'")):
                graph_raw = graph_raw[1:-1]

            graph_str = graph_raw.replace("\\n", "\n")   # 若 LLM 转义了换行符

            # -------- 2. Edge-probabilities 段 --------
            m_edge = re.search(r"Edge-probabilities:\s*(.*)", response, flags=re.S)
            if not m_edge:
                raise ValueError("Cannot find 'Edge-probabilities:' section")
            edge_raw = m_edge.group(1).strip()

            try:
                edge_list = ast.literal_eval(edge_raw)
                # edge_list 可能是 list[str] 或 tuple[str]
                if isinstance(edge_list, (list, tuple)):
                    edge_probs_str = "".join(edge_list).replace("\\n", "\n")
                else:  # 解析不到 list，就保持原样
                    edge_probs_str = str(edge_list)
            except Exception:
                # 解析失败就直接返回原字符串
                edge_probs_str = edge_raw.replace("\\n", "\n")

            return graph_str.strip(), edge_probs_str.strip()
        
        model_selection = convert_model_dict(model_selection)
        total_model = sum(model_selection.values())
        message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS).",)]
        if his_response:                                     # 至少完成过 1 轮
            len_resp = len(his_response)                    # 已完成轮数
            len_hist = min(1, len_resp)                     # 最多取 3 轮

            # 2-A. 最近 ≤3 轮（从旧到新）user / assistant 对
            start_idx = len_resp - len_hist                 # 起始索引
            for i in range(start_idx, len_resp):
                user_prompt = self.prompt_set.get_llm_forward_ablation_role(
                    prev_graph[i], acc[i], edge_probs[i], model_selection
                )
                message.append(Message(role="user",      content=user_prompt))
                message.append(Message(role="assistant", content=his_response[i]))

            # 2-B. 本轮 user prompt（graph 比 his_response 长 1）
            curr_idx = len(prev_graph) - 1
            curr_prompt = self.prompt_set.get_llm_forward_ablation_role(
                prev_graph[curr_idx], acc[curr_idx], edge_probs[curr_idx], model_selection
            )
            message.append(Message(role="user", content=curr_prompt))

        else:
            prompt = self.prompt_set.get_llm_forward_ablation_role(prev_graph[-1],acc[-1],edge_probs[-1],model_selection)
            message.append(Message(role="user", content=prompt))
        max_retries = 3
        CHINESE_RE = re.compile(r"[\u4e00-\u9fa5]")        
        for attempt in range(max_retries):
            if total_model > 5:
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
            response = self.llm.gen(message, 16384)
            graph_str, edge_probs = extract_graph_and_edges(response)

            # 检查 node IDs 合法性
            valid_ids = {n for n in graph.nodes}
            found_ids = re.findall(r'Node\s+([A-Za-z0-9]+)\b', graph_str)
            bad_ids = [nid for nid in found_ids if nid not in valid_ids]

            if CHINESE_RE.search(graph_str) or CHINESE_RE.search(edge_probs):
                print(f"[Retry {attempt}/{max_retries}] Chinese characters detected, regenerating...")
                if his_response:
                    message = [Message(role="system",content="You are a researcher specializing in multi-agent systems (MAS)."),
                            Message(role='user', content=curr_prompt)]
                continue

            if not bad_ids:
                # 校验通过，退出循环
                break
            else:
                print(f"[Warning] Attempt {attempt+1}/{max_retries}: Invalid node IDs {bad_ids}, retrying...")

        else:
            # 如果循环结束还没 break，说明三次都失败
            raise ValueError(f"LLM produced invalid node IDs after {max_retries} attempts: {bad_ids}")

        
        graph.apply_graph_string(graph,graph_str,models)
        return graph, edge_probs, response

    def model_initialize_latency(self,budget,model_cost):

        """def extract_json_dict(text: str):
            # 去掉前后空格
            text = text.strip()

            # 如果是 markdown 格式 ```json ... ```
            if text.startswith("```"):
                # 去掉三引号包裹
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)

            # 用正则提取 { ... } 部分
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text}")

            return json.loads(match.group())"""

        def extract_json_dict(text: str):
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)

            # 提取所有 JSON 对象
            matches = re.findall(r"\{.*?\}", text, re.DOTALL)
            if not matches:
                raise ValueError(f"No JSON object found in response: {text}")

            json_objects = []
            for json_str in matches:
                try:
                    json_objects.append(json.loads(json_str))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON found: {json_str}\nError: {e}")

            # 如果只有一个对象，返回它本身；否则返回列表
            return json_objects
        
        model_combo = self.enumerate_model_combos(budget,model_cost)
        prompt = self.prompt_set.get_model_initialize_latency(model_combo)
        message = [Message(role="system", content=f"You are a researcher specializing in multi-agent systems (MAS)."),Message(role="user", content=prompt)]
        response = self.llm.gen(message,16384)
        model_selection = extract_json_dict(response)
        agents_num = 0
        for dict_m in model_selection:
            agents_count = sum(dict_m.values())
            if agents_count > agents_num:
                agents_num = agents_count

        return model_selection, agents_num
    
    def optimize_workflow_code(self, optimize_prompt):
        messages = [
            Message(role="system", content="You are an expert workflow code optimizer."),
            Message(role="user", content=optimize_prompt),
        ]
        resp, _ = self.llm.gen(messages)
        return str(resp)
    
    def reflect_workflow_code(self, optimize_prompt):
        messages = [
            Message(role="system", content="You are an expert workflow code optimizer."),
            Message(role="user", content=optimize_prompt),
        ]
        resp, _ = self.llm.gen(messages)
        return str(resp)

    
    