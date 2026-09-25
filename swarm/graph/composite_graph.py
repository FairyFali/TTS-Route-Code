import random
from collections import Counter
from typing import Iterable, Tuple, Dict, Hashable, Optional, Union, List
import re,ast

from swarm.graph import Graph
from swarm.graph.node import Node
from swarm.llm import LLMRegistry
from swarm.llm.format import Message
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.optimizer.role_optimizer.role import Role
from swarm.environment.operations import DirectAnswer


class CompositeGraph(Graph):
    """
    The composite graph is a graph that contains other agents as sub-graphs.
    """

    def __init__(self,
                 decision_method: Node,
                 domain: str,
                 model_name: Optional[str] = None,
                 ):
        super().__init__(domain, model_name)
        self.decision_method = decision_method
        self.domain = domain
        self.model_name = model_name
        self.graphs = []
        self.output_nodes = [decision_method]
        self.add_node(self.decision_method)  # 把node加入到self.nodes里面，保证node有id
        self.prompt_set = PromptSetRegistry.get(domain)
        

    def add_graph(self, graph):

        for node in graph.nodes.values():
            # We move the check cycle to parameterization.py
            # if self.check_cycle(node):
            #     #raise Exception(f"Adding node {node.id} would cause a cyclic dependency.")
            self.add_node(node)

        self.graphs.append(graph)
        graph.memory = self.memory
        self.input_nodes.extend(graph.input_nodes)

    def build_graph(self):
        pass
        # for decision_node in self.decision_nodes:
        #     for output_node in self.output_nodes:
        #         output_node.add_successor(decision_node)

    def set_roles(self, roles: dict[int, int]) -> None:
        """
        将一次采样得到的 role 分配写回各 Node 对象。
        若想把 IO / REV 做成两种不同的类，也可以在这里动态 wrap。
        """
        for node_id, role_idx in roles.items():
            node = self.find_node(node_id)
            if node is None:
                continue
            node.role_idx = role_idx          # 存一个整数
            node.role_name = Role(role_idx).name    # 可读性更好

            if isinstance(node, DirectAnswer):
                if node.role_name == 'REV':
                    node.behavior_fn = node.review_behavior
                    node.original_fn = node.review_behavior
                else:  # Role.IO 或其他默认
                    node.behavior_fn = node.io_behavior

    def set_models(
        self,
        models: dict[int, str],           # {node_id: "3b"/"8b"/"70b"}
        default_model: str = "3b",
        model_name_map: dict[str, str] | None = None,
    ) -> None:
        for node_id, node in self.nodes.items():
            # ① 取到本节点该用的 model tag
            model_tag = models.get(node_id, default_model)
            node.model_size = model_tag   # ——> 存一个可读标记

            # ② 如需要，替换它内部用的 LLM（仅对 DirectAnswer 之类有效）
            if model_name_map is not None and model_tag in model_name_map:
                new_llm_name = model_name_map[model_tag]
                if isinstance(node, DirectAnswer):
                    # 只在确实换了模型时重新取，否则沿用之前的实例
                    if node.model_name != new_llm_name:
                        node.model_name = new_llm_name
                        node.llm = LLMRegistry.get(new_llm_name)

    def set_IO(self,roles):
        for node_id, role_idx in roles.items():
            node = self.find_node(node_id)
            if node is None:
                continue
            node.role_idx = 0          # 存一个整数
            node.role_name     = Role(node.role_idx).name    # 可读性更好

            if isinstance(node, DirectAnswer):
                if node.role_name == 'REV':
                    node.behavior_fn = node.review_behavior
                else:  # Role.IO 或其他默认
                    node.behavior_fn = node.io_behavior
    
    def set_IO_graph(self):
        for node_id in self.nodes:
            node = self.find_node(node_id)
            if node is None:
                continue
            node.role_idx = 0          # 存一个整数
            node.role_name     = Role(node.role_idx).name    # 可读性更好

            if isinstance(node, DirectAnswer):
                if node.role_name == 'REV':
                    node.behavior_fn = node.review_behavior
                else:  # Role.IO 或其他默认
                    node.behavior_fn = node.io_behavior

    
    def init(self, init_connection_probability, potential_connections):
        self.learned_connections = []
        for connection in potential_connections:
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if random.random() < init_connection_probability and not self.check_cycle(in_node, {out_node}, set()):
                self.learned_connections.append(connection)
                out_node.add_successor(in_node)

    def mutate(self, max_new_edges, max_remove_edges, potential_connections):
        # Add new edges
        num_new_edges = random.randint(0, max_new_edges)
        num_remove_edges = random.randint(1 if num_new_edges == 0 else 0, max_remove_edges)
        new_edge_count = 0
        for _ in range(num_new_edges * 10):
            connection = random.choice(potential_connections)
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if not self.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                new_edge_count += 1
                self.learned_connections.append(connection)
            if new_edge_count >= num_new_edges:
                break
        # Remove edges
        remove_edge_count = 0
        for _ in range(num_remove_edges * 10):
            if len(self.learned_connections) == 0:
                break
            connection = random.choice(self.learned_connections)
            out_node, in_node = connection
            out_node = self.nodes[out_node]
            in_node = self.nodes[in_node]
            if in_node in self.output_nodes and len(in_node.predecessors) == 1:
                continue
            out_node.remove_successor(in_node)
            remove_edge_count += 1
            self.learned_connections.remove(connection)
            if remove_edge_count >= num_remove_edges:
                break

    def check_cycle(self, new_node, target_nodes, visited=None, rec_stack=None):
        if new_node in target_nodes:
            return True
        for successor in new_node.successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False
    
    def check_role_constraints(
            self,
            roles: Dict[int, int],
            io_idx: int,
            reviewer_idx: int
        ) -> bool:
        """
        roles: {node_id: role_idx}
        io_idx / reviewer_idx: 角色编号（如 0=IO, 1=Reviewer）

        返回 True 表示角色分配满足全部约束
        """
        def is_useful(node):
            if node in self.output_nodes:
                return True
            
            for successor in node.successors:
                if is_useful(successor):
                    return True
            return False

        useful = {nid for nid,n in self.nodes.items() if is_useful(n)}
        # --- R-3: 至少一个 IO ---
        if io_idx not in roles.values():
            return False

        for node_id, role in roles.items():
            node = self.find_node(node_id)
            useful_predecessors = [p for p in node.predecessors if p.id in useful]

            # 若当前节点是 Reviewer -> 检查其前驱数量与角色
            if role == reviewer_idx:
                io_parents = [
                    p for p in useful_predecessors
                    if roles.get(p.id, None) == io_idx
                ]
                # --- R-1 ---
                if len(io_parents) < 1:
                    return False

                # --- R-2 ---
                for p in useful_predecessors:
                    if roles.get(p.id, None) == reviewer_idx:
                        return False
        return True
    
    def delete_nodes(self, node_ids) -> None:
        """
        原地删除节点及其相关边（潜在连接不会删除，只用于推理时剔除）
        """
        for nid in node_ids:
            node = self.nodes.pop(nid, None)
            if node is None:
                continue
            # 从前驱 / 后继里移除
            for pred in list(node.predecessors):
                pred.successors = [s for s in pred.successors if s.id != nid]
            for succ in list(node.successors):
                succ.predecessors = [p for p in succ.predecessors if p.id != nid]

    def delete_nodes_hard(self, node_ids) -> None:
        for nid in node_ids:
            node = self.nodes.pop(nid, None)
            if node is None:
                continue
            # 从前驱 / 后继里移除
            for pred in list(node.predecessors):
                pred.successors = [s for s in pred.successors if s.id != nid]
            for succ in list(node.successors):
                succ.predecessors = [p for p in succ.predecessors if p.id != nid]
        self.graphs = [g for g in self.graphs if g.output_nodes[0].id not in node_ids]


    def graph_str(self,graph,kept_ids):
        """
        将一张 CompositeGraph 转成适合喂给 LLM 的简洁字符串。
        每行一个节点，格式示例：
            Node 3  | model=llama-3b | role=IO | preds=[1,2] | succs=[7]
        """
        lines = []
        edge_cnt = 0
        # 按节点 id 排序，确保输出顺序稳定
        for nid in sorted(kept_ids):
            node = graph.find_node(nid)

            model = getattr(node, "model_name", "UNK")
            if model == 'UNK':
                model = 'FinalDecision'
            if node.node_name == "FinalDecision":
                model = 'FinalDecision'
            role  = getattr(node, "role_idx", "UNK")
            if role == 0:
                role = 'IO'
            else:
                role = 'REV'

            preds = sorted(getattr(node, "predecessors", []), key=lambda n: n.id)
            succs = sorted(getattr(node, "successors",   []), key=lambda n: n.id)

            pred_ids = [p.id for p in preds]
            succ_ids = [s.id for s in succs]
            edge_cnt += len(succs)

            line = (
                f"Node {nid:>2} | model={model} | role={role} | "
                f"preds={pred_ids} | succs={succ_ids}"
            )
            lines.append(line)
        return "\n".join(lines)
    

    def apply_graph_string(self,
                graph,
                graph_str: str,
                models_list = [
    "llama3.2-1b-longcontext:latest",
    "llama3.2-3b-longcontext:latest",
    "llama3.1-8b-longcontext:latest",
    "llama3.1-70b-longcontext:latest"
]
            ):
            # ---------- 1. 解析字符串 → dict ----------
            node_info: Dict[str, Dict] = {}  # {nid: {"model":..,"role":..,"preds":[..],"succs":[..]}}
            for line in graph_str.strip().splitlines():
                line = line.strip().strip('"').strip("'")   # 去首尾引号
                if not line or not line.startswith("Node"):
                    continue

                # Node 3CoH | model=llama... | role=IO | preds=['4Dhq'] | succs=[]
                parts = [p.strip() for p in line.split("|")]
                nid = parts[0].split()[1]

                kv = {}
                for seg in parts[1:]:
                    k, v = seg.split("=", 1)
                    kv[k.strip()] = v.strip()

                # preds / succs 解析为 list
                preds = ast.literal_eval(kv.get("preds", "[]"))
                succs = ast.literal_eval(kv.get("succs", "[]"))

                node_info[nid] = dict(
                    model=kv.get("model", "UNK"),
                    role=kv.get("role", "IO"),
                    preds=preds,
                    succs=succs,
                )

            # ---------- 3. 写入模型 / 角色 ----------
            name2idx = {m: i for i, m in enumerate(models_list)}
            for nid, info in node_info.items():
                node = graph.find_node(nid)
                if node is None:
                    continue  # 若字符串里有模板之外的节点，可选择 raise
                mname = info["model"]
                if mname != "UNK" and mname != 'FinalDecision':
                    node.model_name = mname
                    node.model_idx  = name2idx.get(mname, 0)
                    if hasattr(node, "llm"):
                        node.llm = LLMRegistry.get(mname)
                # 角色

                node.role_idx = 0 if info["role"] == 'IO' else 1
                node.role_name = Role(node.role_idx).name    # 可读性更好
                if isinstance(node, DirectAnswer):
                    if node.role_name == 'REV':
                        node.behavior_fn = node.review_behavior
                        node.original_fn = node.review_behavior
                    else:  # Role.IO 或其他默认
                        node.behavior_fn = node.io_behavior

            for nid, info in node_info.items():
                dst_node = graph.find_node(nid)
                for pred_id in info["preds"]:
                    src_node = graph.find_node(pred_id)
                    if not src_node or not dst_node:
                        continue

                    has_cycle = graph.check_cycle(new_node=dst_node,             
                                                target_nodes={src_node},)
                    if has_cycle:
                        continue 
                    src_node.add_successor(dst_node)
                    dst_node.add_predecessor(src_node)

            return graph
    
    def flops_cost(
    self,
    model_cfg = {
        "llama3.2-1b-longcontext:latest":  {"M": 1_000_000_000,  "L": 16, "D": 2048},
        "llama3.2-3b-longcontext:latest":  {"M": 3_000_000_000,  "L": 28, "D": 3072},
        "llama3.1-8b-longcontext:latest":  {"M": 8_000_000_000,  "L": 32, "D": 4096},
        "llama3.1-70b-longcontext:latest": {"M": 70_000_000_000, "L": 80, "D": 8192},
    },
    nd_override: Optional[Dict[Hashable, int]] = None,
    sep_tokens_per_edge: int = 0,
    return_per_node: bool = False,
) -> Union[Dict[str, float], Tuple[float, Dict[Hashable, float]]]:
        
        if self.domain == 'math':
            A, B = 202, 275
            base_flop = 9.710e+11
            base_multi = 12.698
        elif self.domain == 'mmlu':
            A, B = 213, 215
            base_flop = 8.680e+11
            base_multi = 12.698

        nodes = [node for node in self.nodes]
        edges = set()
        node_model_map = {node.id: node.model_name           # key  : value
                  for node in self.nodes.values()} 
        for node in self.nodes.values():
            for pre in node.predecessors:
                edges.add((pre.id,node.id))
            for suc in node.successors:
                edges.add((node.id,suc.id))

        edges = list(edges)

        nd_override = nd_override or {}

        indeg = Counter(v for _, v in edges)
        for n in nodes:
            indeg.setdefault(n, 0)

        per_node = {}
        total = 0.0

        for n in nodes:
            model_name = node_model_map[n]
            cfg = model_cfg[model_name]
            M = float(cfg["M"])
            L = float(cfg["L"])
            D = float(cfg["D"])

            d_in = indeg[n]
            Np = A + d_in * (B + sep_tokens_per_edge)
            Nd = int(nd_override.get(n, B))

            f_prefill = 2.0 * M * Np + 2.0 * L * D * Np * (Np + 1)
            f_decode = 2.0 * M * Nd + 2.0 * L * D * Nd * (2 * Np + Nd + 1)
            f_node = f_prefill + f_decode

            per_node[n] = f_node
            total += f_node

        if return_per_node:
            return total, per_node
        multi = total/base_flop
        return {"total_flops": total},False if multi > base_multi else True

    """def random_set_models(
            self,
            model_counts: Dict[str, int],
            model_name_map: Dict[str, str] | None = None,
        ):
            if model_name_map is None:
                model_name_map = {
                    "1b": "llama3.2-1b-longcontext:latest",
                    "3b":  "llama3.2-3b-longcontext:latest",
                    "8b":  "llama3.1-8b-longcontext:latest",
                    "70b": "llama3.1-70b-longcontext:latest",
                    "p3b": "gemma1-2b-longcontext:latest"
                }
                reverse_model_map = {v: k for k, v in model_name_map.items()}
                size_order = {"1b": 0, "3b": 1, "8b": 2, "p3b": 1, "70b": 4}
                reverse_size_map = {v: k for k, v in size_order.items()}
                

            all_nodes: List["Node"] = []
            for agent in self.graphs:     
                all_nodes.extend(agent.nodes.values())   

            assert sum(model_counts.values()) == len(all_nodes), \
                "model_counts 总和必须与节点数一致"

            random.shuffle(all_nodes)

            start = 0
            for size_key, k in model_counts.items():
                tag = model_name_map[size_key]
                for node in all_nodes[start:start+k]:
                    node.model_size = size_key
                    node.model_name = tag
                    if hasattr(node, "llm"):
                        node.llm = LLMRegistry.get(tag)
                start += k

            _,ok = self.flops_cost()

            if ok:
                return ok,'',''
            else:
                node_model_map = {
            nid: node.model_name
            for nid, node in self.nodes.items()
            if nid != self.decision_method.id          # 不能删决策节点
        }
            nid_size_map = {
        nid: size_order[reverse_model_map[model_name]]
        for nid, model_name in node_model_map.items()
    }

            min_size_val = min(nid_size_map.values())          # 最小规模值
            smallest_nids = [nid for nid, v in nid_size_map.items()
                            if v == min_size_val]
            del_model = reverse_size_map.get(min_size_val)
            model_counts[del_model] = model_counts.get(del_model,0) - 1

            chosen_nid = random.choice(smallest_nids)
            return ok,chosen_nid,model_counts"""
            
    def random_set_models(
            self,
            model_counts: Dict[str, int],
            model_name_map: Dict[str, str] | None = None,
        ):
            """model_name_map = {
                    "1b": "llama3.2-1b-longcontext:latest",
                    "3b":  "llama3.2-3b-longcontext:latest",
                    "8b":  "llama3.1-8b-longcontext:latest",
                    "70b": "llama3.1-70b-longcontext:latest",
                    "q1b": "qwen2.5-1.5b-longcontext:latest",
                    "q3b": "qwen2.5-3b-longcontext:latest",
                    "q7b": "qwen2.5-7b-longcontext:latest",
                }"""
            model_name_map = {
                    "1b": "qwen2.5-1.5b-longcontext:latest",
                    "3b":  "qwen2.5-3b-longcontext:latest",
                    "8b":  "qwen2.5-7b-longcontext:latest",
                    "70b": "llama3.1-70b-longcontext:latest",
                    "g1b": "gemma3-1b-longcontext:latest",
                    "g3b": "gemma1-2b-longcontext:latest",
                    "g7b": "gemma1-7b-longcontext:latest",
                }
            all_nodes: List["Node"] = []
            for agent in self.graphs:     
                all_nodes.extend(agent.nodes.values())   

            assert sum(model_counts.values()) == len(all_nodes), \
                "model_counts 总和必须与节点数一致"

            # random.shuffle(all_nodes)

            start = 0
            for size_key, k in model_counts.items():
                tag = model_name_map[size_key]
                for node in all_nodes[start:start+k]:
                    node.model_size = size_key
                    node.model_name = tag
                    if hasattr(node, "llm"):
                        node.llm = LLMRegistry.get(tag)
                start += k
            
            return
            
