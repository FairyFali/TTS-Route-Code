from transformers import AutoModel, AutoTokenizer
import torch, os
from swarm.llm.llm_registry import LLMRegistry

@LLMRegistry.register('PRM')
class PRMLLM:
    """
    仅做推断，不做 generate。内部持有 tokenizer & model。
    """
    _GLOBAL_CACHE = {}      # {model_path: (tokenizer, model)}

    def __init__(self, model_path: str, device: str = "cuda:2"):
        self.model_path = os.path.abspath(model_path)

        # ---- 全局缓存: 若已加载则直接复用 ----
        if self.model_path in PRMLLM._GLOBAL_CACHE:
            self.tokenizer, self.model = PRMLLM._GLOBAL_CACHE[self.model_path]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                device_map=device,           # 固定显卡，防止自动 shard
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()

            PRMLLM._GLOBAL_CACHE[self.model_path] = (self.tokenizer, self.model)

    def __deepcopy__(self, memo):
        # 深拷贝时直接返回 self，避免复制 GPU 权重
        return self
