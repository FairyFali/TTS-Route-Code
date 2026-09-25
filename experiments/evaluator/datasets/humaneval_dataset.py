from typing import Dict, Any, List, Union, Literal
import pandas as pd
import json
from pathlib import Path
import re
import random

from experiments.evaluator.datasets.base_dataset import (
    BaseDataset, SwarmInput
)
from swarm.environment.tools.reader.readers import JSONLReader
from swarm.environment.tools.coding.python_executor import PyExecutor

random.seed(0)


def _load_jsonl(path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

class HumanEvalDataset(BaseDataset):
    """
    Wrapper for the Human-Eval benchmark (Chen et al., 2021).

    Each record is one coding task with:
        - prompt : function stub + docstring
        - test   : unit-test string that must pass
    """

    def __init__(
        self,
        split: Union[Literal['train'], Literal['val'], Literal['test']] = 'val',
    ) -> None:

        self._split = split          # kept for API symmetry
        jsonl_path = f"datasets/humaneval/{self._split}.jsonl"
        self._records: List[Dict[str, Any]] = JSONLReader.parse_file(jsonl_path)

        print(f"[HumanEval] total samples: {len(self)}  (split = {self._split})")

    def extract_example(self, text: str) -> list:
        examples = []

        # ---------- 1. 处理 docstring 里的 `>>>` ----------
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pattern = re.compile(
            r"""^assert\s+          # assert 关键字
                candidate\s*\((.*?)\)\s*  # candidate(<args>)
                ==\s*(.+)           # == <expected>
            """, re.VERBOSE)
        for ln in lines:
            m = pattern.match(ln)
            if m:
                args, expected = m.groups()
                examples.append(f"assert candidate({args}) == {expected}")

        return [random.choice(examples)]
    
    # ---------- required by BaseDataset ----------
    @staticmethod
    def get_domain() -> str:
        return "humaneval"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]

    # ---------- Swarm interface ----------
    @staticmethod
    def record_to_swarm_input(record: Dict[str, Any]) -> SwarmInput:
        return {
            "task":  record["prompt"],   # model needs to generate code here
            "tests": record["test"]      # used later by PyExecutor
        }

    def postprocess_answer(self, response: Union[str, List[str]],inputs) -> str:
        test = inputs["tests"]
        internal_tests = self.extract_example(test)
        if not isinstance(response,list):
            response = [response]
        match = re.search(r"```python\s*(.*?)```", response[0], re.DOTALL)
        if not match:
            return False
        response = match.group(1).strip()
        pattern = re.compile(r'def\s+(\w+)\s*\(')
        m = pattern.search(response)
        if not m:
            return False
        fn_name = m.group(1)      
        internal_tests = [re.sub(r'\bcandidate\b', fn_name, t) for t in internal_tests]
        is_solved, feedback, _ = PyExecutor().execute(response, internal_tests, timeout=10)
        return is_solved

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        """
        “正确答案” = 官方测试脚本本身 —— 在 Evaluator 中会调用
        PyExecutor().execute(generated_code, [tests])
        """
        return True

    # ---------- metric type ----------
    @staticmethod
    def get_metric() -> str:
        """
        告诉 Evaluator 采用 CodePass 而不是 Accuracy
        """
        return "code_pass"