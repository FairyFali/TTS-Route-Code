import glob
import pandas as pd
from typing import Dict, Union, List, Literal
import numpy as np
import os
import json
import re

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput


def load_dataset(directory, span=1):
    data = {'question':[], 'level':[], 'type':[], 'solution':[]}  # List to store all data from JSON files
    i = 0
    # Walk through each directory and subdirectory
    # NOTE: os.walk yields directories and files in filesystem order, which is NOT
    # stable across machines or over time. Because questions are addressed downstream
    # by positional index (qid = "math|<idx>"), an unsorted walk silently re-binds every
    # qid to a different problem, making runs non-reproducible and making results keyed
    # by qid non-mergeable across runs. Sorting both levels pins the mapping.
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        # layer by layer, find the dirs and files
        for file in sorted(files):
            if file.endswith('.json'):
                if i < span-1:
                    i += 1
                    continue
                i = 0
                file_path = os.path.join(root, file)  # Full path to the file
                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    data['question'].append(content.get('problem'))
                    data['level'].append(content.get('level'))
                    data['type'].append(content.get('type'))
                    data['solution'].append(content.get('solution'))

    return data


def extract_boxed_answers(text: str) -> List[str]:
    r"""Return the contents of every ``\boxed{...}`` in ``text``.

    Unlike a simple ``\\boxed{([^}]*)}`` regex, this walks the braces so that
    nested groups such as ``\boxed{-\frac{1}{8}}`` are captured in full
    (-> ``-\frac{1}{8}``) instead of being truncated at the first ``}``.
    """
    results: List[str] = []
    marker = r"\boxed{"
    start = text.find(marker)
    while start != -1:
        i = start + len(marker)
        depth = 1
        content_start = i
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        results.append(text[content_start:i])
        start = text.find(marker, i)
    return results

class MATHDataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['train'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/MATH/{self._split}/"
        self.data = load_dataset(data_path, span=10)
        self._total_df: pd.DataFrame = pd.DataFrame.from_dict(self.data)

        print("Total number of questions: ", len(self))

    @staticmethod
    def get_domain() -> str:
        return 'math'

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> Dict:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        # if len(answer) > 0:
        #     answer = answer[0] # Try to format the answer by taking the first letter

        matches = extract_boxed_answers(answer)
        return matches[-1] if matches else ""


    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['solution']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        matches = extract_boxed_answers(correct_answer)
        return matches[0] if matches else ""
