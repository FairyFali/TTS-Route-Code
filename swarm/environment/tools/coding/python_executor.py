#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import *
import traceback
import subprocess
import tempfile
import os
import sys

from swarm.environment.tools.coding.executor_utils import function_with_timeout
from swarm.environment.tools.coding.executor_types import ExecuteResult, Executor
from swarm.utils.log import logger


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):

            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests.append(tests[i])
            except Exception:
                # output = get_output(func, tests[i], timeout=timeout)
                # failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        # feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)
    
    def function_execute(
        self,
        func: str,                 # LLM 生成的代码（包含 class Solution 等）
        test_cases: List[Dict],    # [{'input': '..."', 'output': '0', 'testtype': 'functional'}, ...]
        fn_name: str,              # 例如 "minimumChanges"
        timeout: int = 5,
    ) -> ExecuteResult:
        """
        专门用于 LiveCodeBench 的 LeetCode-style functional 测试：
        - 模型代码里定义了 `class Solution: def fn_name(self, ...)`
        - 每个测试用例的 input 是按行给出的参数字符串，如 '"aabbaa"\\n3'
        - output 是一个合法的 Python 表达式字符串，如 '0'、'"YES"' 等
        """

        success_tests = []
        failed_tests = []
        is_passing = True

        for case in test_cases:
            raw_inp = case["input"]              # 如 '"aabbaa"\n3'
            expected = case["output"].strip()    # 如 '0'

            # 把多行 input 变成函数参数列表："aabbaa", 3
            args = ", ".join(raw_inp.splitlines())

            # 构造完整的可执行代码：
            #   - 先放 LLM 给出的 func（class Solution 等）
            #   - 然后自己实例化 Solution 并断言
            imports = 'from typing import *'
            """test_code = (
                f"{imports}\n"
                f"{func}\n"
                "solution = Solution()\n"
                f"assert solution.{fn_name}({args}) == {expected}\n"
            )"""
            test_code = (
    f"{imports}\n"
    f"{func}\n"
    "solution = Solution()\n"
    f"_actual_ = solution.{fn_name}({args})\n"
    f"_expected_ = {expected}\n"
    "if _actual_ != _expected_:\n"
    "    raise AssertionError(f'got {_actual_!r}, expected {_expected_!r}')\n"
)
            feedback = dict()
            try:
                # 为每个用例用独立的环境，避免变量串台
                local_env = {}
                function_with_timeout(exec, (test_code, local_env), timeout)
                success_tests.append(raw_inp)
            except Exception as e:
                is_passing = False
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                failed_tests.append(
                    f"INPUT:\n{raw_inp}\nEXPECTED: {expected}\nERROR: {err}"
                )
                feedback = {"test_fail_case":{"test_case":case,"traceback":f"{err}"}}

        # 对每个原始 test_case 给一个 True/False 状态
        state = [c["input"] in success_tests for c in test_cases]

        return is_passing, feedback, tuple(state)

    def stdin_execute(
        self,
        code: str,
        tests: List[Dict],     # [{'input': str, 'output': str, 'testtype': 'stdin'}, ...]
        timeout: int = 5,
    ) -> ExecuteResult:
        """
        用于 LiveCodeBench 中 testtype == 'stdin' 的题目。

        code:  LLM 生成的完整 python 程序（通过 input() 读入，通过 print() 输出）
        tests: 每个元素是一个 dict，包含:
               - 'input':  作为 stdin 传给程序的字符串
               - 'output': 期望的 stdout
               - 'testtype': 必须是 'stdin'
        """

        # 1. 把模型代码写入临时文件
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            success_tests = []
            failed_tests = []
            is_passing = True
            state_flags = []

            feedback = dict()

            # 2. 遍历每个 stdin 测试用例
            for case in tests:
                inp = case["input"]
                expected = case["output"]

                try:
                    proc = subprocess.run(
                        [sys.executable, tmp_path],
                        input=inp,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    is_passing = False
                    state_flags.append(False)
                    failed_tests.append(
                        f"INPUT:\n{inp}\nEXPECTED:\n{expected}\nERROR: Timeout (>{timeout}s)"
                    )
                    feedback = {"test_fail_case":{"test_case":case}}
                    continue

                stdout = proc.stdout
                # 为了避免换行/空格差异，做个简单归一化
                got = stdout.strip()
                exp = expected.strip()

                if got == exp:
                    success_tests.append(inp)
                    state_flags.append(True)
                else:
                    is_passing = False
                    state_flags.append(False)
                    failed_tests.append(
                        f"INPUT:\n{inp}\nEXPECTED:\n{expected!r}\nGOT:\n{stdout!r}"
                    )
                    feedback = {"test_fail_case":{"test_case":case}}

            return is_passing, feedback, tuple(state_flags)

        finally:
            # 3. 清理临时文件
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False
        