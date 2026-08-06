from abc import ABC, abstractmethod
import json
import os
import re
import types
import unittest
# from llms import LLMInterface, OpenAI_LLM


def preprocess_string(input_string, lg):
    if f"```{lg}\n" in input_string:
        input_string = input_string[input_string.find(
            f"```{lg}\n") + len(f"```{lg}\n"):]
        input_string = input_string[:input_string.find("```")-1]
    elif "```" in input_string:
        input_string = input_string[input_string.find("```") + 4:]
        input_string = input_string[:input_string.find("```")-1]

    return input_string


def modify_test_code(original_code: str) -> str:
    # 用于存储修改后的代码行
    modified_code_lines = []
    # 逐行遍历原始代码
    lines = original_code.splitlines()
    for line in lines:
        # 如果是assert语句
        if line.strip().startswith("assert"):
            if line.strip().endswith("["):
                return """
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    unit_test_results = []
    try:
        assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    except AssertionError:
        unit_test_results.append('''Assertion failed for: assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']''')
    try:
        assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    except AssertionError:
        unit_test_results.append('''Assertion failed for: assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']''')
    try:
        assert candidate('(()(())((())))') == ['(()(())((())))']
    except AssertionError:
        unit_test_results.append('''Assertion failed for: assert candidate('(()(())((())))') == ['(()(())((())))']''')
    try:
        assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
    except AssertionError:
        unit_test_results.append('''Assertion failed for: assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']''')
    return unit_test_results
"""
            # 获取当前行的缩进
            indent = len(line) - len(line.lstrip())
            # 构造修改后的代码行，添加try-except语句
            modified_code_lines.append(f"{' ' * indent}try:")
            modified_code_lines.append(f"{' ' * indent}    {line.strip()}")
            modified_code_lines.append(
                f"{' ' * indent}except AssertionError:")
            modified_code_lines.append(
                f"{' ' * indent}    unit_test_results.append(''' Assertion failed for: {line.strip()} ''')")
        else:
            modified_code_lines.append(line)
            if line.strip().startswith("def check"):
                modified_code_lines.append("    unit_test_results = []")
    modified_code_lines.append("    return unit_test_results[:3]")
    # 将修改后的代码行合并为字符串并返回
    return "\n".join(modified_code_lines)


class FunctionalTestAgentInterface(ABC):
    # @abstractmethod
    # def generate_test_case(self, user_requirements: str, code: str) -> str:
    #     """
    #     使用LLM生成测试用例
    #     :param user_requirements: 用户需求
    #     :param code:已生成的代码
    #     :return: 生成的测试用例
    #     """
    #     pass

    @abstractmethod
    def run_tests(self, code: str, test_code: str) -> tuple:
        """
        运行生成的测试用例
        :param code: 需要测试的Python代码
        :param test_code: 生成的测试用例代码
        :return: 如果所有测试通过返回True，否则返回False，如果是False，一并返回错误信息
        """
        pass


class LLMFunctionalTestAgent(FunctionalTestAgentInterface):
    def __init__(self, entry, llm):
        self.entry = entry
        self.llm = llm


    def run_tests(self, code: str, test_code: list, humaneval=False) -> tuple:
        """
        运行生成的测试用例
        """
        if humaneval:
            # 提取被测函数
            scope = {}
            try:
                exec(code, scope)
                func_to_test = [obj for obj in scope.values(
                ) if isinstance(obj, types.FunctionType)][-1]
            except Exception as e:
                return False, str(e)
            # 提取测试函数
            exec(test_code, scope)
            check = [obj for obj in scope.values(
            ) if isinstance(obj, types.FunctionType)][-1]
            # 运行测试
            try:
                test_result = check(func_to_test)
                if len(test_result) == 0:
                    return True, None
                else:
                    return False, '\n'.join(test_result)
            except Exception as e:
                return False, str(e)
        else:
            results = []
            scope = {}
            try:
                exec(code, scope)
                func_to_test = [obj for obj in scope.values(
                ) if isinstance(obj, types.FunctionType)][-1]
            except Exception as e:
                return False, str(e)
            for test_case in test_code:
                test_input = test_case['input']
                expected_output = test_case['expected_output']
                try:
                    # 如果返回结果不等于预期输出，则记录失败
                    result = func_to_test(test_input)
                    if result != expected_output:
                        results.append(
                            f"Error: Test input is: {test_input}, got {result}")
                    else:
                        results.append(True)
                except Exception as e:
                    # 捕获并记录错误信息
                    results.append(f"Error: " + str(e))
            for r in results:
                if r != True:
                    return False, str([x for x in results if not x == True])
            return True, None

