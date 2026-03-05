import re
from llms import LLMInterface



class LLMParsingAgent():
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def extract_test_results(self, test_output: str) -> dict:
        """
        提取功能测试的有效信息，包括测试是否通过与失败原因
        """
        prompt = f'''Please provide a concise summary of the following unit test results. Identify any failures, errors, or issues mentioned in the output below, and explain them in simple language.
Unit Test Output:
{test_output}
'''
        result = self.llm.generate(prompt)
        return result

    def extract_static_analysis_results(self, static_analysis_output) -> list:
        """
        提取静态分析工具输出中的安全问题和代码质量问题
        """
        codeql_issues = str(static_analysis_output[0])
        bandit_issues = str(static_analysis_output[1])
        res = codeql_issues + '\n' + bandit_issues
        prompt = f'''Please review the static analysis output below for a piece of code:
{res}
In a concise manner, summarize the issues identified in the analysis and provide recommended fixes for each issue. Use clear and succinct language.
*Issues*:
*Recommended fixes*:
'''
        resp = self.llm.generate(prompt)

        return resp

    def extract_fuzzing_results(self, fuzzing_output: str) -> dict:
        """
            提取fuzzing有效信息
            """
        prompt = f'''Please analyze the following fuzzing test output and provide a concise summary.
Identify any crashes, errors, or unexpected behavior found, and explain them in simple language.
Fuzzing Output:
{fuzzing_output}
'''
        result = self.llm.generate(prompt)
        return result
