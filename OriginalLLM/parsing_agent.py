import re
from llms import LLMInterface

class LLMParsingAgent():
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def extract_test_results(self, test_output: str) -> dict:
        """
        提取功能测试的有效信息，包括测试是否通过与失败原因
        """
        prompt = f'''You are a senior Python test engineer. Analyze and format these unit test results into a professional report. Follow these formatting rules:

**Test Case #[N]**:
- Input Parameters: (if there are)
- Expected Output: (if there is)
- Actual Output: (if there is)
- Status: Failed / Error
- Error Trace: [Short error summary if applicable]

Process the following unit test result: {test_output}
'''
        result = self.llm.generate(prompt)
        return result

    def extract_static_analysis_results(self, static_analysis_output) -> list:
        """
        提取静态分析工具输出中的安全问题和代码质量问题
        """
        codeql_issues = str(static_analysis_output[0])
        bandit_issues = str(static_analysis_output[1])
        prompt = f'''You are a senior python security engineer. Analyze and synthesize results from CodeQL and Bandit scanners to create a comprehensive security report. Follow these guidelines:

**Input Requirements**
Process outputs from:
1. CodeQL: {codeql_issues}
2. Bandit: {bandit_issues}

【Output Format】
### Security Analysis Report
#### Summary
- Total Issues: [X]
  - Critical: [C] 
  - High: [H]
  - Medium: [M]
  - Low: [L]
- Tools Coverage:
  - CodeQL Findings: [N]
  - Bandit Findings: [K]
  - Overlapping Issues: [O]
'''
        issues = self.llm.generate(prompt)

        return issues

    def extract_fuzzing_results(self, fuzzing_output: str) -> dict:
        """
            提取功能测试的有效信息，包括测试是否通过与失败原因
            """
        prompt = f'''You are a senior python security researcher. Analyze these fuzzing results and generate an crash report. Follow these formatting rules:

### Fuzzing test Report
#### Execution Overview
- Errors:
    - Error 1: [Description]
    - Error 2: [Description]
    - ...
- Crashes:
    - Crash 1: [Description]
    - Crash 2: [Description]
    - ...

Process the following fuzzing test result: {fuzzing_output}
'''
        result = self.llm.generate(prompt)
        return result
