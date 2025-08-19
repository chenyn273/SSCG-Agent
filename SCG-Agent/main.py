import re
import time
import types
from tqdm.auto import tqdm
import json
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, AutoConfig
import torch
from executor_agent_safe import FResult, execute_fuzz
from functional_test_agent import LLMFunctionalTestAgent, modify_test_code
from llms import DeepSeek_LLM, Qwen_LLM, OpenAI_LLM
from parsing_agent import LLMParsingAgent
from programmer_agent import ProgrammerAgent
from static_analysis_agent import BanditStaticAnalysisTool, CodeQLStaticAnalyzer
from fuzz_agent import TesterFuzzAgent, InputMutatorAgent
from executor_static import ExecutorStaticAgent
from datasets import load_dataset
import json
import os
import gzip


class MultiAgentSystem:
    def __init__(self, entry, llm):
        # for humaneval dataset
        # 如果entry中的key是Prompt，将其改为prompt
        if 'Prompt' not in entry:
            entry['Prompt'] = entry['prompt']
            del entry['prompt']
        # 如果entry中的key是task_id，将其改为ID
        if 'task_id' in entry:
            entry['ID'] = entry['task_id']
            del entry['task_id']
        # 修改可执行测试用例
        if 'test' in entry:
            entry['test'] = modify_test_code(entry['test'])

        self.llm = llm
        self.entry = entry
        self.programmer_agent = ProgrammerAgent(entry, llm)
        self.tester_fuzz_agent = TesterFuzzAgent(entry, llm)
        self.executor_static = ExecutorStaticAgent(entry)
        self.codeql_static_agent = CodeQLStaticAnalyzer(entry)
        self.bandit_static_agent = BanditStaticAnalysisTool(entry)
        self.func_test_agent = LLMFunctionalTestAgent(entry, llm)
        self.parsing_agent = LLMParsingAgent(llm)
        self.code = None
        self.testcases = None
        self.test_inputs = None

    def run(self, model, tokenizer, device, iterations=100):
        # 记录运行的开始时间
        begin_time = time.time()
        # Step 1: Programmer writes code
        for w_i in range(3):
            self.code = self.programmer_agent.write_code()
            if self.code != '':
                break
        print(f"Programmer's Code:\n{self.code}")
        problems = {}

        problems[entry['ID']] = {'ID': entry['ID']}
        func_test_status = self.unit_test()
        static_analysis_status = self.static_analyze()
        fuzzing_test_status = self.fuzzing(iterations)
        if not os.path.exists("results.json"):
            with open("results.json", 'w') as f:
                pass
        # 记录运行的结束时间
        end_time = time.time()
        total_time = end_time - begin_time
        # result includes code, fuzzing inputs used, static and fuzzing testing status
        task = {**problems[entry['ID']], "code": self.code, "unit_test_status": func_test_status,
                "static_analysis_status": static_analysis_status, "fuzzing_test_status": fuzzing_test_status, "time": total_time}
        with open("results.json", 'a') as f:  # 'a' mode to append to the file
            json.dump(task, f)
            f.write("\n")  # Write a new line after each JSON object

        print(f"Tasks saved to results.json")

    def fuzzing(self, iterations):
        test_inputs_list = []
        self.test_inputs = self.tester_fuzz_agent.generate_test_inputs()
        if (self.test_inputs == {}):
            fuzzing_test_status = "error: no inputs created"
            return fuzzing_test_status
        test_inputs_list.append(self.test_inputs)
        print(f"Initial Test Inputs:\n{self.test_inputs}")

        failed_inputs_fuzz = []
        # Step 5: Execute and mutate in a loop for the given number of iterations
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}")
            # Step 5a: Executor runs the code with current inputs
            try:
                result, passed, functionname = execute_fuzz(
                    self.code, self.test_inputs, 3)
            except Exception as e:
                failed_inputs_fuzz.append(
                    {'inputs': self.test_inputs, 'result': str(e)})
                if len(failed_inputs_fuzz) > 3:
                    break
                continue
                # If there's an error, flag the test as failed
            if not passed:
                if ("No module named" in result):
                    fuzzing_test_status = "error: module missing: " + result
                    return fuzzing_test_status
                else:
                    failed_inputs_fuzz.append(
                        {'inputs': self.test_inputs, 'result': result})
                    if len(failed_inputs_fuzz) > 3:
                        break

            mutator_agent = InputMutatorAgent(
                self.test_inputs, self.code, functionname)
            self.test_inputs = mutator_agent.mutate_inputs()
            test_inputs_list.append(self.test_inputs)
            print(f"Mutated Inputs:\n{self.test_inputs}")

            # Step 6: If errors were found in fuzzing, give feedback to coder to fix
        if len(failed_inputs_fuzz) != 0:
            # Give feedback to Coder up to 3 times
            for i in range(3):
                fuzzing_feedback = self.parsing_agent.extract_fuzzing_results(
                    failed_inputs_fuzz[:3])
                self.code = self.programmer_agent.write_code_feedback_fuzz(
                    self.code, fuzzing_feedback)
                new_failed_inputs = []
                for inputs in failed_inputs_fuzz:
                    try:
                        # run the code with the failing inputs to see if problem is fixed
                        result, passed, _functionname = execute_fuzz(
                            self.code, inputs['inputs'], 3)
                    except Exception as e:
                        new_failed_inputs.append(
                            {'inputs': inputs['inputs'], 'result': str(e)})
                        continue
                    if not passed:
                        new_failed_inputs.append(
                            {'inputs': inputs['inputs'], 'result': result})
                if len(new_failed_inputs) != 0:
                    failed_inputs_fuzz = new_failed_inputs
                    continue
                else:
                    fuzzing_test_status = f'fixed, round: {i + 1}'
                    return fuzzing_test_status
            if len(new_failed_inputs) != 0:
                fuzzing_test_status = 'error:' + \
                    ' '.join([x['result'] for x in new_failed_inputs])
                return fuzzing_test_status
        else:
            fuzzing_test_status = 'success'
            return fuzzing_test_status

    def static_analyze(self):
        result, error_description = self.executor_static.execute_static_analysis_gpt(
            self.code, self.codeql_static_agent, self.bandit_static_agent)
        print('result')
        print(result)
        # If error, give feedback to programmer agent up to 4 times
        if result.name != FResult.SAFE.name:
            error_description = self.parsing_agent.extract_static_analysis_results(
                error_description)
            for i in range(3):
                self.code = self.programmer_agent.write_code_feedback_static(
                    self.code, str(error_description), "")
                result, error_description = self.executor_static.execute_static_analysis_gpt(
                    self.code, self.codeql_static_agent, self.bandit_static_agent)
                if result.name == FResult.SAFE.name:
                    static_analysis_status = f'fixed, round: {i + 1}'
                    return static_analysis_status
                error_description = self.parsing_agent.extract_static_analysis_results(
                    error_description)
            if result.name != FResult.SAFE.name:
                static_analysis_status = 'fail: ' + error_description
                return static_analysis_status
        else:
            static_analysis_status = 'success'
            return static_analysis_status

    def unit_test(self):
        if 'test' in self.entry:
            self.testcases = self.entry['test']
            func_test_status, func_test_res = self.func_test_agent.run_tests(
                self.code, self.testcases, True)
            if not func_test_status:
                if not ("No module named" in func_test_res):
                    for i in range(3):
                        # 提取功能测试结果的有效信息
                        func_test_res = self.parsing_agent.extract_test_results(
                            func_test_res)
                        self.code = self.programmer_agent.write_code_feedback_func(
                            self.code, func_test_res)
                        func_test_status, func_test_res = self.func_test_agent.run_tests(
                            self.code, self.testcases, True)
                        if func_test_status:
                            func_test_status = f'fixed, round: {i + 1}'
                            return func_test_status
                    if not func_test_status:
                        func_test_status = 'fail: ' + func_test_res
                        return func_test_status
                else:
                    func_test_status = 'error: module missing: ' + func_test_res
                    return func_test_status
            else:
                func_test_status = 'success'
                return func_test_status
        else:
            func_test_status = 'skipped'
            return func_test_status


if __name__ == "__main__":
    openai_api_key = 'your-key'
    ali_api_key = 'your-key'
    deep_seek_key = 'your-key'
    # Human eval dataset
    humaneval_ds = load_dataset("openai_humaneval", split="test")

    # 将 Humaneval 数据写入 JSONL 文件
    with open("humaneval.jsonl", "w", encoding="utf-8") as f:
        for item in humaneval_ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 2. 定义一个函数读取 JSONL 文件
    def read_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # 分别读取两个 JSONL 文件
    data_humaneval = read_jsonl("humaneval.jsonl")
    data_security = read_jsonl("SecurityEval.jsonl")

    # 3. 合并两个列表
    dataset = data_humaneval + data_security

    # set llm and run
    llm = Qwen_LLM(ali_api_key, 'qwen-max')  # qwen-max

    for i, entry in enumerate(dataset):
        system = MultiAgentSystem(entry, llm)
        system.run(None, None, None)
