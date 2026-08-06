from datasets import load_dataset
import json

from executor_agent_safe import execute_fuzz
from executor_static import ExecutorStaticAgent
from functional_test_agent import LLMFunctionalTestAgent

from enum import Enum

from fuzz_agent import InputMutatorAgent, TesterFuzzAgent
from llms import OpenAI_LLM, Qwen_LLM
from static_analysis_agent import BanditStaticAnalysisTool, CodeQLStaticAnalyzer


class MultiAgentSystem:
    def __init__(self, entry, skip_code, llm):
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
        self.entry = entry

        self.tester_fuzz_agent = TesterFuzzAgent(entry, llm)

        self.code = skip_code
        self.testcases = None
        self.test_inputs = None

    def run(self, model, tokenizer, device, iterations=100):
        fuzzing_test_status = self.fuzzing(iterations)
        return fuzzing_test_status

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
            return 'fail'
        else:
            fuzzing_test_status = 'success'
            return fuzzing_test_status


def count_fuzzing_status(file_path):
    status_counts = {}
    skipped_ids = []
    skipped_codes = {}
    fail_ids = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            status = data.get('fuzzing_test_status')
            if status:
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts[status] = 1
                if status == 'skipped':
                    skipped_ids.append(data.get('ID'))
                    skipped_codes[data.get('ID')] = data.get('code')
                if 'fail' in status or 'error' in status:
                    fail_ids.append(i)

    # for status, count in status_counts.items():
    #     print(f"{status} --------------- {count}")

    return skipped_ids, skipped_codes, status_counts, fail_ids


if __name__ == "__main__":
    file_path = 'target.json'
    skipped_ids, skipped_codes, temp_status, fail_ids = count_fuzzing_status(
        file_path)
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

    skipped_status = []
    ali_api_key = 'your-key'

    llm = Qwen_LLM(ali_api_key, 'qwen-max')  # qwen-max
    for i, entry in enumerate(dataset):
        if 'task_id' in entry:
            entry['ID'] = entry['task_id']
            del entry['task_id']
        if entry['ID'] in skipped_ids:
            system = MultiAgentSystem(
                entry, skipped_codes[entry['ID']], llm)
            s = system.run(None, None, None)
            skipped_status.append(s)
            if 'fail' in s or 'error' in s:
                fail_ids.append(i)
    status_summary = {}
    for status in skipped_status:
        if status in status_summary:
            status_summary[status] += 1
        else:
            status_summary[status] = 1

    for status, count in temp_status.items():
        if status in status_summary:
            status_summary[status] += count
        else:
            status_summary[status] = count

    success = 0
    fixed = 0
    for status, count in status_summary.items():
        if status == 'success':
            success += count
        if 'fixed, round' in status:
            fixed += count

    print('success:', success)
    print('fixed', fixed)
    print('total', success+fixed)
    print('fail:', fail_ids)
