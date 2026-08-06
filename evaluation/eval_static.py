from datasets import load_dataset
import json

from executor_static import ExecutorStaticAgent
from functional_test_agent import LLMFunctionalTestAgent

from enum import Enum

from static_analysis_agent import BanditStaticAnalysisTool, CodeQLStaticAnalyzer


class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


class MultiAgentSystem:
    def __init__(self, entry, skip_code):
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

        self.executor_static = ExecutorStaticAgent(entry)
        self.codeql_static_agent = CodeQLStaticAnalyzer(entry)
        self.bandit_static_agent = BanditStaticAnalysisTool(entry)

        self.code = skip_code
        self.testcases = None

    def run(self, model, tokenizer, device, iterations=100):
        static_analysis_status = self.static_analyze()
        return static_analysis_status

    def static_analyze(self):
        result, error_description = self.executor_static.execute_static_analysis_gpt(
            self.code, self.codeql_static_agent, self.bandit_static_agent)
        print('result')
        print(result)
        # If error, give feedback to programmer agent up to 4 times
        if result.name != FResult.SAFE.name:
            return 'fail: ' + str(error_description)
        else:
            static_analysis_status = 'success'
            return static_analysis_status


def count_static_status(file_path):
    status_counts = {}
    skipped_ids = []
    skipped_codes = {}
    fail_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            status = data.get('static_analysis_status')
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
    skipped_ids, skipped_codes, temp_status, fail_ids = count_static_status(
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
    for i, entry in enumerate(dataset):
        if 'task_id' in entry:
            entry['ID'] = entry['task_id']
            del entry['task_id']
        if entry['ID'] in skipped_ids:
            system = MultiAgentSystem(entry, skipped_codes[entry['ID']])
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
