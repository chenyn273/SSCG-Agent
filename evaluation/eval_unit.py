from datasets import load_dataset
import json

from functional_test_agent import LLMFunctionalTestAgent


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
        if 'test' in entry:
            entry['test'] = modify_test_code(entry['test'])

        self.entry = entry

        self.func_test_agent = LLMFunctionalTestAgent(entry, None)

        self.code = skip_code
        self.testcases = None

    def run(self, model, tokenizer, device, iterations=100):
        func_test_status = self.unit_test()
        return func_test_status

    def unit_test(self):
        if 'test' in self.entry:
            self.testcases = self.entry['test']
            func_test_status, func_test_res = self.func_test_agent.run_tests(
                self.code, self.testcases, True)
            if not func_test_status:
                return 'fail:' + func_test_res
            else:
                func_test_status = 'success'
                return func_test_status


def count_unit_test_status(file_path):
    status_counts = {}
    skipped_ids = []
    skipped_codes = {}
    fail_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            status = data.get('unit_test_status')
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
    skipped_ids, skipped_codes, temp_status, fail_ids = count_unit_test_status(
        file_path)
    dataset = load_dataset("openai_humaneval", split="test")
    skipped_status = []
    for i, entry in enumerate(dataset):
        if entry['task_id'] in skipped_ids:
            system = MultiAgentSystem(entry, skipped_codes[entry['task_id']])
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
