import json
from collections import Counter
import sys


def count_statuses(jsonl_file):
    cnt = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            cnt += 1
            line = line.strip()
            if not line:
                continue  # 跳过空行
            record = json.loads(line)

            # 获取各个字段的值，如果不存在可以使用 'unknown' 作为默认值
            unit_test_status = record.get('unit_test_status', 'unknown')
            static_analysis_status = record.get(
                'static_analysis_status', 'unknown')
            fuzzing_test_status = record.get('fuzzing_test_status', 'unknown')

            if 'fixed, round: 3' in unit_test_status:
                cnt += 6
            if 'fixed, round: 2' in unit_test_status:
                cnt += 4
            if 'fixed, round: 1' in unit_test_status:
                cnt += 2
            if str(unit_test_status).startswith('fail:'):
                cnt += 6

            if 'fixed, round: 3' in static_analysis_status:
                cnt += 6
            if 'fixed, round: 2' in static_analysis_status:
                cnt += 4
            if 'fixed, round: 1' in static_analysis_status:
                cnt += 2
            if str(static_analysis_status).startswith('fail:'):
                cnt += 6

            if 'fixed, round: 3' in fuzzing_test_status:
                cnt += 7
            if 'fixed, round: 2' in fuzzing_test_status:
                cnt += 5
            if 'fixed, round: 1' in fuzzing_test_status:
                cnt += 3
            if 'success' in fuzzing_test_status:
                cnt += 1
            if str(fuzzing_test_status).startswith('error'):
                cnt += 7
    return cnt


if __name__ == '__main__':
    jsonl_file = 'target.json'
    cnt = count_statuses(jsonl_file)

    print(cnt/285)
