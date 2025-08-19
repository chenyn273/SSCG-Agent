import json
import os


def count_status_in_jsonl(file_path):
    fixed_count = 0
    round_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                sample = json.loads(line)  # 解析jsonl中的每一行
                # 获取不同的status字段
                unit_test_status = sample.get('unit_test_status', '')
                static_analysis_status = sample.get(
                    'static_analysis_status', '')
                fuzzing_test_status = sample.get('fuzzing_test_status', '')

                # 统计包含 "fixed" 和 "round:" 的数量
                if 'fixed' in unit_test_status or unit_test_status.startswith('fail:'):
                    fixed_count += 1
                    if 'round: 1' in unit_test_status:
                        round_count += 1
                    elif 'round: 2' in unit_test_status:
                        round_count += 2
                    elif 'round: 3' in unit_test_status:
                        round_count += 3
                    else:
                        round_count += 3
                    
                # 统计static_analysis_status和fuzzing_test_status的相关字段
                if 'fixed' in static_analysis_status or static_analysis_status.startswith('fail:'):
                    fixed_count += 1
                    if 'round: 1' in static_analysis_status:
                        round_count += 1
                    elif 'round: 2' in static_analysis_status:
                        round_count += 2
                    elif 'round: 3' in static_analysis_status:
                        round_count += 3
                    else:
                        round_count += 3
                    
                if 'fixed' in fuzzing_test_status or fuzzing_test_status.startswith('error:'):
                    fixed_count += 1
                    if 'round: 1' in fuzzing_test_status:
                        round_count += 1
                    elif 'round: 2' in fuzzing_test_status:
                        round_count += 2
                    elif 'round: 3' in fuzzing_test_status:
                        round_count += 3
                    else:
                        round_count += 3

            except json.JSONDecodeError:
                print("Error decoding JSON, skipping line.")

    return fixed_count, round_count


def count_status_in_all_jsonl_files(directory_path):
    total_fixed_count = 0
    total_round_count = 0

    # 遍历目录中的所有jsonl文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):  # 仅处理jsonl文件
            file_path = os.path.join(directory_path, filename)
            fixed_count, round_count = count_status_in_jsonl(
                file_path)

            # 合并统计
            total_fixed_count += fixed_count
            total_round_count += round_count

    return total_fixed_count, total_round_count


# 使用时输入文件夹路径
directory_path = 'target_path'
total_fixed_count, total_round_count = count_status_in_all_jsonl_files(
    directory_path)

print("Total fixed count:", total_fixed_count)
print("Total round count:", total_round_count)
print(total_round_count/total_fixed_count)
