import json


def calculate_average_time(jsonl_file):
    total_time = 0
    count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            total_time += data.get('time', 0)
            count += 1

    if count == 0:
        return 0

    return total_time / count


if __name__ == "__main__":
    jsonl_file_path = 'target.json'
    average_time = calculate_average_time(jsonl_file_path)
    print(f"The average time is: {average_time}")
