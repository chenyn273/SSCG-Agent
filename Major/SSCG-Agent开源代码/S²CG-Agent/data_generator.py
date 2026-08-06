import json
import os
from typing import Dict, List, Tuple


def load_jsonl(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_prompt_map_humaneval(data: List[Dict]) -> Dict[str, str]:
    prompt_map = {}
    for item in data:
        task_id = item.get('task_id', '')
        prompt = item.get('prompt', '')
        prompt_map[task_id] = prompt
    return prompt_map


def build_prompt_map_securityeval(data: List[Dict]) -> Dict[str, str]:
    prompt_map = {}
    for item in data:
        task_id = item.get('ID', '')
        prompt = item.get('Prompt', '')
        prompt_map[task_id] = prompt
    return prompt_map


def status_to_label(status: str) -> int:
    if status == "success" or status == "skipped":
        return 0
    else:
        return 1


def generate_training_data(
    humaneval_path: str,
    securityeval_path: str,
    results_folder: str,
    output_folder: str,
    train_ratio: float = 0.8
):
    print("Loading benchmark data...")
    humaneval_data = load_jsonl(humaneval_path)
    securityeval_data = load_jsonl(securityeval_path)
    
    humaneval_prompts = build_prompt_map_humaneval(humaneval_data)
    securityeval_prompts = build_prompt_map_securityeval(securityeval_data)
    
    print(f"HumanEval prompts: {len(humaneval_prompts)}")
    print(f"SecurityEval prompts: {len(securityeval_prompts)}")
    
    all_prompts = {}
    all_prompts.update(humaneval_prompts)
    all_prompts.update(securityeval_prompts)
    
    print("\nProcessing result files...")
    all_samples = []
    
    for filename in os.listdir(results_folder):
        if not filename.endswith('.json'):
            continue
        
        llm_name = filename.replace('.json', '')
        file_path = os.path.join(results_folder, filename)
        
        print(f"  Processing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                data = json.loads(line)
                task_id = data.get('ID', '')
                code = data.get('code', '')
                
                ut_status = data.get('unit_test_status', '')
                sa_status = data.get('static_analysis_status', '')
                ft_status = data.get('fuzzing_test_status', '')
                
                ut_label = status_to_label(ut_status)
                sa_label = status_to_label(sa_status)
                ft_label = status_to_label(ft_status)
                
                if ut_label == 0 and sa_label == 0 and ft_label == 0:
                    labels = [0, 0, 0, 1]
                else:
                    labels = [ut_label, sa_label, ft_label, 0]
                
                prompt = all_prompts.get(task_id, '')
                if not prompt:
                    print(f"    Warning: No prompt found for {task_id}")
                    continue
                
                input_text = prompt + ' </s> ' + code
                
                sample = {
                    'id': task_id,
                    'llm': llm_name,
                    'input': input_text,
                    'labels': labels,
                    'raw_status': [ut_status, sa_status, ft_status]
                }
                all_samples.append(sample)
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    train_size = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    train_path = os.path.join(output_folder, 'train.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\nSaved training data to {train_path}")
    
    val_path = os.path.join(output_folder, 'val.jsonl')
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved validation data to {val_path}")
    
    stats = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'label_distribution': {
            'all_zero': sum(1 for s in all_samples if s['labels'] == [0, 0, 0, 1]),
            'ut_only': sum(1 for s in all_samples if s['labels'] == [1, 0, 0, 0]),
            'sa_only': sum(1 for s in all_samples if s['labels'] == [0, 1, 0, 0]),
            'ft_only': sum(1 for s in all_samples if s['labels'] == [0, 0, 1, 0]),
            'ut_sa': sum(1 for s in all_samples if s['labels'] == [1, 1, 0, 0]),
            'ut_ft': sum(1 for s in all_samples if s['labels'] == [1, 0, 1, 0]),
            'sa_ft': sum(1 for s in all_samples if s['labels'] == [0, 1, 1, 0]),
            'all_one': sum(1 for s in all_samples if s['labels'] == [1, 1, 1, 0]),
        }
    }
    
    stats_path = os.path.join(output_folder, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to {stats_path}")
    
    print("\nLabel distribution:")
    for key, value in stats['label_distribution'].items():
        print(f"  {key}: {value} ({value/len(all_samples)*100:.1f}%)")
    
    return train_samples, val_samples


if __name__ == "__main__":
    HUMANEVAL_PATH = "task1_data_leakage/humaneval.jsonl"
    SECURITYEVAL_PATH = "task1_data_leakage/SecurityEval.jsonl"
    RESULTS_FOLDER = "SSCG-Agent开源代码/results/OriginalLLM"
    OUTPUT_FOLDER = "task2_moe_scheduler/data"
    
    generate_training_data(
        humaneval_path=HUMANEVAL_PATH,
        securityeval_path=SECURITYEVAL_PATH,
        results_folder=RESULTS_FOLDER,
        output_folder=OUTPUT_FOLDER,
        train_ratio=0.8
    )
