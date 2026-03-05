import json
import re
import os
from collections import defaultdict

file_path = 'major_revision_results/eval_summary.json'

def get_cwe_id(fail_id):
    match = re.match(r'(CWE-\d+)', fail_id)
    if match:
        return match.group(1)
    return None

def main():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Structure: model -> cwe_id -> count
    # Only for SCG-Agent (collaborative_{model}_th0.25)
    stats = defaultdict(lambda: defaultdict(int))
    total_fails_per_model = defaultdict(int)
    
    # Target models as requested in the LaTeX snippet:
    # GPT-4o, Claude-3-5-Sonnet, Gemini-1.5-Pro, DeepSeek-Chat (DeepSeek-V3), Qwen-Coder-Plus, Qwen-Max
    target_models = {
        'gpt-4o': 'GPT-4o',
        'claude-3-5-sonnet-20241022': 'Claude-3.5-Sonnet',
        'gemini-1.5-pro': 'Gemini-1.5-Pro',
        'deepseek-v3': 'DeepSeek-V3',
        'qwen-coder-plus': 'Qwen-Coder-Plus',
        'qwen-max': 'Qwen-Max'
    }

    # Aggregate all CWE failures across all models to find Top 5 globally
    global_cwe_counts = defaultdict(int)

    for key, value in data.items():
        if key.startswith('collaborative_') and '_th0.25' in key:
            model_key = key.replace('collaborative_', '').replace('_th0.25', '')
            
            if model_key not in target_models:
                continue
                
            model_name = target_models[model_key]
            
            if 'static_analysis' in value and 'fail_ids' in value['static_analysis']:
                fail_ids = value['static_analysis']['fail_ids']
                total_fails_per_model[model_name] = len(fail_ids)
                
                for fid in fail_ids:
                    cwe = get_cwe_id(fid)
                    if cwe:
                        stats[model_name][cwe] += 1
                        global_cwe_counts[cwe] += 1

    # Find Top 5 CWEs globally
    sorted_global_cwes = sorted(global_cwe_counts.items(), key=lambda x: x[1], reverse=True)
    top5_cwes = [cwe for cwe, count in sorted_global_cwes[:5]]
    
    print("Top 5 CWEs:", top5_cwes)
    
    # Print Table Header
    # LLM Backend | Total | SA Fail | CWE-A | CWE-B | ... | Others
    
    header = ["LLM Backend", "Total", "SA Fail"] + top5_cwes + ["Others"]
    print(" | ".join(header))
    print("|---" * len(header) + "|")
    
    # Rows
    # Order: GPT-4o, Claude, Gemini, DeepSeek, Qwen-Coder, Qwen-Max
    display_order = ['GPT-4o', 'Claude-3.5-Sonnet', 'Gemini-1.5-Pro', 'DeepSeek-V3', 'Qwen-Coder-Plus', 'Qwen-Max']
    
    for model in display_order:
        total_fail = total_fails_per_model[model]
        
        row = [model, "121", str(total_fail)]
        
        other_count = 0
        for cwe in top5_cwes:
            count = stats[model].get(cwe, 0)
            row.append(str(count))
        
        # Calculate Others
        # Sum of counts for CWEs NOT in top5
        for cwe, count in stats[model].items():
            if cwe not in top5_cwes:
                other_count += count
        
        row.append(str(other_count))
        
        print(" | ".join(row))

if __name__ == "__main__":
    main()
