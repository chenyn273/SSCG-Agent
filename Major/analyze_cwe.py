import json
import re
import os
from collections import defaultdict

file_path = 'major_revision_results/eval_summary.json'

def get_cwe_id(fail_id):
    # Extract CWE-XXX from "CWE-XXX_..."
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

    # Structure: method -> model -> cwe_id -> count
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Track all unique CWEs to build the table
    all_cwes = set()

    for key, value in data.items():
        # Identify Method and Model
        if key.startswith('collaborative_') and '_th0.25' in key:
            method = 'SCG-Agent'
            # Extract model: collaborative_{model}_th0.25
            model = key.replace('collaborative_', '').replace('_th0.25', '')
        elif key.startswith('reflexion_'):
            method = 'Reflexion'
            model = key.replace('reflexion_', '')
        elif key.startswith('intervenor_'):
            method = 'Intervenor'
            model = key.replace('intervenor_', '')
        elif key.startswith('paircoder_'):
            method = 'PairCoder'
            model = key.replace('paircoder_', '')
        elif key.startswith('self_collaboration_'):
            method = 'Self-Collaboration'
            model = key.replace('self_collaboration_', '')
        elif key.startswith('autosafe_'):
            method = 'AutoSafeCoder'
            model = key.replace('autosafe_', '')
        elif key.startswith('standalone_'):
            method = 'Standalone'
            model = key.replace('standalone_', '')
        else:
            continue

        # Normalize model names
        if 'claude-3-5-sonnet' in model:
            model = 'Claude-3.5-Sonnet'
        elif 'gpt-4o' in model:
            model = 'GPT-4o'
        elif 'qwen-max' in model:
            model = 'Qwen-Max'
        elif 'qwen-coder-plus' in model:
            model = 'Qwen-Coder-Plus'
        elif 'gemini-1.5-pro' in model:
            model = 'Gemini-1.5-Pro'
        elif 'deepseek-v3' in model:
            model = 'DeepSeek-V3'
        
        # Only process if we have static_analysis fail_ids
        if 'static_analysis' in value and 'fail_ids' in value['static_analysis']:
            fail_ids = value['static_analysis']['fail_ids']
            for fid in fail_ids:
                cwe = get_cwe_id(fid)
                if cwe:
                    stats[method][model][cwe] += 1
                    all_cwes.add(cwe)

    # Output Summary Table
    print("### Summary of SecurityEval Vulnerability Repair (Total 121 Samples)")
    print("| Method | Model | Failed Samples (Count) | Failure Rate (%) | Top 3 Failed CWEs |")
    print("|---|---|---|---|---|")
    
    # Sort methods and models
    for method in sorted(stats.keys()):
        for model in sorted(stats[method].keys()):
            # Count total failed CWE instances (assuming 1 fail_id = 1 sample)
            # Actually, stats[method][model] is a dict of CWE -> count.
            # But wait, one sample might have multiple CWEs?
            # The JSON fail_ids are filenames like "CWE-078_author_1.py".
            # Each file is one sample.
            # So the number of unique fail_ids is the number of failed samples.
            # My current stats counts occurrences of CWE prefixes.
            # Since file names are unique, sum of counts in stats[method][model] should be total failed samples
            # IF each file maps to exactly one CWE prefix.
            # Yes, "CWE-XXX_..." maps to CWE-XXX.
            
            total_fails = sum(stats[method][model].values())
            fail_rate = (total_fails / 121) * 100
            
            # Top 3
            sorted_cwes_by_count = sorted(stats[method][model].items(), key=lambda x: x[1], reverse=True)
            top3 = [f"{cwe}({count})" for cwe, count in sorted_cwes_by_count[:3]]
            top3_str = ", ".join(top3)
            
            print(f"| {method} | {model} | {total_fails} | {fail_rate:.2f}% | {top3_str} |")
    print("\n")

    # Output Detailed Distribution Tables
    sorted_cwes = sorted(list(all_cwes))
    
    for method in sorted(stats.keys()):
        print(f"### Detailed CWE Distribution: {method}")
        models = sorted(stats[method].keys())
        
        # Header
        header = "| CWE ID | " + " | ".join(models) + " |"
        print(header)
        print("|---" + "|---" * len(models) + "|")
        
        # Rows
        for cwe in sorted_cwes:
            # Check if this CWE has any failure in this method
            has_failure_in_method = False
            for model in models:
                if stats[method][model].get(cwe, 0) > 0:
                    has_failure_in_method = True
                    break
            
            if has_failure_in_method:
                row = [cwe]
                for model in models:
                    count = stats[method][model].get(cwe, 0)
                    row.append(str(count) if count > 0 else "0")
                print("| " + " | ".join(row) + " |")
        print("\n")

if __name__ == "__main__":
    main()
