import os
import sys
import json
import subprocess
import shutil

# Add the evaluation directory to sys.path
BASE_DIR = "/evaluation"
sys.path.append(BASE_DIR)

# Define script paths
SCRIPTS = {
    "api_calls": os.path.join(BASE_DIR, "api_calls.py"),
    "unit_test": os.path.join(BASE_DIR, "eval_unit.py"),
    "fuzzing": os.path.join(BASE_DIR, "eval_fuzzing.py"),
    "static_analysis": os.path.join(BASE_DIR, "eval_static.py")
}

SUMMARY_FILE = 'eval_summary.json'


def load_and_prepare_data(input_path):
    """
    Load data from file or directory, and prepare a temporary JSONL file for evaluation.
    Returns: (temp_file_path, is_baseline_dir, model_name)
    """
    # Use PID to make temp file unique for parallel runs
    temp_file = f"temp_eval_{os.getpid()}.jsonl"
    
    # Infer model name
    # e.g., .../standalone/gpt-4o.json -> standalone_gpt-4o
    # e.g., .../paircoder/deepseek-v3 -> paircoder_deepseek-v3
    abs_path = os.path.abspath(input_path)
    parts = abs_path.rstrip(os.sep).split(os.sep)
    
    if os.path.isfile(input_path):
        is_baseline = False
        
        # Improve model_name extraction logic
        if len(parts) >= 3 and parts[-1] == "result.jsonl":
            # Likely baseline format: .../baseline_name/model_name/result.jsonl
            # We want key: baseline_name_model_name
            baseline_name = parts[-3]
            model_name_part = parts[-2]
            model_name = f"{baseline_name}_{model_name_part}"
        elif len(parts) >= 2:
            # Fallback or SCG format: .../category/model.json
            category = parts[-2]
            name = parts[-1].replace('.jsonl', '').replace('.json', '')
            model_name = f"{category}_{name}"
        else:
            model_name = os.path.basename(input_path).replace('.jsonl', '').replace('.json', '')
            
        # Determine if it's SCG or Baseline based on 'api_calls' key in the first line
        # SCG results (standalone/collaborative) usually have api_calls inside but structure is different?
        # Actually, user said:
        # 1. SCG results have 3 status keys, but NO 'api_calls' (or at least we don't use it for avg calc from root?)
        # 2. Baseline results have 'api_calls', but NO 3 status keys.
        
        # Let's peek at the first line to determine mode
        has_api_calls = False
        has_status = False
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    if 'api_calls' in data:
                        has_api_calls = True
                    if 'unit_test_status' in data:
                        has_status = True
        except:
            pass
            
        if has_api_calls and not has_status:
            is_baseline = True
            print(f"Detected Baseline format (has api_calls, no status)")
        else:
            is_baseline = False
            print(f"Detected SCG format")

        with open(input_path, 'r', encoding='utf-8') as f_in, open(temp_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    record = json.loads(line)
                    # Normalize ID
                    if 'task_id' in record and 'ID' not in record:
                        record['ID'] = record['task_id']
                    
                    # Ensure status fields exist for Baseline-like files
                    # FORCE RESET if it is baseline format
                    if is_baseline:
                        record['unit_test_status'] = 'skipped'
                        record['fuzzing_test_status'] = 'skipped'
                        record['static_analysis_status'] = 'skipped'
                        
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                except:
                    pass
        
    elif os.path.isdir(input_path):
        # Support directory input as Baseline (legacy support, but user wants unified file input)
        # We can keep this for backward compatibility or just fail.
        # Let's keep it but warn.
        print("Warning: Directory input detected. Assuming Baseline format.")
        is_baseline = True
        is_baseline = True
        if len(parts) >= 2:
            model_name = f"{parts[-2]}_{parts[-1]}"
        else:
            model_name = os.path.basename(input_path)
            
        # Check for result.jsonl
        result_file = os.path.join(input_path, "result.jsonl")
        
        merged_data = []
        
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'task_id' in record and 'ID' not in record:
                            record['ID'] = record['task_id']
                        
                        # FORCE RESET STATUS for Baseline
                        record['unit_test_status'] = 'skipped'
                        record['fuzzing_test_status'] = 'skipped'
                        record['static_analysis_status'] = 'skipped'
                        
                        merged_data.append(record)
                    except:
                        pass
        else:
            print(f"Error: result.jsonl not found in directory {input_path}")
            sys.exit(1)
        
        if not merged_data:
            print(f"Error: No valid data found in {result_file}")
            sys.exit(1)
            
        with open(temp_file, 'w', encoding='utf-8') as f:
            for record in merged_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
    else:
        print(f"Error: Invalid input path {input_path}")
        sys.exit(1)
        
    return temp_file, is_baseline, model_name


def evaluate_api_calls(temp_file, is_baseline):
    if not is_baseline:
        # Use api_calls.py for SCG
        try:
            output = subprocess.check_output([sys.executable, SCRIPTS["api_calls"], temp_file], text=True)
            return float(output.strip())
        except Exception as e:
            print(f"Error running api_calls.py: {e}")
            return 0
    else:
        # Calculate average api_calls field for Baseline
        total_calls = 0
        count = 0
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'api_calls' in data:
                    total_calls += data['api_calls']
                    count += 1
        return total_calls / count if count > 0 else 0


def parse_script_output(output):
    """
    Parses the stdout from evaluation scripts.
    Expected format often includes:
    success: N
    fixed N
    total N
    fail: [list]
    """
    res = {"success": 0, "fixed": 0, "total": 0, "fail_ids": []}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("success:"):
            try:
                res["success"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("fixed"): # "fixed 3" or "fixed: 3"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    res["fixed"] = int(parts[1])
                except: pass
        elif line.startswith("total"):
            try:
                res["total"] = int(line.split()[1])
            except: pass
        elif line.startswith("fail:"):
            # fail: [1, 2, 3] -> These are INDICES in the dataset loaded by the script
            try:
                content = line.split(":", 1)[1].strip()
                res["fail_ids_indices"] = json.loads(content) # Store indices temporarily
            except:
                pass
    return res


def map_indices_to_ids(indices, evaluation_type):
    """
    Maps dataset indices (returned by scripts) to real Task IDs.
    This requires knowing which dataset the script iterated over.
    
    eval_unit: load_dataset("openai_humaneval", split="test") -> 164 items
    eval_fuzzing: load_dataset("openai_humaneval", split="test") + SecurityEval (from local)
    eval_static: load_dataset("openai_humaneval", split="test") + SecurityEval (from local)
    
    Wait, eval_unit only iterates HumanEval.
    eval_fuzzing and eval_static iterate combined (HumanEval + SecurityEval).
    """
    from datasets import load_dataset
    
    # Load HumanEval (standard order)
    humaneval = list(load_dataset("openai_humaneval", split="test"))
    
    # Load SecurityEval
    securityeval_path = "SecurityEval.jsonl"
    securityeval = []
    if os.path.exists(securityeval_path):
        with open(securityeval_path, 'r', encoding='utf-8') as f:
            for line in f:
                securityeval.append(json.loads(line))
                
    if evaluation_type == 'unit_test':
        dataset = humaneval
    else:
        dataset = humaneval + securityeval
        
    mapped_ids = []
    for idx in indices:
        if 0 <= idx < len(dataset):
            item = dataset[idx]
            # ID field name varies
            tid = item.get('task_id') or item.get('ID')
            mapped_ids.append(tid)
            
    return mapped_ids


def run_subprocess_eval(script_name, temp_file):
    print(f"Running {script_name} evaluation...")
    try:
        # The scripts expect the input file path as argument
        # They print results to stdout
        output = subprocess.check_output([sys.executable, SCRIPTS[script_name], temp_file], text=True)
        # print(output) # Debug
        
        parsed = parse_script_output(output)
        
        # Map indices to IDs
        if "fail_ids_indices" in parsed:
            indices = parsed.pop("fail_ids_indices")
            parsed["fail_ids"] = map_indices_to_ids(indices, script_name)
        else:
            parsed["fail_ids"] = []
            
        return parsed
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(e.output)
        return {"success": 0, "fixed": 0, "total": 0, "fail_ids": ["ERROR_EXECUTION"]}
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return {"success": 0, "fixed": 0, "total": 0, "fail_ids": ["ERROR_UNKNOWN"]}


def save_intermediate_results(model_name, results):
    """Saves results incrementally to summary file."""
    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except:
            summary = {}
    else:
        summary = {}
        
    summary[model_name] = results
    
    try:
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        # print(f"Intermediate results saved.")
    except Exception as e:
        print(f"Error saving intermediate results: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_scg_eval.py <input_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    temp_file, is_baseline, model_name = load_and_prepare_data(input_path)
    
    print(f"Start Evaluation for: {model_name}")
    print(f"Mode: {'Baseline' if is_baseline else 'SCG/Standalone'}")
    
    # Load existing results for resume
    results = {}
    if os.path.exists(SUMMARY_FILE):
        try:
            with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                results = summary.get(model_name, {})
        except:
            pass
    
    # 1. API Calls
    # Check if already evaluated
    if "avg_api_calls" in results and results["avg_api_calls"] > 0:
        print(f"Skipping API Calls (already evaluated): {results['avg_api_calls']}")
    else:
        results["avg_api_calls"] = evaluate_api_calls(temp_file, is_baseline)
        print(f"Avg API Calls: {results['avg_api_calls']}")
        save_intermediate_results(model_name, results)
    
    # 2. Unit Test (HumanEval)
    def run_and_stream(script_name, temp_file):
        print(f"Running {script_name} evaluation...")
        cmd = [sys.executable, SCRIPTS[script_name], temp_file]
        
        full_output = []
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            print(line, end='') # Stream to console
            full_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error running {script_name}, exit code: {process.returncode}")
            return {"success": 0, "fixed": 0, "total": 0, "fail_ids": ["ERROR_EXECUTION"], "fail_count": 0}
            
        output_str = "".join(full_output)
        parsed = parse_script_output(output_str)
        
        if "fail_ids_indices" in parsed:
            indices = parsed.pop("fail_ids_indices")
            parsed["fail_ids"] = map_indices_to_ids(indices, script_name)
        else:
            parsed["fail_ids"] = []
            
        parsed["fail_count"] = len(parsed["fail_ids"])
        
        # Log failure details if available in raw output
        # But parse_script_output doesn't extract details yet.
        # Let's save the full output to a log file for debugging
        log_file = f"{temp_file}.{script_name}.log"
        with open(log_file, 'w') as f:
            f.write(output_str)
        print(f"  [Log saved to {log_file}]")
        
        return parsed

    # 2. Unit Test
    if "unit_test" in results and results["unit_test"].get("total", 0) > 0:
        ut_res = results["unit_test"]
        print(f"Skipping Unit Test (already evaluated): Success={ut_res['success']}, Fixed={ut_res['fixed']}, Fail={ut_res.get('fail_count', len(ut_res.get('fail_ids', [])))}")
    else:
        ut_res = run_and_stream("unit_test", temp_file)
        results["unit_test"] = ut_res
        print(f"Unit Test: Success={ut_res['success']}, Fixed={ut_res['fixed']}, Fail={ut_res['fail_count']}")
        save_intermediate_results(model_name, results)
    
    # 3. Static Analysis
    if "static_analysis" in results and results["static_analysis"].get("total", 0) > 0:
        sa_res = results["static_analysis"]
        print(f"Skipping Static Analysis (already evaluated): Success={sa_res['success']}, Fixed={sa_res['fixed']}, Fail={sa_res.get('fail_count', len(sa_res.get('fail_ids', [])))}")
    else:
        sa_res = run_and_stream("static_analysis", temp_file)
        results["static_analysis"] = sa_res
        print(f"Static Analysis: Success={sa_res['success']}, Fixed={sa_res['fixed']}, Fail={sa_res['fail_count']}")
        save_intermediate_results(model_name, results)
        
    # 4. Fuzzing
    if "fuzzing" in results and results["fuzzing"].get("total", 0) > 0:
        ft_res = results["fuzzing"]
        print(f"Skipping Fuzzing (already evaluated): Success={ft_res['success']}, Fixed={ft_res['fixed']}, Fail={ft_res.get('fail_count', len(ft_res.get('fail_ids', [])))}")
    else:
        ft_res = run_and_stream("fuzzing", temp_file)
        results["fuzzing"] = ft_res
        print(f"Fuzzing: Success={ft_res['success']}, Fixed={ft_res['fixed']}, Fail={ft_res['fail_count']}")
        save_intermediate_results(model_name, results)
    
    print(f"Evaluation finished. Results saved to {SUMMARY_FILE}")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()
