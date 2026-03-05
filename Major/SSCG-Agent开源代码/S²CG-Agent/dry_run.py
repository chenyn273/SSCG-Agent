import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import sys

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SSCG-Agent开源代码", "S²CG-Agent"))

from main import MultiAgentSystem
from codebert_decision import CodeBERTMOE
from llms import OpenAI_LLM

def run_dry_run(mode='collaborative'):
    print(f"Starting Dry Run: {mode} mode")
    
    # Configuration
    MODEL_PATH = 'task2_moe_scheduler/checkpoints/best_model_moe.pt'
    MODEL_NAME = 'microsoft/codebert-base' # Or local path if downloaded
    OUTPUT_DIR = f"SSCG-Agent开源代码/major_revision_results/{mode}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Model
    print("Loading Scheduler Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Note: CodeBERTMOE structure must match checkpoint
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = CodeBERTMOE.from_pretrained(MODEL_NAME, config=config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Data (Subset)
    print("Loading Data Subset...")
    humaneval_ds = load_dataset("openai_humaneval", split="test")
    # Take first 2 samples
    data_humaneval = [item for i, item in enumerate(humaneval_ds) if i < 2]
    
    # SecurityEval (mock load since file reading might be complex, or read first few lines)
    # Let's try reading the existing jsonl if available, otherwise skip for dry run
    security_file = "SecurityEval.jsonl" # Assuming relative path or need absolute
    # For dry run, let's just use Humaneval to test the flow
    dataset = data_humaneval
    print(f"Dry run dataset size: {len(dataset)}")
    
    # Setup LLM
    print("Setting up LLM (qwen-max)...")
    api_key = ""
    base_url = ""
    llm = OpenAI_LLM(api_key, 'qwen-max', base_url=base_url)
    
    # Run
    print("Running...")
    for i, entry in enumerate(dataset):
        print(f"Processing sample {i+1}/{len(dataset)}: {entry.get('task_id', 'unknown')}")
        system = MultiAgentSystem(entry, llm)
        # Use reduced iterations for dry run
        system.run(model, tokenizer, device, mode=mode, output_dir=OUTPUT_DIR, iterations=1)
        
    print(f"Dry Run {mode} Completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='collaborative')
    args = parser.parse_args()
    
    run_dry_run(args.mode)
