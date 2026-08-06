import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os

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

def extract_prompts_humaneval(data: List[Dict]) -> List[Tuple[str, str]]:
    prompts = []
    for item in data:
        task_id = item.get('task_id', '')
        prompt = item.get('prompt', '')
        prompts.append((task_id, prompt))
    return prompts

def extract_prompts_mbpp(data: List[Dict]) -> List[Tuple[str, str]]:
    prompts = []
    for item in data:
        task_id = f"MBPP/{item.get('task_id', '')}"
        prompt = item.get('text', '')
        prompts.append((task_id, prompt))
    return prompts

def extract_prompts_securityeval(data: List[Dict]) -> List[Tuple[str, str]]:
    prompts = []
    for item in data:
        task_id = item.get('ID', '')
        prompt = item.get('Prompt', '')
        prompts.append((task_id, prompt))
    return prompts

def extract_prompts_llmseceval(data: List[Dict]) -> List[Tuple[str, str]]:
    prompts = []
    for item in data:
        task_id = item.get('Prompt ID', '')
        prompt = item.get('LLM-generated NL Prompt', '')
        prompts.append((task_id, prompt))
    return prompts

def calculate_similarity_matrix(model: SentenceTransformer, 
                                 train_prompts: List[Tuple[str, str]], 
                                 test_prompts: List[Tuple[str, str]]) -> np.ndarray:
    train_texts = [p[1] for p in train_prompts]
    test_texts = [p[1] for p in test_prompts]
    
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    
    similarity_matrix = np.dot(train_embeddings, test_embeddings.T)
    
    return similarity_matrix

def find_duplicates(similarity_matrix: np.ndarray, 
                    train_prompts: List[Tuple[str, str]], 
                    test_prompts: List[Tuple[str, str]], 
                    threshold: float = 0.7) -> List[Dict]:
    duplicates = []
    
    for i, (train_id, train_prompt) in enumerate(train_prompts):
        max_sim = 0
        max_sim_test_id = ""
        max_sim_test_prompt = ""
        
        for j, (test_id, test_prompt) in enumerate(test_prompts):
            sim = similarity_matrix[i, j]
            if sim > max_sim:
                max_sim = sim
                max_sim_test_id = test_id
                max_sim_test_prompt = test_prompt
        
        if max_sim >= threshold:
            duplicates.append({
                'train_id': train_id,
                'train_prompt': train_prompt[:200] + '...' if len(train_prompt) > 200 else train_prompt,
                'test_id': max_sim_test_id,
                'test_prompt': max_sim_test_prompt[:200] + '...' if len(max_sim_test_prompt) > 200 else max_sim_test_prompt,
                'similarity': float(max_sim)
            })
    
    return duplicates

def main():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\nLoading benchmark data...")
    humaneval_data = load_jsonl('humaneval.jsonl')
    mbpp_data = load_jsonl('mbpp.jsonl')
    securityeval_data = load_jsonl('SecurityEval.jsonl')
    llmseceval_data = load_json('LLMSecEval-Prompts_dataset.json')
    
    print(f"HumanEval samples: {len(humaneval_data)}")
    print(f"MBPP samples: {len(mbpp_data)}")
    print(f"SecurityEval samples: {len(securityeval_data)}")
    print(f"LLMSecEval samples: {len(llmseceval_data)}")
    
    print("\nExtracting prompts...")
    humaneval_prompts = extract_prompts_humaneval(humaneval_data)
    mbpp_prompts = extract_prompts_mbpp(mbpp_data)
    securityeval_prompts = extract_prompts_securityeval(securityeval_data)
    llmseceval_prompts = extract_prompts_llmseceval(llmseceval_data)
    
    test_prompts = humaneval_prompts + securityeval_prompts
    print(f"Total test prompts: {len(test_prompts)}")
    
    results = {}
    
    print("\n" + "="*60)
    print("Processing MBPP vs Test benchmarks (HumanEval + SecurityEval)")
    print("="*60)
    
    mbpp_sim_matrix = calculate_similarity_matrix(model, mbpp_prompts, test_prompts)
    mbpp_duplicates = find_duplicates(mbpp_sim_matrix, mbpp_prompts, test_prompts, threshold=0.7)
    
    results['mbpp'] = {
        'total_samples': len(mbpp_prompts),
        'duplicates_found': len(mbpp_duplicates),
        'duplicate_ids': [d['train_id'] for d in mbpp_duplicates],
        'duplicate_details': mbpp_duplicates
    }
    
    print(f"\nMBPP Results:")
    print(f"  Total samples: {len(mbpp_prompts)}")
    print(f"  Duplicates found (similarity >= 0.7): {len(mbpp_duplicates)}")
    print(f"  Samples to keep: {len(mbpp_prompts) - len(mbpp_duplicates)}")
    
    if mbpp_duplicates:
        print(f"\n  Top 5 duplicates by similarity:")
        sorted_dups = sorted(mbpp_duplicates, key=lambda x: x['similarity'], reverse=True)[:5]
        for dup in sorted_dups:
            print(f"    {dup['train_id']} <-> {dup['test_id']}: {dup['similarity']:.4f}")
    
    print("\n" + "="*60)
    print("Processing LLMSecEval vs Test benchmarks (HumanEval + SecurityEval)")
    print("="*60)
    
    llmseceval_sim_matrix = calculate_similarity_matrix(model, llmseceval_prompts, test_prompts)
    llmseceval_duplicates = find_duplicates(llmseceval_sim_matrix, llmseceval_prompts, test_prompts, threshold=0.7)
    
    results['llmseceval'] = {
        'total_samples': len(llmseceval_prompts),
        'duplicates_found': len(llmseceval_duplicates),
        'duplicate_ids': [d['train_id'] for d in llmseceval_duplicates],
        'duplicate_details': llmseceval_duplicates
    }
    
    print(f"\nLLMSecEval Results:")
    print(f"  Total samples: {len(llmseceval_prompts)}")
    print(f"  Duplicates found (similarity >= 0.7): {len(llmseceval_duplicates)}")
    print(f"  Samples to keep: {len(llmseceval_prompts) - len(llmseceval_duplicates)}")
    
    if llmseceval_duplicates:
        print(f"\n  Top 5 duplicates by similarity:")
        sorted_dups = sorted(llmseceval_duplicates, key=lambda x: x['similarity'], reverse=True)[:5]
        for dup in sorted_dups:
            print(f"    {dup['train_id']} <-> {dup['test_id']}: {dup['similarity']:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_train = len(mbpp_prompts) + len(llmseceval_prompts)
    total_duplicates = len(mbpp_duplicates) + len(llmseceval_duplicates)
    
    print(f"\nTraining set statistics:")
    print(f"  MBPP: {len(mbpp_prompts)} samples, {len(mbpp_duplicates)} duplicates ({len(mbpp_duplicates)/len(mbpp_prompts)*100:.1f}%)")
    print(f"  LLMSecEval: {len(llmseceval_prompts)} samples, {len(llmseceval_duplicates)} duplicates ({len(llmseceval_duplicates)/len(llmseceval_prompts)*100:.1f}%)")
    print(f"  Total: {total_train} samples, {total_duplicates} duplicates ({total_duplicates/total_train*100:.1f}%)")
    print(f"  After deduplication: {total_train - total_duplicates} samples")
    
    print(f"\nTest set statistics:")
    print(f"  HumanEval: {len(humaneval_prompts)} samples")
    print(f"  SecurityEval: {len(securityeval_prompts)} samples")
    print(f"  Total: {len(test_prompts)} samples")
    
    with open('deduplication_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to deduplication_results.json")

if __name__ == "__main__":
    main()
