import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from codebert_decision import CodeBERTMOE, train, load_and_split_data_moe, CodeDataset


class MOEDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample['input']
        labels = sample['labels']
        
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # if pos_weight is not None:
            #     loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            #     loss = loss_fct(outputs['logits'], labels.float())
            # else:
            #     loss = outputs['loss']
            loss = outputs['loss']
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    MODEL_NAME = 'microsoft/codebert-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 2e-5
    
    DATA_DIR = "for_smodel_training"
    OUTPUT_DIR = "task2_moe_scheduler/checkpoints"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading tokenizer and model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = CodeBERTMOE.from_pretrained(MODEL_NAME, config=config).to(device)
    
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model_moe.pt')
    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Checkpoint loaded successfully!")
    
    print("\nLoading datasets...")
    # Using load_and_split_data_moe which internally calls status_to_label
    # This function handles the raw json files in for_smodel_training
    (train_codes, train_labels), (val_codes, val_labels) = load_and_split_data_moe(DATA_DIR)
    
    # Use CodeDataset (list-based) instead of MOEDataset (file-based)
    train_dataset = CodeDataset(
        train_codes, train_labels, tokenizer, MAX_LENGTH
    )
    val_dataset = CodeDataset(
        val_codes, val_labels, tokenizer, MAX_LENGTH
    )
    
    # MOEDataset class is no longer needed since we use CodeDataset from codebert_decision.py
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    try_times = 0
    
    # MOEDataset definition is no longer used, we can keep it or remove it.
    # We will use CodeDataset from codebert_decision which handles lists.
    # The MOEDataset class was handling JSONL files directly.
    # Now we load everything into memory with load_and_split_data_moe.
    
    # Check if we need to remove MOEDataset class to avoid confusion.
    # For now, we just don't use it.
    
    print("\nComputing pos_weight from training data...")
    
    # Calculate pos_weight
    # train_dataset[i]['labels'] is a tensor
    # We need to iterate over the dataset to sum labels
    
    pos_counts = torch.zeros(3)
    # Using a larger batch size for quick summation or just iterating
    temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*4, shuffle=False)
    for batch in temp_loader:
        pos_counts += batch['labels'].sum(dim=0)
    neg_counts = len(train_dataset) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1e-8)).to(device)
    print(f"pos_weight: {pos_weight.tolist()}")

    print("\nStarting training...")
    # 设置 pos_weight 到 model 中，避免在 train 循环中重复计算 loss
    if hasattr(model, 'pos_weight'):
         # 如果模型支持，直接设置属性
         model.pos_weight = pos_weight
    
    # 传递 pos_weight=None 给 train 函数，因为已经在 forward 中处理了（如果设置了的话）
    # 或者如果没法设置属性，保持原样传递 pos_weight
    
    # 为了保险，我们还是传递 pos_weight 给 train，虽然会有一点点重复计算开销，但逻辑最安全
    # 优化方案：修改 train 函数不传递 pos_weight，而是依赖 model.forward
    
    # Let's set it as an attribute to be safe and cleaner
    model.pos_weight = pos_weight

    for epoch in range(start_epoch, EPOCHS):
        # Pass pos_weight to train/evaluate functions
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(OUTPUT_DIR, 'best_model_moe.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to {model_path}")
            try_times = 0
        else:
            try_times += 1
            if try_times >= 5:
                print("Early Stopping!")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model weights
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_moe.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
    
    results = {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_log.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return model


if __name__ == "__main__":
    main()
