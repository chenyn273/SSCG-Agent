from tqdm.auto import tqdm
import json
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel, AutoConfig, RobertaPreTrainedModel
import torch.nn.functional as F


SPLIT = '<SPLIT>'

class CodeBERTMultiTask(RobertaPreTrainedModel):
    def __init__(self, config, dropout_prob=0.1):
        super().__init__(config)
        self.roberta = AutoModel.from_config(config)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.classifier1 = nn.Linear(config.hidden_size, 1)  # 功能测试
        self.classifier2 = nn.Linear(config.hidden_size, 1)  # 静态分析
        self.classifier3 = nn.Linear(config.hidden_size, 1)  # 模糊测试
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output  # 使用pooler输出
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用CLS token的输出，避免 pooler_output 为 None

        pooled_output1 = self.dropout1(pooled_output)
        pooled_output2 = self.dropout2(pooled_output)
        pooled_output3 = self.dropout3(pooled_output)
        
        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)
        logits3 = self.classifier3(pooled_output3)

        logits = torch.cat([logits1, logits2, logits3], dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return {'loss': loss, 'logits': logits}


class ExpertMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CodeBERTMOE(RobertaPreTrainedModel):
    def __init__(self, config, dropout_prob=0.1, num_experts=3, expert_hidden_size=32, decision_threshold=0.5):
        super().__init__(config)
        # 1. 修复权重加载：使用 config 初始化空壳，权重由外层 from_pretrained 加载
        self.roberta = AutoModel.from_config(config)
        self.decision_threshold = decision_threshold
        
        self.experts = nn.ModuleList([
            ExpertMLP(config.hidden_size, expert_hidden_size) 
            for _ in range(num_experts)
        ])
        
        # 2. 修复静态门控：改为根据输入动态生成权重的线性层
        self.gate = nn.Linear(config.hidden_size, num_experts)
        
        # 5. 修复冗余设计：直接对融合后的向量进行分类
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(expert_hidden_size, 3)
        )
        
        # 初始化自定义层的权重
        self._init_moe_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.pooler_output
        pooled_output = outputs.last_hidden_state[:, 0, :] # 使用CLS token的输出，避免 pooler_output 为 None
        
        # [batch_size, num_experts, expert_hidden_size]
        expert_outputs = torch.stack([expert(pooled_output) for expert in self.experts], dim=1)
        
        # [batch_size, num_experts]
        gate_logits = self.gate(pooled_output)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 加权融合: [batch_size, expert_hidden_size]
        # sum( [batch, experts, 1] * [batch, experts, hidden] ) -> [batch, hidden]
        fused_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        # [batch_size, 3]
        logits = self.classifier(fused_output)
        probs = torch.sigmoid(logits)
        decision = self._get_decision(probs)
        
        loss = None
        if labels is not None:
            # 支持传入 pos_weight
            pos_weight = getattr(self, 'pos_weight', None)
            if pos_weight is not None:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {'loss': loss, 'logits': logits, 'probs': probs, 'decision': decision}
    
    def _get_decision(self, final_probs):
        # 抛弃 for 循环，利用 PyTorch 底层 C++ 向量化计算
        return (final_probs >= self.decision_threshold).float()

    def _init_moe_weights(self):
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.fc1.weight)
            nn.init.zeros_(expert.fc1.bias)
            nn.init.xavier_uniform_(expert.fc2.weight)
            nn.init.zeros_(expert.fc2.bias)
        
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            code,
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 训练函数


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for i, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        # loss 已经在 model.forward 中计算过了
        loss = outputs['loss']
        
        total_loss += loss.item()

        loss.backward()
        
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 实时更新进度条信息
        progress_bar.set_postfix({
            'batch_loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(i+1):.4f}" 
        })

    return total_loss / len(dataloader)
# 推理函数


def predict(code, model, tokenizer, device, max_length=512):
    model.eval()

    encoding = tokenizer.encode_plus(
        code,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

    logits = outputs['logits']
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {
        'unit_test_agent': probs[0],
        'static_analysis_agent': probs[1],
        'fuzzing_agent': probs[2]
    }


# 新增数据加载函数


def load_and_split_data(folder_path, test_size=0.2, val_size=0.5, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_train = []
    all_val = []
    all_test = []

    # 遍历文件夹中所有jsonl文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(folder_path, filename)
        samples = []

        # 读取单个文件
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                label = [
                    data['unit_test_status'],
                    data['static_analysis_status'],
                    data['fuzzing_test_status']
                ]
                samples.append((data['code'], label))

        # 打乱数据
        random.shuffle(samples)

        # 分割训练集和临时测试集
        train_data, temp_data = train_test_split(
            samples,
            test_size=test_size,
            random_state=random_seed
        )

        # 分割验证集和测试集
        val_data, test_data = train_test_split(
            temp_data,
            test_size=val_size,
            random_state=random_seed
        )

        all_train.extend(train_data)
        all_val.extend(val_data)
        all_test.extend(test_data)

    # 解包数据
    def unpack(data):
        codes = [d[0] for d in data]
        labels = [d[1] for d in data]
        return codes, labels

    return unpack(all_train), unpack(all_val), unpack(all_test)
# 在测试集上评估


def code_analysis_predictor(code_str, model, tokenizer, device, max_length=512):
    """
    代码分析预测函数
    参数：
        code_str: 需要分析的代码字符串
        model: 加载好的训练模型
        tokenizer: 代码分词器
        device: 计算设备
        max_length: 最大序列长度（默认512）

    返回：
        dict: 包含三类测试方法的预测结果
            {
                "unit_test_agent": True/False,
                "static_analysis_agent": True/False, 
                "fuzzing_agent": True/False
            }
    """
    # 输入验证
    if not isinstance(code_str, str) or len(code_str.strip()) == 0:
        raise ValueError("输入必须是有效的非空代码字符串")

    # 执行预测
    return predict(code_str, model, tokenizer, device, max_length)


# 修改后的主程序部分
def run_train():
    # 配置参数
    MODEL_NAME = 'microsoft/codebert-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 5e-5
    DATA_FOLDER = "/Users/chenyn/博/论文/CoderAgents/results/data_for_train_decision_model"  # 修改为实际路径

    # 加载数据
    (train_codes, train_labels), (val_codes, val_labels), (test_codes,
                                                           test_labels) = load_and_split_data(DATA_FOLDER)

    # 初始化模型和tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = CodeBERTMultiTask.from_pretrained(
        MODEL_NAME, config=config).to(device)

    # 创建DataLoader
    train_dataset = CodeDataset(
        train_codes, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = CodeDataset(val_codes, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = CodeDataset(test_codes, test_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 添加验证函数
    def evaluate(model, dataloader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

                loss = outputs['loss']
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # 修改后的训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    try_times = 0
    for epoch in range(EPOCHS):
        # 训练阶段
        train_loss = train(model, train_loader, optimizer, device)

        # 验证阶段
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            try_times = 0
        else:
            try_times += 1
            if try_times >= 5:
                print("Early Stopping!")
                break


def run_predict(code, model, tokenizer, device):
    prediction = code_analysis_predictor(code, model, tokenizer, device)
    return prediction['unit_test_agent'], prediction['static_analysis_agent'], prediction['fuzzing_agent']


def predict_moe(code, model, tokenizer, device, max_length=512):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        code,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    probs = outputs['probs'].cpu().numpy()[0]
    decision = outputs['decision'].cpu().numpy()[0]
    order = list(reversed(np.argsort(probs)))
    
    return {
        'unit_test_agent': float(probs[0]),
        'static_analysis_agent': float(probs[1]),
        'fuzzing_agent': float(probs[2]),
        'decision': [float(decision[0]), float(decision[1]), float(decision[2])],
        'order': [int(order[0]), int(order[1]), int(order[2])]
    }


def status_to_label(status) -> int:
    # 逻辑修正：1 表示需要执行（失败），0 表示不需要执行（成功/跳过）
    # BCEWithLogitsLoss: 1 = 正类(Positive), 0 = 负类(Negative)
    # 我们希望预测“是否需要执行”，所以“失败”=需要执行=正类=1，“成功”=不需要执行=负类=0
    if status is True: # status=True 表示通过(Success)，不需要执行 -> 0
        return 0
    elif status is False: # status=False 表示失败(Fail)，需要执行 -> 1
        return 1
    elif status == "success" or status == "skipped":
        return 0
    else: # 其他情况视为失败，需要执行 -> 1
        return 1


def load_and_split_data_moe(folder_path, test_size=0.2, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_samples = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                ut_label = status_to_label(data['unit_test_status'])
                sa_label = status_to_label(data['static_analysis_status'])
                ft_label = status_to_label(data['fuzzing_test_status'])
                
                label = [ut_label, sa_label, ft_label]
                
                all_samples.append((data['code'], label))
    
    random.shuffle(all_samples)
    
    train_data, val_data = train_test_split(
        all_samples,
        test_size=test_size,
        random_state=random_seed
    )
    
    def unpack(data):
        codes = [d[0] for d in data]
        labels = [d[1] for d in data]
        return codes, labels
    
    return unpack(train_data), unpack(val_data)


def load_prepared_data_moe(data_dir):
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    def read(path):
        codes = []
        labels = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                label = data.get('labels', [])
                labels.append(label[:3])
                codes.append(data['input'])
        return codes, labels
    return read(train_path), read(val_path)


def run_train_moe(data_folder, output_dir='./checkpoints'):
    MODEL_NAME = 'microsoft/codebert-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 24
    EPOCHS = 50
    LR = 2e-5
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(data_folder, "train.jsonl")):
        (train_codes, train_labels), (val_codes, val_labels) = load_prepared_data_moe(data_folder)
    else:
        (train_codes, train_labels), (val_codes, val_labels) = load_and_split_data_moe(data_folder)
    
    print(f"Training samples: {len(train_codes)}")
    print(f"Validation samples: {len(val_codes)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = CodeBERTMOE.from_pretrained(MODEL_NAME, config=config).to(device)
    
    train_dataset = CodeDataset(train_codes, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = CodeDataset(val_codes, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    def evaluate(model, dataloader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                total_loss += loss.item()
        return total_loss / len(dataloader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float('inf')
    try_times = 0
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_moe.pt'))
            try_times = 0
        else:
            try_times += 1
            if try_times >= 5:
                print("Early Stopping!")
                break
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    return model


def run_predict_moe(code, model, tokenizer, device):
    prediction = predict_moe(code, model, tokenizer, device)
    return prediction['unit_test_agent'], prediction['static_analysis_agent'], prediction['fuzzing_agent']
