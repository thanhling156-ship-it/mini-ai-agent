import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, BertTokenizerFast, BertForMaskedLM

# ==========================================
# 1. GPT SYSTEM CONFIGURATION & HYPERPARAMETERS
# ==========================================
EMBEDDING_DIM = 128        # Fixed dimensions matching BERT's embeddings
MAX_LEN = 64               # Sequence length for Agent action trajectory
NUM_HEADS = 4              
NUM_LAYERS = 4             
BATCH_SIZE = 8             # Balanced batch size for expanded corpus
EPOCHS = 50             # Converges faster due to rich data distribution
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_OUTPUT_DIR = "./bert_embeddings_output"
GPT_OUTPUT_DIR = "./gpt_agent_output"
os.makedirs(GPT_OUTPUT_DIR, exist_ok=True)

# Ensure BERT outputs exist before copying weights
assert os.path.exists(BERT_OUTPUT_DIR), "Error: Cannot find bert_embeddings_output directory!"

# Reload Tokenizer from BERT output to sync vocabulary IDs
tokenizer = BertTokenizerFast.from_pretrained(BERT_OUTPUT_DIR)

# Inject special operational agent tokens into Tokenizer
special_tokens_dict = {'additional_special_tokens': ['[ACT]', '[END]']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = "[PAD]"

# ==========================================
# 2. DATA AUGMENTATION (REACT STRUCTURE IN ENGLISH)
# ==========================================
# Causal Structure: [CLS] Request [SEP] step x intent obj [ACT] result [SEP] ... [END]

# Tách biệt 2 bộ mẫu câu đã được tối ưu hóa duy nhất
templates_open = [
    "{action} {app}",
    "{action} {app} for me",
    "{action} {app} immediately",
    "could you {action} {app}",
    "{action} {app} please",
    "please {action} {app}",
    "{action} {app} for me please",
    "can you {action} {app}",
    "can you {action} {app} for me",
    "would you {action} {app}",
    "would you {action} {app} for me",
    "get {app} {action}",
    "I want to {action} {app}",
    "I need to {action} {app}",
    "I'd like to {action} {app}"
]

templates_close = [
    "{action} {app} please",
    "can you {action} {app}",
    "{action} the {app} process",
    "please {action} {app}",
    "{action} {app} right now",
    "{action} {app}",
    "could you {action} {app}",
    "would you {action} {app} for me",
    "can you {action} {app} for me",
    "please {action} {app} for me",
    "{action} {app} now",
    "I want to {action} {app}",
    "I need to {action} {app}"
]

# Chia hành động thành 2 nhóm rõ ràng
actions_open = {
    "open": "open", "launch": "open", "start": "open",
    "run": "open", "bring up": "open", "fire up": "open",
    "pull up": "open", "load": "open", "boot up": "open", "open up": "open"
}

actions_close = {
    "close": "close", "turn off": "close", "kill": "close",
    "shutdown": "close", "terminate": "close", "exit": "close",
    "stop": "close", "end": "close", "quit": "close", "shut down": "close"
}

app_mapping = {
    "terminal": "terminal",
    "task manager": "task manager",
    "chrome": "chrome"
}

generated_corpus = []

# --- SINH DỮ LIỆU CHO NHÓM OPEN ---
for temp in templates_open:
    for raw_act, react_act in actions_open.items():
        for raw_app, react_app in app_mapping.items():
            user_request = temp.format(action=raw_act, app=raw_app)
            
            # Kịch bản thành công
            react_success = f"[CLS] {user_request} [SEP] step 1 {react_act} {react_app} [ACT] success [SEP] done [END]"
            generated_corpus.append(react_success)
            
            # Kịch bản thất bại rồi thử lại
            react_fail = f"[CLS] {user_request} [SEP] step 1 {react_act} {react_app} [ACT] failed [SEP] step 2 call admin privilege [ACT] success [SEP] step 3 {react_act} {react_app} [ACT] success [SEP] done [END]"
            generated_corpus.append(react_fail)

# --- SINH DỮ LIỆU CHO NHÓM CLOSE ---
for temp in templates_close:
    for raw_act, react_act in actions_close.items():
        for raw_app, react_app in app_mapping.items():
            user_request = temp.format(action=raw_act, app=raw_app)
            
            # Kịch bản thành công
            react_success = f"[CLS] {user_request} [SEP] step 1 {react_act} {react_app} [ACT] success [SEP] done [END]"
            generated_corpus.append(react_success)
            
            # Kịch bản thất bại rồi thử lại
            react_fail = f"[CLS] {user_request} [SEP] step 1 {react_act} {react_app} [ACT] failed [SEP] step 2 call admin privilege [ACT] success [SEP] step 3 {react_act} {react_app} [ACT] success [SEP] done [END]"
            generated_corpus.append(react_fail)

corpus_gpt = list(set(generated_corpus))
print(f"-> Total generated ReAct trajectories for GPT Agent: {len(corpus_gpt)} lines.")


class GPTAgentReActDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Tạo bản sao nhãn từ input_ids
        labels = input_ids.clone()

        # Biến cờ hiệu kiểm soát vùng bối cảnh cố định của User
        is_user_zone = True
        
        for i in range(len(input_ids)):
            token_id = input_ids[i].item()

            # 1. KIỂM TRA ĐẦU TIÊN: Bắt gặp token [SEP] đầu tiên (kết thúc câu hỏi của user)
            # Phải đặt điều kiện này lên đầu để bắt đúng điểm chuyển giao vùng dữ liệu
            if token_id == self.tokenizer.sep_token_id and is_user_zone:
                is_user_zone = False
                labels[i] = -100  # Mask chính token [SEP] đầu tiên này
                continue

            # 2. KIỂM TRA THỨ HAI: Mask toàn bộ vùng câu hỏi của user (trước [SEP] đầu tiên)
            # Điều kiện này chỉ chạy khi hệ thống chưa đi qua token [SEP] đầu tiên
            if is_user_zone:
                labels[i] = -100
                continue

            # 3. KIỂM TRA THỨ BA: Mask token [PAD] để không tính loss khi chuỗi kết thúc
            if token_id == self.tokenizer.pad_token_id:
                labels[i] = -100
                continue
                
            # CÁC TOKEN CÒN LẠI (Gồm các từ hành động [ACT], các token [SEP] phân bước phía sau, [END]):
            # Giữ nguyên ID gốc trong ma trận labels để mô hình tự tính toán và học tập.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

dataset = GPTAgentReActDataset(corpus_gpt, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. INITIALIZE GPT & PARTIAL EMBEDDING WEIGHT TRANSFER
# ==========================================
config = GPT2Config(
    vocab_size=len(tokenizer),        # Complete vocabulary including new specialized tokens
    n_positions=MAX_LEN,
    n_embd=EMBEDDING_DIM,             
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids("[END]")
)

model_gpt = GPT2LMHeadModel(config)

print("Extracting contextual embedding weight matrix from pretrained BERT...")
model_bert = BertForMaskedLM.from_pretrained(BERT_OUTPUT_DIR)

with torch.no_grad():
    bert_embeddings_weight = model_bert.bert.embeddings.word_embeddings.weight.clone()
    num_bert_tokens = bert_embeddings_weight.size(0)
    
    # Initialize the expanded weight matrix using Normal Distribution instead of zeroing out
    gpt_embeddings_weight = torch.randn((len(tokenizer), EMBEDDING_DIM)) * 0.02
    
    # Overwrite the base matrix with inherited BERT vectors
    gpt_embeddings_weight[:num_bert_tokens, :] = bert_embeddings_weight
    
    # Inject combined weights into GPT Token Embeddings layer (wte)
    model_gpt.transformer.wte.weight.copy_(gpt_embeddings_weight)

print("-> Alignment successful. Discarding BERT cache.")
del model_bert

# ---- GRADIENT FILTER VIA BACKWARD HOOKS (FREEZE INHERITED WORDS ONLY) ----
model_gpt.transformer.wte.weight.requires_grad = True

# Move model TRƯỚC để grad_mask cùng device với weight
model_gpt.to(DEVICE)

# Matrix mask: 0.0 zeroes out gradients for old tokens, 1.0 allows updates for [ACT] and [END]
# Phải tạo SAU .to(DEVICE) để grad_mask cùng device với weight
grad_mask = torch.ones((len(tokenizer), 1), device=DEVICE)
grad_mask[:num_bert_tokens, :] = 0.0

def freeze_old_embeddings_hook(grad):
    return grad * grad_mask

model_gpt.transformer.wte.weight.register_hook(freeze_old_embeddings_hook)
print(f"-> Partials Locked: {num_bert_tokens} inherited tokens frozen. Training specialized token vectors only.")

# ==========================================
# 4. TRAINING LOOP (CAUSAL QKV ATTENTION)
# ==========================================
trainable_params = [p for p in model_gpt.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print("\nTraining GPT Agent for operational trajectory mapping...")
model_gpt.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model_gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_gpt.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

# Export weights and vocabulary
model_gpt.save_pretrained(GPT_OUTPUT_DIR)
tokenizer.save_pretrained(GPT_OUTPUT_DIR)
print(f"\nTraining finalized! GPT Agent artifacts exported to: {GPT_OUTPUT_DIR}")