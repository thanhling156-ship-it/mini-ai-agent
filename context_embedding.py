import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

# ==========================================
# 1. SYSTEM CONFIGURATION & HYPERPARAMETERS
# ==========================================
VOCAB_SIZE = 2000          
MAX_LEN = 32               
EMBEDDING_DIM = 128        
NUM_HEADS = 4              
NUM_LAYERS = 4             
BATCH_SIZE = 32
EPOCHS = 100               # Optimized from 500 to 300 due to Cosine Learning Rate
LEARNING_RATE = 5e-4         # Started with a higher LR for faster initial learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create temporary directory for the tokenizer structure
os.makedirs("./bert_tokenizer", exist_ok=True)

# ==========================================
# 2. TARGET VOCABULARY DEFINITION (ENGLISH)
# ==========================================
# Natural English sentence templates for User Requests
# Phân loại rõ ràng template theo nhóm hành động để tránh rác dữ liệu
templates_close = [
    # Các mẫu câu duy nhất sau khi gom nhóm
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

actions_close = ["close", "turn off", "kill", "shutdown", "terminate", "exit", "stop", "end", "quit", "shut down"]
actions_open = ["open", "launch", "start", "run", "bring up", "fire up", "pull up", "load", "boot up", "open up"]
apps = ["terminal", "task manager", "chrome"]

augmented_corpus = set()

# Sinh dữ liệu cho nhóm CLOSE với 2 vòng lặp (template -> action -> app)
for temp in templates_close:
    for action in actions_close:
        for app in apps:
            sentence = temp.format(action=action, app=app)
            augmented_corpus.add(sentence)

# Sinh dữ liệu cho nhóm OPEN với 2 vòng lặp (template -> action -> app)
for temp in templates_open:
    for action in actions_open:
        for app in apps:
            sentence = temp.format(action=action, app=app)
            augmented_corpus.add(sentence)

corpus_bert = list(augmented_corpus)
print(f"-> Total generated sentences for BERT: {len(corpus_bert)}.")

# Step 1: Extract all unique words from the corpus
words_in_corpus = set()
for sentence in corpus_bert:
    for word in sentence.split():
        words_in_corpus.add(word)

# Step 2: Extended vocabulary to prevent overfitting and add ReAct functional words
extended_words = [
    # Extra applications / system tasks
    "excel", "word", "vlc", "cmd", "discord", "spotify", "photoshop", "calculator",
    # Extra action words / synonyms
    "stop", "reboot", "hide", "show", "activate", "minimize",
    # Fillers / Adverbs for contextual richness
    "now", "right", "immediately", "urgent", "quickly", "please", "me", "for", "can", "you", "could",
    
    # --- CRITICAL REACT FUNCTIONAL WORDS IN ENGLISH ---
    "step", "1", "2", "3", "success", "failed", "done", "admin", "privilege", "call"
]

# Merge and clean the final vocabulary list
final_vocabulary = sorted(list(words_in_corpus.union(set(extended_words))))

# Step 3: Create standard BERT vocab.txt structure
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
full_vocab_list = special_tokens + final_vocabulary

# Write directly to the target vocabulary file
vocab_path = "./bert_tokenizer/vocab.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    for token in full_vocab_list:
        f.write(token + "\n")

print(f"-> Target vocabulary successfully created with {len(full_vocab_list)} tokens.")

# Load into Fast Tokenizer directly from the local vocab file
tokenizer = BertTokenizerFast.from_pretrained(
    "./bert_tokenizer", 
    local_files_only=True,
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# ==========================================
# 3. CUSTOM DATASET & MASKING MECHANISM
# ==========================================
class CommandMaskedDataset(Dataset):
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
        
        labels_mask = torch.full(input_ids.shape, -100, dtype=torch.long)
        
        valid_indices = [
            i for i, token_id in enumerate(input_ids) 
            if token_id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        ]
        
        num_to_mask = max(1, int(len(valid_indices) * 0.15))
        
        if num_to_mask > 0:
            perm = torch.randperm(len(valid_indices))[:num_to_mask]
            mask_indices = [valid_indices[i] for i in perm]
            
            for idx_mask in mask_indices:
                labels_mask[idx_mask] = input_ids[idx_mask] 
                input_ids[idx_mask] = self.tokenizer.mask_token_id 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_mask
        }

dataset = CommandMaskedDataset(corpus_bert, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. INITIALIZE CUSTOM BERT MODEL (128D)
# ==========================================
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=EMBEDDING_DIM,          
    num_hidden_layers=NUM_LAYERS,       
    num_attention_heads=NUM_HEADS,      
    intermediate_size=EMBEDDING_DIM * 4,
    max_position_embeddings=MAX_LEN
)

model = BertForMaskedLM(config)
model.to(DEVICE)

# ==========================================
# 5. INITIALIZE OPTIMIZER & LR SCHEDULER
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Cosine LR Scheduler: Smoothly decays LR from 5e-4 to 1e-6 over 300 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
print("Starting BERT context embedding training...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # thêm dòng này để tránh lỗi exploding gradients
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step() # Update learning rate for the next epoch
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

print("\nBERT Training completed!")

# ==========================================
# 7. EXTRACT & SAVE CONTEXT EMBEDDINGS
# ==========================================
output_dir = "./bert_embeddings_output"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"-> Saved Hugging Face model formats to: {output_dir}")

model.eval()
with torch.no_grad():
    word_embeddings = model.bert.embeddings.word_embeddings.weight.cpu().numpy()

vectors_file_path = os.path.join(output_dir, "word_vectors_128d.txt")
with open(vectors_file_path, "w", encoding="utf-8") as f:
    for token, token_id in tokenizer.get_vocab().items():
        # Đảm bảo token_id hợp lệ trong ma trận trọng số đề phòng lỗi lệch vocab
        if token_id < len(word_embeddings):
            vector = word_embeddings[token_id]
            vector_str = " ".join([f"{val:.6f}" for val in vector])
            f.write(f"{token} {vector_str}\n")

print(f"-> Exported custom word vectors to: {vectors_file_path}")
print(f"-> Embedding matrix shape: {word_embeddings.shape}")