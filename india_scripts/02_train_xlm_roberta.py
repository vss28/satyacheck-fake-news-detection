"""
Step 2 (Indian Pipeline) — Fine-tune XLM-RoBERTa for Indian Fake News Detection
=============================================================================
Uses merged Indian datasets (English + Hindi/Hinglish).
Backbone: xlm-roberta-base (Multilingual: 100 languages)
Custom classification head: Dropout -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, 2)
Saves model to: models/indian/xlm_roberta/
"""

import os, time, sys
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

DATA_DIR = "data/indian/merged"
MODEL_DIR = "models/indian/xlm_roberta"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_MODEL = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WARMUP_PCT = 0.1
WEIGHT_DECAY = 0.01
FREEZE_LAYERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndianFakeNewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class IndianFakeNewsClassifier(nn.Module):
    def __init__(self, base_model_name: str, freeze_layers: int = 6):
        super().__init__()
        self.xlm = AutoModel.from_pretrained(base_model_name)
        
        # Freeze bottom layers if requested
        if hasattr(self.xlm, "encoder") and hasattr(self.xlm.encoder, "layer"):
            for i, layer in enumerate(self.xlm.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        hidden = self.xlm.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / max(1, len(loader)), correct / max(1, total)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds += logits.argmax(dim=-1).cpu().tolist()
        all_labels += labels.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return total_loss / max(1, len(loader)), acc, all_preds, all_labels

def main():
    print(f"🖥️  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"❌ Could not find {train_path}. Please run `python india_scripts/01_prepare_data.py` first.")
        return

    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).dropna(subset=["text", "label"])
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text", "label"])

    print(f"\n📂 Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print(f"\n🔤 Loading Tokenizer ({BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = IndianFakeNewsDataset(train_df, tokenizer, MAX_LEN)
    val_ds = IndianFakeNewsDataset(val_df, tokenizer, MAX_LEN)
    test_ds = IndianFakeNewsDataset(test_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

    # Class balancing
    labels_arr = train_df["label"].to_numpy().astype(int)
    classes = np.unique(labels_arr)
    if len(classes) > 1:
        weights = compute_class_weight("balanced", classes=classes, y=labels_arr)
        weight_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    else:
        weight_tensor = torch.tensor([1.0, 1.0], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    print(f"\n🤖 Initializing {BASE_MODEL} with custom classification head...")
    model = IndianFakeNewsClassifier(BASE_MODEL, freeze_layers=FREEZE_LAYERS).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps, pct_start=WARMUP_PCT, anneal_strategy="cos")

    print(f"\n🏋️  Training for {EPOCHS} epochs...")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {elapsed:.0f}s")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"   💾 Checkpoint saved (val_acc={val_acc:.4f})")

    print(f"\n🎯 Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)

    print(f"🏁 Final Test Accuracy: {test_acc:.4f}")
    if len(np.unique(test_labels)) > 1:
        print(classification_report(test_labels, test_preds, target_names=["REAL", "FAKE"]))

    with open(os.path.join(MODEL_DIR, "config.txt"), "w", encoding="utf-8") as f:
        f.write(f"base_model={BASE_MODEL}\nmax_len={MAX_LEN}\nbest_val_acc={best_val_acc:.4f}\ntest_acc={test_acc:.4f}\n")

    print(f"\n✅ Indian XLM-RoBERTa model saved in {MODEL_DIR}")

if __name__ == "__main__":
    main()
