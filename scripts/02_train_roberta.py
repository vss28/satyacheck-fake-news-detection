"""
Step 2 — Fine-tune RoBERTa for Fake News Detection
Uses merged LIAR + ISOT dataset.
Custom classification head: Dropout → Linear(768,256) → GELU → Dropout → Linear(256,2)
Class-weighted loss for imbalanced data.
Freezes bottom 6 RoBERTa layers for faster training.
Saves model to: models/roberta_fakenews/
"""

import os, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/merged"
MODEL_DIR   = "models/roberta_fakenews"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_MODEL   = "roberta-base"
MAX_LEN      = 256          # 256 tokens: ~2x faster than 512, minimal accuracy loss
BATCH_SIZE   = 16           # safe for RTX 4060 8GB at 256 tokens
EPOCHS       = 5
LR           = 2e-5
WARMUP_PCT   = 0.1
WEIGHT_DECAY = 0.01
FREEZE_LAYERS = 6           # freeze bottom N of 12 RoBERTa layers
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts    = df["text"].tolist()
        self.labels   = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─── Model ───────────────────────────────────────────────────────────────────
class FakeNewsClassifier(nn.Module):
    def __init__(self, base_model_name: str, freeze_layers: int = 6):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(base_model_name)

        # Freeze bottom N encoder layers
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden = self.roberta.config.hidden_size   # 768 for roberta-base

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]   # [CLS] representation
        return self.classifier(cls_token)

# ─── Training Loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in tqdm(loader, desc="  Eval", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        total_loss  += loss.item()
        all_preds  += logits.argmax(dim=-1).cpu().tolist()
        all_labels += labels.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"🖥️  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print("\n📂 Loading merged data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_df = train_df.dropna(subset=["text"])
    val_df   = val_df.dropna(subset=["text"])
    test_df  = test_df.dropna(subset=["text"])

    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Tokenizer
    print(f"\n🔤 Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Datasets & Loaders
    train_ds = FakeNewsDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = FakeNewsDataset(val_df,   tokenizer, MAX_LEN)
    test_ds  = FakeNewsDataset(test_df,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Class weights (handles LIAR imbalance)
    labels_arr = train_df["label"].to_numpy()
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_arr)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    )
    print(f"\n⚖️  Class weights: REAL={class_weights[0]:.3f} FAKE={class_weights[1]:.3f}")

    # Model
    print(f"\n🤖 Loading {BASE_MODEL} (freezing bottom {FREEZE_LAYERS} layers)...")
    model = FakeNewsClassifier(BASE_MODEL, freeze_layers=FREEZE_LAYERS).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=WARMUP_PCT, anneal_strategy="cos"
    )

    # Training
    print(f"\n🏋️  Training for {EPOCHS} epochs...\n")
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_loss,   val_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"   ✅ New best model saved (val_acc={val_acc:.4f})")

    # Final test evaluation
    print(f"\n📊 Loading best model (epoch {best_epoch}, val_acc={best_val_acc:.4f})...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)

    print(f"\n🎯 Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["REAL", "FAKE"]))

    # Save model config info
    with open(os.path.join(MODEL_DIR, "config.txt"), "w") as f:
        f.write(f"base_model={BASE_MODEL}\n")
        f.write(f"max_len={MAX_LEN}\n")
        f.write(f"best_val_acc={best_val_acc:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
        f.write(f"freeze_layers={FREEZE_LAYERS}\n")

    print(f"\n✅ Model saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
