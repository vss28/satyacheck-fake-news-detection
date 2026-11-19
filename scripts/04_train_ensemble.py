"""
Step 4 — Train the Ensemble Combiner
Loads both trained models and generates predictions on the validation set.
Trains a LogisticRegression combiner on:
  [fake_prob, real_prob, sentiment_score, speaker_cred_score]
Evaluates final ensemble on the test set.
Saves ensemble to: models/ensemble/
"""

import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data/merged"
ROBERTA_DIR   = "models/roberta_fakenews"
SENTI_DIR     = "cardiffnlp/twitter-roberta-base-sentiment"
ENSEMBLE_DIR  = "models/ensemble"
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

ROBERTA_MAX_LEN = 256
SENTI_MAX_LEN   = 128
BATCH_SIZE      = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Re-declare RoBERTa model class (must match 02_train_roberta.py) ──────────
class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        hidden = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ─── Dataset helper ────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

# ─── Inference helpers ─────────────────────────────────────────────────────────
@torch.no_grad()
def get_roberta_probs(model, tokenizer, texts, max_len=256, batch_size=32):
    """Returns Nx2 numpy array: [[real_prob, fake_prob], ...]"""
    model.eval()
    ds     = TextDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    probs  = []
    for batch in tqdm(loader, desc="  RoBERTa inference", leave=False):
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE)
        )
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)   # shape (N, 2)

@torch.no_grad()
def get_sentiment_probs(model, tokenizer, texts, max_len=128, batch_size=32):
    """Returns Nx2 numpy array: [[neg_prob, pos_prob], ...]"""
    model.eval()
    ds     = TextDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    probs  = []
    for batch in tqdm(loader, desc="  Sentiment inference", leave=False):
        out = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE)
        )
        p = torch.softmax(out.logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)   # shape (N, 2)

# ─── Build features ───────────────────────────────────────────────────────────
def build_features(df, roberta_model, roberta_tok, senti_model, senti_tok):
    texts = df["text"].tolist()

    fake_probs  = get_roberta_probs(roberta_model, roberta_tok, texts, ROBERTA_MAX_LEN, BATCH_SIZE)
    senti_probs = get_sentiment_probs(senti_model, senti_tok, texts, SENTI_MAX_LEN, BATCH_SIZE)

    # speaker_cred_score: use column if present, else 0.5
    if "speaker_cred_score" in df.columns:
        cred = df["speaker_cred_score"].fillna(0.5).to_numpy().reshape(-1, 1)
    else:
        cred = np.full((len(df), 1), 0.5)

    # Feature matrix: [real_prob, fake_prob, neg_sentiment, pos_sentiment, speaker_cred]
    X = np.hstack([fake_probs, senti_probs, cred])   # shape (N, 5)
    return X

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"🖥️  Device: {DEVICE}")
    # Load datasets
    print("\n📂 Loading split data...")
    val_df  = pd.read_csv(os.path.join(DATA_DIR, "val.csv")).dropna(subset=["text"])
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])

    val_labels  = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    # Load RoBERTa
    print(f"\n🤖 Loading fake news model from {ROBERTA_DIR}...")
    roberta_tok   = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    roberta_model = FakeNewsClassifier().to(DEVICE)
    roberta_model.load_state_dict(
        torch.load(os.path.join(ROBERTA_DIR, "best_model.pt"), map_location=DEVICE, weights_only=True)
    )

    # Load Sentiment model
    print(f"\n💬 Loading sentiment model from {SENTI_DIR}...")
    senti_tok   = AutoTokenizer.from_pretrained(SENTI_DIR)
    senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_DIR, use_safetensors=True).to(DEVICE)

    # Build features
    print("\n🏗️  Building ensemble features (val set)...")
    X_val   = build_features(val_df,  roberta_model, roberta_tok, senti_model, senti_tok)
    y_val   = np.array(val_labels)

    print("\n🏗️  Building ensemble features (test set)...")
    X_test  = build_features(test_df, roberta_model, roberta_tok, senti_model, senti_tok)
    y_test  = np.array(test_labels)

    # Baseline: solo RoBERTa performance
    roberta_val_preds  = X_val[:, 1].round().astype(int)   # fake_prob > 0.5 → FAKE
    roberta_test_preds = X_test[:, 1].round().astype(int)
    print(f"\n📊 Solo RoBERTa — Val Acc:  {accuracy_score(y_val,  roberta_val_preds):.4f}")
    print(f"📊 Solo RoBERTa — Test Acc: {accuracy_score(y_test, roberta_test_preds):.4f}")

    # Scale features
    scaler = StandardScaler()
    X_val_scaled  = scaler.fit_transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble
    print("\n🧩 Training ensemble LogisticRegression on val features...")
    ensemble = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    ensemble.fit(X_val_scaled, y_val)

    # Evaluate
    ens_val_preds  = ensemble.predict(X_val_scaled)
    ens_test_preds = ensemble.predict(X_test_scaled)
    print(f"\n📊 Ensemble — Val Acc:  {accuracy_score(y_val,  ens_val_preds):.4f}")
    print(f"📊 Ensemble — Test Acc: {accuracy_score(y_test, ens_test_preds):.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, ens_test_preds, target_names=["REAL", "FAKE"]))

    # Save ensemble artifacts
    with open(os.path.join(ENSEMBLE_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    with open(os.path.join(ENSEMBLE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save val features for reference
    np.save(os.path.join(ENSEMBLE_DIR, "X_val.npy"),  X_val)
    np.save(os.path.join(ENSEMBLE_DIR, "X_test.npy"), X_test)

    print(f"\n✅ Ensemble saved to: {ENSEMBLE_DIR}")

if __name__ == "__main__":
    main()
