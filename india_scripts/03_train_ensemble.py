"""
Step 3 (Indian Pipeline) — Train Ensemble Combiner
=================================================
Combines:
1. XLM-RoBERTa real/fake probabilities [real_prob, fake_prob]
2. Twitter-RoBERTa sentiment probabilities [neg_prob, neu_prob, pos_prob]
3. Language indicator feature [is_devanagari_or_indic]

Trains LogisticRegression combiner and saves artifacts to:
    models/indian/ensemble/ensemble.pkl
    models/indian/ensemble/scaler.pkl
"""

import os, re, pickle, sys
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

DATA_DIR = "data/indian/merged"
XLM_DIR = "models/indian/xlm_roberta"
# Check if local sentiment exists or fallback to HF
LOCAL_SENTI = "models/sentiment_roberta"
SENTI_DIR = LOCAL_SENTI if os.path.exists(LOCAL_SENTI) else "cardiffnlp/twitter-roberta-base-sentiment"
ENSEMBLE_DIR = "models/indian/ensemble"
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

XLM_MAX_LEN = 256
SENTI_MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndianFakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlm = AutoModel.from_pretrained("xlm-roberta-base")
        hidden = self.xlm.config.hidden_size
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

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [str(t) for t in texts]
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
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

@torch.no_grad()
def get_xlm_probs(model, tokenizer, texts):
    model.eval()
    ds = TextDataset(texts, tokenizer, XLM_MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs = []
    for b in tqdm(loader, desc="  XLM-RoBERTa inference", leave=False):
        logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs) if probs else np.empty((0, 2))

@torch.no_grad()
def get_senti_probs(model, tokenizer, texts):
    model.eval()
    ds = TextDataset(texts, tokenizer, SENTI_MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs = []
    for b in tqdm(loader, desc="  Sentiment inference", leave=False):
        out = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
        p = torch.softmax(out.logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs) if probs else np.empty((0, 3))

def extract_language_features(texts):
    """Detects Indian script presence (Devanagari, Bengali, Tamil, etc.)."""
    features = []
    for t in texts:
        text = str(t)
        # Check Devanagari or other common Indic Unicode blocks
        has_indic = bool(re.search(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]", text))
        features.append([1.0 if has_indic else 0.0])
    return np.array(features)

def build_features(df, xlm_model, xlm_tok, senti_model, senti_tok):
    texts = df["text"].tolist()
    xlm_probs = get_xlm_probs(xlm_model, xlm_tok, texts)      # [real, fake] -> 2
    senti_probs = get_senti_probs(senti_model, senti_tok, texts) # [neg, neu, pos] -> 3
    lang_feat = extract_language_features(texts)             # [is_indic] -> 1
    return np.hstack([xlm_probs, senti_probs, lang_feat])    # total: 6 features

def main():
    print(f"🖥️  Device: {DEVICE}")
    val_file = os.path.join(DATA_DIR, "val.csv")
    test_file = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(val_file):
        print(f"❌ Could not find {val_file}. Please run data prep first.")
        return

    val_df = pd.read_csv(val_file).dropna(subset=["text", "label"])
    test_df = pd.read_csv(test_file).dropna(subset=["text", "label"])

    model_pt = os.path.join(XLM_DIR, "best_model.pt")
    if not os.path.exists(model_pt):
        print(f"❌ XLM-RoBERTa model checkpoint not found at {model_pt}. Please run `python india_scripts/02_train_xlm_roberta.py` first.")
        return

    print("🤖 Loading trained XLM-RoBERTa model...")
    xlm_tok = AutoTokenizer.from_pretrained(XLM_DIR)
    xlm_model = IndianFakeNewsClassifier().to(DEVICE)
    xlm_model.load_state_dict(torch.load(model_pt, map_location=DEVICE, weights_only=True))

    print(f"💬 Loading sentiment model from {SENTI_DIR}...")
    senti_tok = AutoTokenizer.from_pretrained(SENTI_DIR)
    senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_DIR).to(DEVICE)

    print("\n🏗️  Building features for Validation Set...")
    X_val = build_features(val_df, xlm_model, xlm_tok, senti_model, senti_tok)
    y_val = val_df["label"].to_numpy().astype(int)

    print("🏗️  Building features for Test Set...")
    X_test = build_features(test_df, xlm_model, xlm_tok, senti_model, senti_tok)
    y_test = test_df["label"].to_numpy().astype(int)

    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n🧩 Training Logistic Regression combiner on Indian features...")
    ensemble = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    ensemble.fit(X_val_scaled, y_val)

    val_preds = ensemble.predict(X_val_scaled)
    test_preds = ensemble.predict(X_test_scaled)

    print(f"\n📊 Indian Ensemble — Val Acc:  {accuracy_score(y_val, val_preds):.4f}")
    print(f"📊 Indian Ensemble — Test Acc: {accuracy_score(y_test, test_preds):.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["REAL", "FAKE"]))

    with open(os.path.join(ENSEMBLE_DIR, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    with open(os.path.join(ENSEMBLE_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Indian ensemble artifacts saved to {ENSEMBLE_DIR}")

if __name__ == "__main__":
    main()
