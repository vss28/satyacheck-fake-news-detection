"""
Indian Fake News Detection Pipeline — Inference
===============================================
Loads models for Indian context (XLM-RoBERTa + Sentiment + Source Credibility).
Supports Hindi, Hinglish, English, and regional Indian contexts.

Usage:
    python predict_india.py "UNESCO declares Indian national anthem best in the world"
    python predict_india.py --interactive
"""

import os, sys, pickle, argparse, re
import numpy as np
import torch
import torch.nn as nn

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

import transformers
transformers.logging.set_verbosity_error()

from india_scripts.source_credibility import (
    get_source_score,
    apply_source_adjustment,
    compute_calibrated_fake_probability,
    detect_forward_and_hoaxes
)

XLM_DIR = "models/indian/xlm_roberta"
SENTI_DIR = "models/sentiment_roberta" if os.path.exists("models/sentiment_roberta") else "cardiffnlp/twitter-roberta-base-sentiment"
ENSEMBLE_DIR = "models/indian/ensemble"

XLM_MAX_LEN = 256
SENTI_MAX_LEN = 128
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

class IndianFakeNewsPipeline:
    def __init__(self):
        print("🔄 Loading Indian Fake News Pipeline...")
        self.is_custom_trained = os.path.exists(os.path.join(XLM_DIR, "best_model.pt"))

        if self.is_custom_trained:
            print(f"   [Model] Loading custom-trained XLM-RoBERTa from {XLM_DIR}...")
            self.xlm_tok = AutoTokenizer.from_pretrained(XLM_DIR)
            self.xlm_model = IndianFakeNewsClassifier().to(DEVICE)
            self.xlm_model.load_state_dict(
                torch.load(os.path.join(XLM_DIR, "best_model.pt"), map_location=DEVICE, weights_only=True)
            )
        else:
            print("   [Notice] Custom XLM-RoBERTa weights not found yet. Using base XLM-RoBERTa head...")
            self.xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.xlm_model = IndianFakeNewsClassifier().to(DEVICE)

        self.xlm_model.eval()

        print(f"   [Emotion] Loading Sentiment Engine from {SENTI_DIR}...")
        self.senti_tok = AutoTokenizer.from_pretrained(SENTI_DIR)
        self.senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_DIR).to(DEVICE)
        self.senti_model.eval()

        ens_path = os.path.join(ENSEMBLE_DIR, "ensemble.pkl")
        scaler_path = os.path.join(ENSEMBLE_DIR, "scaler.pkl")

        if os.path.exists(ens_path) and os.path.exists(scaler_path):
            print(f"   [Ensemble] Loading Indian Logistic Combiner from {ENSEMBLE_DIR}...")
            try:
                with open(ens_path, "rb") as f:
                    self.ensemble = pickle.load(f)
                if not hasattr(self.ensemble, "multi_class"):
                    self.ensemble.multi_class = "auto"
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.has_ensemble = True
            except Exception as e:
                print(f"   [Warning] Failed to load ensemble ({e}), falling back to direct neural softmax.")
                self.has_ensemble = False
        else:
            self.has_ensemble = False
            print("   [Notice] Indian ensemble not trained yet. Pipeline will use direct neural softmax.")

        print("✅ Pipeline ready.\n")

    def detect_language(self, text: str) -> str:
        """Detects script / language format."""
        if re.search(r"[\u0900-\u097F]", text):
            return "Hindi / Devanagari 🇮🇳"
        elif re.search(r"[\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]", text):
            return "Regional Indic Script 🇮🇳"
        
        # Simple Hinglish check (Latin script with frequent Hindi romanized words)
        hinglish_words = {"kya", "nahi", "hain", "sarkar", "desh", "modi", "yojana", "paisa", "bhai", "shri", "bharat"}
        words = set(re.findall(r"\b\w+\b", text.lower()))
        if len(words.intersection(hinglish_words)) >= 2:
            return "Hinglish (Romanized Hindi) 🇮🇳"

        return "English (Indian Context) 🌐"

    @torch.no_grad()
    def _xlm_probs(self, text: str) -> np.ndarray:
        enc = self.xlm_tok(text, max_length=XLM_MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = self.xlm_model(enc["input_ids"], enc["attention_mask"])
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    @torch.no_grad()
    def _senti_probs(self, text: str) -> np.ndarray:
        enc = self.senti_tok(text, max_length=SENTI_MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = self.senti_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return torch.softmax(out.logits, dim=-1).cpu().numpy()[0]

    def predict(self, text: str, source: str = "") -> dict:
        """
        Runs complete prediction on Indian text with optional source credibility.
        """
        xlm_p = self._xlm_probs(text)      # [real, fake]
        senti_p = self._senti_probs(text)  # [neg, neu, pos]
        lang_str = self.detect_language(text)

        is_indic = 1.0 if "🇮🇳" in lang_str else 0.0

        raw_fake_prob = float(xlm_p[1])
        if self.has_ensemble:
            try:
                feats = np.hstack([xlm_p.reshape(1, -1), senti_p.reshape(1, -1), np.array([[is_indic]])])
                feats_scaled = self.scaler.transform(feats)
                comb_probs = self.ensemble.predict_proba(feats_scaled)[0]
                raw_fake_prob = float(comb_probs[1])
            except Exception as ens_err:
                raw_fake_prob = float(xlm_p[1])

        # Calibrated multi-evidence fusion (neural score + source credibility + viral hoaxes + forwards)
        calib = compute_calibrated_fake_probability(
            raw_fake_prob=raw_fake_prob,
            source=source,
            text=text,
            sentiment_neg_prob=float(senti_p[0])
        )

        verdict = calib["verdict"]
        confidence = calib["confidence"]
        fake_probability = calib["fake_probability"]
        real_probability = calib["real_probability"]
        source_label = calib["source_credibility"]
        source_score = calib["source_score"]
        risk_level = calib["risk_level"]
        matched_hoaxes = calib["matched_hoaxes"]
        is_forward = calib["is_forward"]

        # Sentiment label
        if senti_p[2] > senti_p[0] and senti_p[2] > senti_p[1]:
            sentiment = "Positive 😊"
        elif senti_p[0] > senti_p[1] and senti_p[0] > senti_p[2]:
            sentiment = "Negative 😟"
        else:
            sentiment = "Neutral 😐"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "raw_fake_prob": f"{raw_fake_prob * 100:.1f}%",
            "language": lang_str,
            "source_entered": source.strip() if source else "None (default 0.5)",
            "source_credibility": source_label,
            "source_score": source_score,
            "risk_level": risk_level,
            "matched_hoaxes": matched_hoaxes,
            "is_forward": is_forward,
            "sentiment": sentiment,
            "sentiment_neg_prob": f"{senti_p[0] * 100:.1f}%",
            "sentiment_neu_prob": f"{senti_p[1] * 100:.1f}%",
            "sentiment_pos_prob": f"{senti_p[2] * 100:.1f}%",
            "xlm_real_prob": f"{xlm_p[0] * 100:.1f}%",
            "xlm_fake_prob": f"{xlm_p[1] * 100:.1f}%",
            "is_custom_trained": self.is_custom_trained
        }

def main():
    parser = argparse.ArgumentParser(description="Indian Fake News Detector")
    parser.add_argument("text", nargs="?", default=None, help="Input news headline or text")
    parser.add_argument("--source", type=str, default="", help="Optional source URL or portal name")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal mode")
    args = parser.parse_args()

    pipeline = IndianFakeNewsPipeline()

    if args.interactive or not args.text:
        print("🇮🇳 Indian Fake News Detector (Interactive CLI)")
        print("   Supports: English, Hindi, Hinglish, Indic text\n")
        while True:
            t = input("\n📝 Enter news/claim (or 'exit'): ").strip()
            if t.lower() in ("exit", "quit", "q"):
                break
            s = input("🌐 Enter source / website (optional, press Enter to skip): ").strip()
            res = pipeline.predict(t, s)
            print("-" * 55)
            print(f"Verdict     : {'🔴 FAKE' if res['verdict'] == 'FAKE' else '🟢 REAL'}")
            print(f"Confidence  : {res['confidence']}")
            print(f"Fake Prob   : {res['fake_probability']}")
            print(f"Language    : {res['language']}")
            print(f"Sentiment   : {res['sentiment']}")
            print(f"Source Bias : {res['source_credibility']}")
            print("-" * 55)
    else:
        res = pipeline.predict(args.text, args.source)
        print(res)

if __name__ == "__main__":
    main()
