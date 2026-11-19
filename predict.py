"""
Full Fake News Detection Pipeline — Inference
Loads all trained models and produces a final prediction for any input text.

Usage:
    python predict.py "Breaking: Government announces..."
    python predict.py --interactive
"""

import os, sys, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Suppress HuggingFace load warnings for clean terminal output
import transformers
transformers.logging.set_verbosity_error()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROBERTA_DIR  = "models/roberta_fakenews"
SENTI_DIR    = "models/sentiment_roberta"
ENSEMBLE_DIR = "models/ensemble"

ROBERTA_MAX_LEN = 256
SENTI_MAX_LEN   = 128
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── RoBERTa Model (must match 02_train_roberta.py) ──────────────────────────
class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta    = AutoModel.from_pretrained("roberta-base")
        hidden          = self.roberta.config.hidden_size
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

# ─── Pipeline ─────────────────────────────────────────────────────────────────
class FakeNewsPipeline:
    def __init__(self):
        print("🔄 Loading models...")

        # RoBERTa
        self.roberta_tok = AutoTokenizer.from_pretrained(ROBERTA_DIR)
        self.roberta     = FakeNewsClassifier().to(DEVICE)
        self.roberta.load_state_dict(
            torch.load(os.path.join(ROBERTA_DIR, "best_model.pt"), map_location=DEVICE, weights_only=True)
        )
        self.roberta.eval()

        # Sentiment
        self.senti_tok  = AutoTokenizer.from_pretrained(SENTI_DIR)
        self.senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_DIR, use_safetensors=True).to(DEVICE)
        self.senti_model.eval()

        # Ensemble
        with open(os.path.join(ENSEMBLE_DIR, "ensemble.pkl"), "rb") as f:
            self.ensemble = pickle.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "scaler.pkl"), "rb") as f:
            self.scaler   = pickle.load(f)

        print("✅ All models loaded.\n")

    @torch.no_grad()
    def _encode(self, text: str, tokenizer, max_len: int) -> dict:
        enc = tokenizer(text, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        return {k: v.to(DEVICE) for k, v in enc.items()}

    @torch.no_grad()
    def _roberta_probs(self, text: str) -> np.ndarray:
        enc    = self._encode(text, self.roberta_tok, ROBERTA_MAX_LEN)
        logits = self.roberta(enc["input_ids"], enc["attention_mask"])
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs   # [real_prob, fake_prob]

    @torch.no_grad()
    def _senti_probs(self, text: str) -> np.ndarray:
        enc   = self._encode(text, self.senti_tok, SENTI_MAX_LEN)
        out   = self.senti_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        return probs   # [neg_prob, pos_prob]

    def predict(self, text: str) -> dict:
        """
        Args:
            text: Input news text or tweet.
        Returns:
            dict with verdict, confidence, fake_prob, sentiment, and raw scores.
        """
        rob_probs   = self._roberta_probs(text)      # [real, fake]
        senti_probs = self._senti_probs(text)         # [neg, neutral, pos]
        cred        = np.array([[0.5]])                # neutral credibility (internal)

        features = np.hstack([rob_probs.reshape(1, -1),
                               senti_probs.reshape(1, -1),
                               cred])                 # shape (1, 6)

        features_scaled = self.scaler.transform(features)
        final_probs     = self.ensemble.predict_proba(features_scaled)[0]   # [real_prob, fake_prob]

        verdict    = "FAKE" if final_probs[1] > 0.5 else "REAL"
        confidence = max(final_probs) * 100
        
        if senti_probs[2] > senti_probs[0] and senti_probs[2] > senti_probs[1]:
            sentiment = "Positive 😊"
        elif senti_probs[0] > senti_probs[1] and senti_probs[0] > senti_probs[2]:
            sentiment = "Negative 😟"
        else:
            sentiment = "Neutral 😐"

        return {
            "verdict":            verdict,
            "confidence":         f"{confidence:.1f}%",
            "fake_probability":   f"{final_probs[1]*100:.1f}%",
            "real_probability":   f"{final_probs[0]*100:.1f}%",
            "sentiment":          sentiment,
            "sentiment_neg_prob": f"{senti_probs[0]*100:.1f}%",
            "sentiment_neu_prob": f"{senti_probs[1]*100:.1f}%",
            "sentiment_pos_prob": f"{senti_probs[2]*100:.1f}%",
            "roberta_real_prob":  f"{rob_probs[0]*100:.1f}%",
            "roberta_fake_prob":  f"{rob_probs[1]*100:.1f}%",
        }

    def display(self, result: dict, text: str):
        print("\n" + "═" * 60)
        print(f"📰 Input: {text[:120]}{'...' if len(text) > 120 else ''}")
        print("─" * 60)
        verdict_icon = "🔴 FAKE NEWS" if result["verdict"] == "FAKE" else "🟢 REAL NEWS"
        print(f"  Verdict       : {verdict_icon}")
        print(f"  Confidence    : {result['confidence']}")
        print(f"  Fake Prob     : {result['fake_probability']}")
        print(f"  Real Prob     : {result['real_probability']}")
        print(f"  Sentiment     : {result['sentiment']} ({result['sentiment_pos_prob']} positive)")
        print(f"  RoBERTa Fake  : {result['roberta_fake_prob']}")
        print("═" * 60 + "\n")

# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("text", nargs="?", default=None, help="Input text to classify")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode (keep typing inputs)")
    args = parser.parse_args()

    pipeline = FakeNewsPipeline()

    if args.interactive:
        print("🤖 Fake News Detector — Interactive Mode")
        print("   [RoBERTa Logical Engine Accuracy: 92.20%]")
        print("   [Sentiment Emotional Engine:      Pre-trained (58M tweets)]")
        print("   [Ensemble Combiner Accuracy:      93.97%]\n")
        print("   Type your news text and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("📝 Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            if text:
                result = pipeline.predict(text)
                pipeline.display(result, text)

    elif args.text:
        result = pipeline.predict(args.text)
        pipeline.display(result, args.text)

    else:
        # Interactive mode — prompt the user for input
        print("🤖 Fake News Detector — Interactive Mode")
        print("   [RoBERTa Logical Engine Accuracy: 92.20%]")
        print("   [Sentiment Emotional Engine:      Pre-trained (58M tweets)]")
        print("   [Ensemble Combiner Accuracy:      93.97%]\n")
        print("   Type your news text and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("📝 Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            if text:
                result = pipeline.predict(text)
                pipeline.display(result, text)

if __name__ == "__main__":
    main()