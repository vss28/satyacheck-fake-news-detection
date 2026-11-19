"""
Step 1 (Indian Pipeline) — Data Preparation & Merging
======================================================
Prepares and merges Indian fake news datasets:
1. IFND (Indian Fake News Dataset) - English
2. HinFakeNews (or Hindi Fake News Detection Dataset) - Hindi

Inputs expected in:
    data/indian/raw/ifnd/ (e.g. IFND.csv or ifnd_dataset.csv)
    data/indian/raw/hinfakenews/ (e.g. Hindi_Fake_News.csv or similar)

Outputs:
    data/indian/merged/train.csv
    data/indian/merged/val.csv
    data/indian/merged/test.csv

If raw datasets are not yet downloaded, this script can generate sample synthetic
Indian news data so the entire training/inference code can be verified end-to-end.
"""

import os
import glob
import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

RAW_DIR = "data/indian/raw"
IFND_DIR = os.path.join(RAW_DIR, "ifnd")
HINDI_DIR = os.path.join(RAW_DIR, "hinfakenews")
OUT_DIR = "data/indian/merged"

os.makedirs(IFND_DIR, exist_ok=True)
os.makedirs(HINDI_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_lang(text: str) -> str:
    # Quick regex check for Devanagari script (Hindi/Marathi)
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi"
    return "english"

def load_ifnd() -> pd.DataFrame:
    """Finds and standardizes IFND dataset CSV."""
    csv_files = glob.glob(os.path.join(IFND_DIR, "*.csv"))
    if not csv_files:
        print("⚠️  No IFND CSV found in", IFND_DIR)
        return pd.DataFrame(columns=["text", "label", "language", "source"])

    filepath = csv_files[0]
    print(f"📖 Loading IFND from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip")
    
    # Identify text column
    text_col = None
    for c in ["Statement", "statement", "headline", "news_headline", "title", "text", "clean_text"]:
        if c in df.columns:
            text_col = c
            break

    # Identify label column
    label_col = None
    for c in ["Label", "label", "class", "target"]:
        if c in df.columns:
            label_col = c
            break

    if not text_col or not label_col:
        print(f"❌ Could not match columns in IFND. Columns present: {list(df.columns)}")
        return pd.DataFrame(columns=["text", "label", "language", "source"])

    clean_df = pd.DataFrame()
    clean_df["text"] = df[text_col].apply(clean_text)
    
    # In IFND: 0 = Fake, 1 = True/Real or vice versa. Standardize to: 1 = FAKE, 0 = REAL
    # Usually in IFND: 1 = True, 0 = Fake. Let's check common representation
    # Standard throughout our project: 1 is FAKE, 0 is REAL
    def map_label(val):
        s = str(val).strip().lower()
        if s in ["fake", "0", "false"]:
            return 1  # FAKE
        elif s in ["true", "real", "1"]:
            return 0  # REAL
        return None

    clean_df["label"] = df[label_col].apply(map_label)
    clean_df = clean_df.dropna(subset=["text", "label"])
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["language"] = "english"
    clean_df["source"] = "ifnd"
    clean_df = clean_df[clean_df["text"].str.len() > 10]
    
    print(f"   ✅ IFND processed: {len(clean_df)} items (FAKE={sum(clean_df['label']==1)}, REAL={sum(clean_df['label']==0)})")
    return clean_df

def load_hinfakenews() -> pd.DataFrame:
    """Finds and standardizes Hindi Fake News CSV."""
    csv_files = glob.glob(os.path.join(HINDI_DIR, "*.csv"))
    if not csv_files:
        print("⚠️  No Hindi fake news CSV found in", HINDI_DIR)
        return pd.DataFrame(columns=["text", "label", "language", "source"])

    filepath = csv_files[0]
    print(f"📖 Loading Hindi dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip")
    
    text_col = None
    for c in ["text", "Text", "news", "headline", "Statement", "Hindi_Headline"]:
        if c in df.columns:
            text_col = c
            break

    label_col = None
    for c in ["label", "Label", "class", "target"]:
        if c in df.columns:
            label_col = c
            break

    if not text_col or not label_col:
        print(f"❌ Could not match columns in Hindi dataset. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["text", "label", "language", "source"])

    clean_df = pd.DataFrame()
    clean_df["text"] = df[text_col].apply(clean_text)

    def map_label(val):
        s = str(val).strip().lower()
        if s in ["fake", "1", "false", "झूठ"]:
            return 1
        elif s in ["true", "0", "real", "सच"]:
            return 0
        return None

    clean_df["label"] = df[label_col].apply(map_label)
    clean_df = clean_df.dropna(subset=["text", "label"])
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["language"] = "hindi"
    clean_df["source"] = "hinfakenews"
    clean_df = clean_df[clean_df["text"].str.len() > 10]
    
    print(f"   ✅ Hindi processed: {len(clean_df)} items (FAKE={sum(clean_df['label']==1)}, REAL={sum(clean_df['label']==0)})")
    return clean_df

def generate_sample_dataset() -> pd.DataFrame:
    """Fallback sample dataset with Indian political/social context in English and Hindi."""
    print("ℹ️  Generating starter benchmark dataset with Indian news context...")
    samples = [
        # English Real
        ("Reserve Bank of India announces new digital rupee initiative pilot with 8 commercial banks.", 0, "english", "synthetic"),
        ("ISRO successfully launches Chandrayaan meteorological payload from Sriharikota.", 0, "english", "synthetic"),
        ("Indian Railways introduces 10 new Vande Bharat Express routes linking key metropolitan centers.", 0, "english", "synthetic"),
        ("Ministry of External Affairs holds high-level bilateral talks in New Delhi on regional security.", 0, "english", "synthetic"),
        ("Finance Ministry reports GST collections crossed 1.7 lakh crore milestone in recent month.", 0, "english", "synthetic"),
        ("Election Commission of India issues formal notification for upcoming assembly polling schedule.", 0, "english", "synthetic"),
        ("Supreme Court of India delivers constitutional bench verdict on electoral bonds transparency.", 0, "english", "synthetic"),
        ("IIT Bombay and AIIMS collaborate on AI medical diagnostics for rural primary healthcare.", 0, "english", "synthetic"),

        # English Fake
        ("UNESCO declares Indian National Anthem as world best anthem in emergency United Nations summit.", 1, "english", "synthetic"),
        ("RBI to demonetize all 500 rupee currency notes starting midnight tonight confirm leaked circulars.", 1, "english", "synthetic"),
        ("NASA satellite imagery reveals Lord Rama bridge completely renewed underwater overnight.", 1, "english", "synthetic"),
        ("Forward this WhatsApp notice: Drinking boiled ginger water cures 100 percent of viral pneumonia.", 1, "english", "synthetic"),
        ("Government announces free 5000 rupees recharge for every Indian smartphone user click here.", 1, "english", "synthetic"),
        ("Government to monitor all WhatsApp video calls with three green ticks starting tomorrow.", 1, "english", "synthetic"),
        ("Foreign agency report says secret GPS nanochip embedded inside newly released currency bills.", 1, "english", "synthetic"),
        ("Breaking: PM announces cancellation of income tax for all citizens from upcoming Monday.", 1, "english", "synthetic"),

        # Hindi Real
        ("भारतीय रिज़र्व बैंक ने प्रमुख ब्याज दरों में कोई बदलाव नहीं करने की घोषणा की।", 0, "hindi", "synthetic"),
        ("इसरो ने श्रीहरिकोटा से नए उपग्रह का सफल प्रक्षेपण किया, वैज्ञानिकों को बधाई मिली।", 0, "hindi", "synthetic"),
        ("सुप्रीम कोर्ट ने चुनावी सुधारों पर सुनवाई करते हुए केंद्र सरकार से जवाब मांगा।", 0, "hindi", "synthetic"),
        ("रेल मंत्रालय ने वंदे भारत एक्सप्रेस के नए रूटों की आधिकारिक सूची जारी कर दी है।", 0, "hindi", "synthetic"),
        ("मौसम विभाग ने उत्तर भारत के कई राज्यों में भारी बारिश की चेतावनी जारी की।", 0, "hindi", "synthetic"),
        ("प्रधानमंत्री ने नई संसद में राष्ट्रीय खेल पुरस्कार विजेताओं को सम्मानित किया।", 0, "hindi", "synthetic"),

        # Hindi Fake
        ("यूनेस्को ने भारतीय राष्ट्रगान को दुनिया का सबसे सर्वश्रेष्ठ गान घोषित कर दिया है।", 1, "hindi", "synthetic"),
        ("आज रात 12 बजे से 500 के सभी नोट बंद हो जाएंगे, सरकार ने जारी किया आपातकालीन नोटिस।", 1, "hindi", "synthetic"),
        ("नासा की सेटेलाइट ने भारत में सोने का बड़ा भंडार ढूंढ निकाला, सोशल मीडिया पर तस्वीरें वायरल।", 1, "hindi", "synthetic"),
        ("व्हाट्सएप पर यह मैसेज 10 लोगों को भेजें, सरकार देगी 3 महीने का फ्री मोबाइल रिचार्ज।", 1, "hindi", "synthetic"),
        ("गर्म पानी में हल्दी मिलाकर पीने से हर तरह का गंभीर संक्रमण 24 घंटे में खत्म हो जाता है।", 1, "hindi", "synthetic"),
        ("नए नोटों में लगी है जीपीएस चिप, जमीन के 120 फीट नीचे भी सिग्नल पकड़ सकती है सरकार।", 1, "hindi", "synthetic"),
    ] * 20  # Replicate to form sufficient initial seed split

    return pd.DataFrame(samples, columns=["text", "label", "language", "source"])

def main():
    print("=" * 60)
    print("🇮🇳 Indian Fake News Dataset Merger & Preparation")
    print("=" * 60)

    ifnd_df = load_ifnd()
    hindi_df = load_hinfakenews()

    dfs = [df for df in [ifnd_df, hindi_df] if len(df) > 0]

    if not dfs:
        print("\n⚠️  No raw datasets found in data/indian/raw/.")
        print("   Generating synthetic starter dataset so scripts & UI can run immediately.")
        merged_df = generate_sample_dataset()
    else:
        merged_df = pd.concat(dfs, ignore_index=True)

    # Clean text column
    merged_df["text"] = merged_df["text"].apply(clean_text)
    merged_df = merged_df.dropna(subset=["text", "label"])
    merged_df = merged_df.drop_duplicates(subset=["text"])
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n📊 Total Cleaned Records: {len(merged_df)}")
    print(f"   FAKE (1): {int((merged_df.label == 1).sum())}")
    print(f"   REAL (0): {int((merged_df.label == 0).sum())}")

    # Stratified Split: 80% train, 10% val, 10% test
    train_val, test_df = train_test_split(
        merged_df, test_size=0.10, stratify=merged_df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.111, stratify=train_val["label"], random_state=42
    )

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"\n💾 Saved splits to {OUT_DIR}/:")
    print(f"   - train.csv: {len(train_df)} rows")
    print(f"   - val.csv:   {len(val_df)} rows")
    print(f"   - test.csv:  {len(test_df)} rows")
    print("\n✅ Data preparation complete.")

if __name__ == "__main__":
    main()
