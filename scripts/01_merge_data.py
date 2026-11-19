"""
Step 1 — Data Merging
Merges LIAR (train/val/test) + ISOT into a unified binary fake/real dataset.
Also computes a speaker credibility score from LIAR metadata.
Outputs: data/merged/train.csv, val.csv, test.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
LIAR_DIR  = "data/liar-dataset"
ISOT_FILE = "data/iso-dataset/isot_clean.csv"
OUT_DIR   = "data/merged"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── LIAR Label Mapping (6-class → binary) ────────────────────────────────────
FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}

def map_liar_label(label: str) -> int | None:
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1   # FAKE
    if label in REAL_LABELS:
        return 0   # REAL
    return None    # drop unknown

# ─── Speaker Credibility Score ─────────────────────────────────────────────────
def compute_credibility(row):
    """
    lie_score = (barely_true + false + pants_fire) / (total + 1)
    Returns a float in [0, 1] where 1 = most likely to lie.
    """
    try:
        lie   = row["Barely_True_Count"] + row["False_Count"] + row["Pants_On_Fire_Count"]
        total = (row["Barely_True_Count"] + row["False_Count"] + row["Pants_On_Fire_Count"]
                 + row["Half_True_Count"] + row["Mostly_True_Count"])
        return float(lie) / (float(total) + 1.0)
    except Exception:
        return 0.5   # neutral if data missing

# ─── Load LIAR splits ─────────────────────────────────────────────────────────
def load_liar(split_file: str) -> pd.DataFrame:
    df = pd.read_csv(split_file)
    df["label"] = df["Label"].apply(map_liar_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Fill missing count columns with 0
    count_cols = ["Barely_True_Count", "False_Count", "Pants_On_Fire_Count",
                  "Half_True_Count", "Mostly_True_Count"]
    for c in count_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    df["speaker_cred_score"] = df.apply(compute_credibility, axis=1)
    df["text"]               = df["Statement"].astype(str).str.strip()
    df["source"]             = "liar"

    print(f"  LIAR {os.path.basename(split_file)}: {len(df)} rows | "
          f"FAKE={int((df.label==1).sum())} REAL={int((df.label==0).sum())}")
    return df[["text", "label", "source", "speaker_cred_score"]]

# ─── Load ISOT ────────────────────────────────────────────────────────────────
def load_isot() -> pd.DataFrame:
    df = pd.read_csv(ISOT_FILE)
    # isot_clean.csv: label 0 = Real, 1 = Fake → keep as-is (0=Real, 1=Fake)
    df = df.rename(columns={"final_text": "text"})
    df["text"]               = df["text"].astype(str).str.strip()
    df["source"]             = "isot"
    df["speaker_cred_score"] = 0.5   # no speaker metadata in ISOT
    df["label"]              = df["label"].astype(int)
    df = df.dropna(subset=["text", "label"])
    print(f"  ISOT: {len(df)} rows | "
          f"FAKE={int((df.label==1).sum())} REAL={int((df.label==0).sum())}")
    return df[["text", "label", "source", "speaker_cred_score"]]

# ─── Merge ────────────────────────────────────────────────────────────────────
print("Loading LIAR splits...")
liar_train = load_liar(os.path.join(LIAR_DIR, "clean_train.csv"))
liar_val   = load_liar(os.path.join(LIAR_DIR, "clean_valid.csv"))
liar_test  = load_liar(os.path.join(LIAR_DIR, "clean_test.csv"))

print("\nLoading ISOT...")
isot_all = load_isot()

# Split ISOT 80/10/10 stratified
isot_train_val, isot_test = train_test_split(
    isot_all, test_size=0.10, stratify=isot_all["label"], random_state=42)
isot_train, isot_val = train_test_split(
    isot_train_val, test_size=0.111, stratify=isot_train_val["label"], random_state=42)

print(f"\n  ISOT train: {len(isot_train)} | val: {len(isot_val)} | test: {len(isot_test)}")

# Concatenate
train_df = pd.concat([liar_train, isot_train], ignore_index=True).sample(frac=1, random_state=42)
val_df   = pd.concat([liar_val,   isot_val  ], ignore_index=True).sample(frac=1, random_state=42)
test_df  = pd.concat([liar_test,  isot_test ], ignore_index=True).sample(frac=1, random_state=42)

# ─── Save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
test_df.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)

print("\n[DONE] Merged datasets saved to data/merged/")
print(f"   Train : {len(train_df)} rows | FAKE={int((train_df.label==1).sum())} REAL={int((train_df.label==0).sum())}")
print(f"   Val   : {len(val_df)}   rows | FAKE={int((val_df.label==1).sum())}   REAL={int((val_df.label==0).sum())}")
print(f"   Test  : {len(test_df)}  rows | FAKE={int((test_df.label==1).sum())}  REAL={int((test_df.label==0).sum())}")
