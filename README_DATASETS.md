# 🇮🇳 Indian Fake News Datasets — Download & Training Guide

This guide explains how to get the 2 Indian datasets and train your dedicated **XLM-RoBERTa Indian Fake News Model**.

---

## 📁 1. Directory Setup

The folders have already been created for you:
```
data/indian/
  ├── raw/
  │    ├── ifnd/          ← Put IFND English dataset CSV here
  │    └── hinfakenews/   ← Put Hindi dataset CSV here
  └── merged/             ← Will be auto-populated by 01_prepare_data.py
```

---

## 📥 2. Download Datasets (Free Kaggle Account)

### Dataset A: IFND (Indian Fake News Dataset) — English
- **Source:** Kaggle (Sharma & Garg, 2021)
- **Link:** [Kaggle: IFND Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/fake-and-real-news-dataset) or search `IFND: A Benchmark Dataset for Fake News Detection` on Kaggle.
- **Action:** Download the `.csv` file (e.g. `IFND.csv` or `Dataset.csv`) and place it in:
  `data/indian/raw/ifnd/`

### Dataset B: Hindi Fake News Dataset — Hindi
- **Source:** Kaggle
- **Link:** Search `Hindi Fake News Detection Dataset` or `Hindi News Fake and Real` on Kaggle.
- **Action:** Download the `.csv` file and place it in:
  `data/indian/raw/hinfakenews/`

> **Note:** If you run the preparation script without adding files, it automatically generates a starter synthetic dataset covering Indian politics, WhatsApp rumors, and ISRO/RBI headlines so you can verify the entire pipeline immediately!

---

## 🏋️ 3. Execution Pipeline (Run in `gpu_env`)

Open your PowerShell terminal with `gpu_env` activated:

### Step 1: Clean & Merge the Datasets
```powershell
python india_scripts/01_prepare_data.py
```
*Outputs: `data/indian/merged/train.csv`, `val.csv`, `test.csv`*

### Step 2: Fine-tune XLM-RoBERTa on GPU
```powershell
python india_scripts/02_train_xlm_roberta.py
```
*Uses RTX 4060 GPU with mixed precision and layer freezing. Saves to `models/indian/xlm_roberta/best_model.pt`.*

### Step 3: Train Indian Logistic Combiner Ensemble
```powershell
python india_scripts/03_train_ensemble.py
```
*Saves combiner to `models/indian/ensemble/ensemble.pkl`.*

---

## 🚀 4. Launching the Modern Web Application

Run the FastAPI server:
```powershell
python api.py
```
Then open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

### Features in the New App:
- **XLM-RoBERTa multilingual inference** for Hindi, Hinglish, and Indian English
- **Real-Time Source Credibility bias slider** (The Hindu, NDTV, Postcard News, WhatsApp forwards)
- **Interactive SVG Fake News Probability Gauge**
- **Live Indian News Radar Feed** fetching and screening live headlines in real time
- **Preserved Existing System:** Your original US-model Streamlit app (`streamlit run app.py`) remains 100% intact and unaffected.
