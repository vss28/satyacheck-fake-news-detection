# Fake News Detection Suite

A student project exploring machine learning approaches to news credibility. It includes an Indian multilingual detector called **SatyaCheck** and a separate US-focused Streamlit demo.

> **Important:** This project is an educational prototype. Its predictions are model outputs, not verified fact checks. Do not use them as the sole basis for decisions or share sensitive personal information with the application.

## Project members

The project has two parts with different contributors:

| Contributor | Contribution |
| --- | --- |
| Vedant Shelke, and Arin Govekar | Developed the Indian news extension: SatyaCheck, its FastAPI backend, and live news monitoring. |
| Shubhan Telnag, Vaishnavi Muley | Contributed to the original US news detector, which has a more limited feature set. |

## What it includes

- **Original US detector (team contribution):** A Streamlit interface for an English-language model with a limited feature set.
- **Indian news extension (individual contribution):** SatyaCheck, developed by Vedant Shelke, with a FastAPI backend, browser interface, and live news monitoring. The project describes support for English, Hindi, and Hinglish.
- Model training scripts, dataset instructions, and detailed project documentation.

The project README and documentation report validation/test metrics for the trained models. Those figures depend on the datasets and evaluation setup; they do not guarantee accuracy on new claims or current events.

## Repository contents

- `api.py` — FastAPI server for SatyaCheck.
- `frontend/` — browser interface served by the API.
- `predict_india.py`, `india_scripts/` — Indian model inference and training code.
- `app.py`, `predict.py`, `scripts/` — US model demo and training code.
- `requirements.txt` — Python dependencies.
- `PROJECT_DOCUMENTATION.md` — detailed architecture and implementation guide.
- `README_DATASETS.md` — dataset setup and training notes.

Datasets, local environments, generated outputs, duplicate copies, and trained model weights are excluded from this GitHub repository. See [`models/README.md`](models/README.md) and [`README_DATASETS.md`](README_DATASETS.md) for setup guidance.

## Requirements

- Python 3.10 or newer
- Internet access on first run to download public pretrained transformer models
- A GPU is recommended for training and faster inference; CPU may be slower
- The trained project-specific checkpoints must be present under `models/` for local inference. The repository does not include those checkpoints.

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Follow `README_DATASETS.md` and the training instructions in `PROJECT_DOCUMENTATION.md` to prepare data and create the project-specific model checkpoints. Do not commit datasets or checkpoints unless you have permission to redistribute them and have arranged suitable storage for large files.

## Run the apps

Start the Indian SatyaCheck API and interface:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Start the US Streamlit demo in a second terminal:

```powershell
streamlit run app.py
```

## Dataset and model notes

The repository intentionally excludes `data/` and trained files under `models/`. Dataset terms and redistribution rights vary; use the dataset instructions to obtain data from its source and review its license before use. Model checkpoints can be large, so use Git LFS or a model hosting service only if you intend to distribute them and have confirmed the relevant licenses.

## Project documentation

- [Detailed project guide](PROJECT_DOCUMENTATION.md)
- [Dataset and training guide](README_DATASETS.md)

## License

No license has been added yet. Unless the project authors add a license, others may view the repository but do not automatically receive permission to reuse, modify, or redistribute its code. Add a license only after the project members agree on one and confirm that included third-party materials allow it.
