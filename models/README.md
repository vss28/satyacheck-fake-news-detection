# Model files

Trained model checkpoints are intentionally not stored in this GitHub repository. The local project expects project-specific artifacts in these paths:

- `indian/xlm_roberta/`
- `indian/ensemble/`
- `roberta_fakenews/`
- `sentiment_roberta/` (the code can fall back to a public Hugging Face sentiment model)
- `ensemble/`

Use the training steps in [`../README_DATASETS.md`](../README_DATASETS.md) and [`../PROJECT_DOCUMENTATION.md`](../PROJECT_DOCUMENTATION.md) to create the relevant checkpoints. The dataset guide primarily documents the Indian training flow; consult the project guide for the US pipeline.

If you distribute trained checkpoints, check the licenses of the base models and training data first. These files can be large; use Git LFS or a model hosting service rather than ordinary Git for large artifacts.
