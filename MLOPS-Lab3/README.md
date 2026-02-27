# TFDV + TensorFlow (Keras) Lab — Iris Dataset

This lab demonstrates **TensorFlow Data Validation (TFDV)** for data profiling and validation, followed by a **TensorFlow (Keras) neural network** for classification — using the **Iris** dataset.

## What I did in this lab

1. **Used a different dataset:** Iris (150 rows, 4 numeric features, 3 classes).
2. **Generated dataset statistics** using TFDV (feature distributions, missing values, etc.).
3. **Inferred a schema** from the training data automatically.
4. **Validated a new split (test data)** against the schema and checked for anomalies.
5. **Trained a different model:** a small **Keras Dense Neural Network** (multi-class classifier).
6. **Evaluated** the model and saved it.

## Project structure

```
TFDV_Iris_Lab/
  README.md
  requirements.txt
  TFDV_Iris_Lab.ipynb
  data/
    iris.csv
    train.csv
    test.csv
  src/
    data_validation.py
    train_model.py
```

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> Note: `tensorflow-data-validation` can be a large dependency. If installation is slow, try a fresh environment.

## How to run

### Option A — Notebook (recommended)
Open the notebook and run cell-by-cell:
```bash
jupyter notebook
```
Then open `TFDV_Iris_Lab.ipynb`.

### Option B — Run scripts
1) Data profiling + schema + anomaly detection:
```bash
python src/data_validation.py
```

2) Model training:
```bash
python src/train_model.py
```

## Outputs

- Console output for:
  - dataset shapes,
  - schema summary,
  - anomaly results (if any),
  - model accuracy.
- Saved Keras model:
  - `src/saved_model/`

## Notes

- The dataset files are included locally inside `data/` so the lab runs without downloading anything.
- The target label is an integer class:
  - `0 = setosa`, `1 = versicolor`, `2 = virginica`

---

**Author:** Manoj  
**Date:** 2026-02-27
