# Streamlit + FastAPI Lab (Custom Dataset + Custom Code)

This project is a **complete replica** of the Streamlit lab idea, but with a **different dataset** and **different code**.

**What it does**
- A trained ML model is hosted by **FastAPI**
- A **Streamlit** dashboard calls the FastAPI `/predict` endpoint and shows results

✅ Included:
- Dataset: `data/wine_sample.csv`
- Trained model: `backend/wine_model.pkl`
- Backend: `backend/main.py`
- Dashboard: `frontend/Dashboard.py`
- Example request JSON: `data/sample_request.json`

---

## 1) Create & activate a virtual environment

```bash
python3 -m venv streamlitenv
source ./streamlitenv/bin/activate
```

(Windows)
```bat
.streamlitenv\Scripts\activate
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3) Run FastAPI backend (Model server)

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Check:
- http://localhost:8000/
- http://localhost:8000/health

---

## 4) Run Streamlit UI

In another terminal (same venv), from the project root:

```bash
streamlit run frontend/Dashboard.py
```

Open:
- http://localhost:8501

---

## 5) Predict using JSON upload

Upload `data/sample_request.json` in the Streamlit sidebar, or use sliders.

Example JSON shape (you can send either `{"input": {...}}` or `{...}`):

```json
{
  "input": {
    "alcohol": 13.2,
    "malic_acid": 2.7,
    "ash": 2.3,
    "alcalinity_of_ash": 18.0,
    "magnesium": 100.0,
    "total_phenols": 2.2,
    "flavanoids": 2.0,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.6,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280/od315_of_diluted_wines": 2.8,
    "proline": 750.0
  }
}
```

---

## Project structure

```
.
├── backend
│   ├── main.py
│   └── wine_model.pkl
├── frontend
│   └── Dashboard.py
├── data
│   ├── wine_sample.csv
│   └── sample_request.json
├── requirements.txt
└── README.md
```
