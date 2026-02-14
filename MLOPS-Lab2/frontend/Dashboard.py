import json
import os

import requests
import streamlit as st

FASTAPI_BACKEND_ENDPOINT = os.getenv("FASTAPI_BACKEND_ENDPOINT", "http://localhost:8000")

FEATURE_SPECS = [
    ("alcohol", 10.0, 15.0, 13.0, 0.1),
    ("malic_acid", 0.5, 6.0, 2.0, 0.1),
    ("ash", 1.3, 3.5, 2.3, 0.1),
    ("alcalinity_of_ash", 10.0, 30.0, 19.0, 0.5),
    ("magnesium", 70.0, 165.0, 100.0, 1.0),
    ("total_phenols", 0.5, 4.5, 2.0, 0.1),
    ("flavanoids", 0.0, 5.5, 2.0, 0.1),
    ("nonflavanoid_phenols", 0.0, 1.0, 0.3, 0.05),
    ("proanthocyanins", 0.4, 4.0, 1.5, 0.1),
    ("color_intensity", 1.0, 13.0, 5.0, 0.1),
    ("hue", 0.4, 1.7, 1.0, 0.05),
    ("od280/od315_of_diluted_wines", 1.2, 4.0, 2.8, 0.05),
    ("proline", 250.0, 1700.0, 750.0, 10.0),
]

def backend_status():
    try:
        r = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
        return False, {"status_code": r.status_code}
    except Exception as e:
        return False, {"error": str(e)}

def build_sidebar():
    st.sidebar.title("Wine Model Demo 🍷")
    ok, info = backend_status()
    if ok:
        st.sidebar.success("Backend online ✅")
    else:
        st.sidebar.error("Backend offline 😱")
        st.sidebar.caption(info)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Inputs")

    mode = st.sidebar.radio("Choose input method", ["Sliders", "Upload JSON"], index=0)

    if mode == "Sliders":
        payload = {}
        for name, mn, mx, default, step in FEATURE_SPECS:
            payload[name] = st.sidebar.slider(name, float(mn), float(mx), float(default), float(step))
        st.session_state["payload"] = payload
        st.session_state["HAS_PAYLOAD"] = True
    else:
        up = st.sidebar.file_uploader("Upload a JSON file", type=["json"])
        if up is not None:
            data = json.load(up)
            st.sidebar.write("Preview")
            st.sidebar.json(data)
            payload = data.get("input", data)
            st.session_state["payload"] = payload
            st.session_state["HAS_PAYLOAD"] = True
        else:
            st.session_state["HAS_PAYLOAD"] = False

    return st.sidebar.button("Predict", type="primary")

def main():
    st.set_page_config(page_title="Wine Prediction Dashboard", page_icon="🍷", layout="wide")

    predict_clicked = build_sidebar()

    st.title("Wine Class Prediction Dashboard 🍇🍷")
    st.write("This UI talks to a FastAPI backend. Use sliders or upload a JSON file.")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Request payload")
        if st.session_state.get("HAS_PAYLOAD"):
            st.json(st.session_state.get("payload"))
        else:
            st.info("Provide inputs in the sidebar to enable prediction.")

    with col2:
        st.subheader("Prediction")
        placeholder = st.empty()

        if predict_clicked:
            if not st.session_state.get("HAS_PAYLOAD"):
                placeholder.error("No input payload provided.")
                return

            payload = st.session_state.get("payload", {})
            with st.spinner("Predicting..."):
                try:
                    r = requests.post(
                        f"{FASTAPI_BACKEND_ENDPOINT}/predict",
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        out = r.json()
                        placeholder.success(f"Predicted: {out['predicted_label']}")
                        st.metric("Predicted class", out["predicted_class"])
                        st.write("Probabilities:")
                        st.bar_chart(out["probabilities"])
                    else:
                        placeholder.error(f"Backend error: {r.status_code}")
                        st.code(r.text)
                except Exception as e:
                    placeholder.error("Problem talking to backend. Check that FastAPI is running.")
                    st.code(str(e))

    st.markdown("---")
    st.caption("Backend command: `uvicorn backend.main:app --reload --port 8000`")

if __name__ == "__main__":
    main()
