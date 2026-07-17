import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------------
# Load model (cached so it's only loaded once per session)
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("📊 Student Score Predictor")
st.write(
    "This app uses a **K-Nearest Neighbors Regressor** to predict a "
    "student's score based on study habits and past performance."
)

st.divider()
st.subheader("Enter student details")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Hours Studied (per day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5
    )
    sleep_hours = st.number_input(
        "Sleep Hours (per day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5
    )

with col2:
    attendance_percent = st.number_input(
        "Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0
    )
    previous_scores = st.number_input(
        "Previous Scores", min_value=0.0, max_value=100.0, value=70.0, step=1.0
    )

st.divider()

if st.button("🔮 Predict Score", use_container_width=True):
    features = np.array(
        [[hours_studied, sleep_hours, attendance_percent, previous_scores]]
    )
    prediction = model.predict(features)[0]

    st.success(f"### Predicted Score: **{prediction:.2f}**")

    with st.expander("See input summary"):
        st.write(
            {
                "Hours Studied": hours_studied,
                "Sleep Hours": sleep_hours,
                "Attendance (%)": attendance_percent,
                "Previous Scores": previous_scores,
            }
        )

st.divider()
st.caption("Model: scikit-learn KNeighborsRegressor")
