import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Main title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #cfd8dc;
        font-size: 1.05rem;
        margin-top: 0px;
        margin-bottom: 1.5rem;
    }

    /* Card style container */
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 14px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141e30, #243b55);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.4em;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 18px rgba(0,198,255,0.6);
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #e3f2fd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

FEATURES = list(getattr(model, "feature_names_in_", [
    "hours_studied", "sleep_hours", "attendance_percent", "previous_scores"
]))

# Reasonable defaults / ranges derived from the training data the model was fit on
FEATURE_CONFIG = {
    "hours_studied": {"label": "📚 Hours Studied / Day", "min": 0.0, "max": 15.0, "default": 6.0, "step": 0.5},
    "sleep_hours": {"label": "😴 Sleep Hours / Day", "min": 0.0, "max": 12.0, "default": 7.0, "step": 0.5},
    "attendance_percent": {"label": "🏫 Attendance (%)", "min": 0.0, "max": 100.0, "default": 80.0, "step": 1.0},
    "previous_scores": {"label": "📝 Previous Exam Score", "min": 0.0, "max": 100.0, "default": 65.0, "step": 1.0},
}

TRAIN_TARGET_MIN, TRAIN_TARGET_MAX = 18.3, 51.3  # observed range in training labels

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Score Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by a K-Nearest Neighbors model — adjust the inputs and predict instantly</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# SIDEBAR — INPUTS
# --------------------------------------------------------------------------------
st.sidebar.header("🧮 Input Parameters")
st.sidebar.caption("Tune the sliders to describe the student profile")

input_values = {}
for feat in FEATURES:
    cfg = FEATURE_CONFIG.get(feat, {"label": feat, "min": 0.0, "max": 100.0, "default": 50.0, "step": 1.0})
    input_values[feat] = st.sidebar.slider(
        cfg["label"],
        min_value=float(cfg["min"]),
        max_value=float(cfg["max"]),
        value=float(cfg["default"]),
        step=float(cfg["step"]),
    )

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🚀 Predict Score")
st.sidebar.markdown("---")
st.sidebar.info(
    f"Model: **K-Nearest Neighbors Regressor**\n\n"
    f"Neighbors used (k): **{getattr(model, 'n_neighbors', 5)}**\n\n"
    f"Trained on **{getattr(model, 'n_samples_fit_', 'N/A')}** samples"
)

# --------------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------------
input_df = pd.DataFrame([input_values], columns=FEATURES)

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown("#### 📥 Current Input Snapshot")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.dataframe(
        input_df.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar chart of the normalized inputs
    categories = [FEATURE_CONFIG.get(f, {}).get("label", f) for f in FEATURES]
    norm_values = []
    for f in FEATURES:
        cfg = FEATURE_CONFIG.get(f, {"min": 0, "max": 100})
        rng = (cfg["max"] - cfg["min"]) or 1
        norm_values.append((input_values[f] - cfg["min"]) / rng * 100)

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=norm_values + [norm_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color="#00c6ff"),
        fillcolor="rgba(0,198,255,0.3)",
        name="Profile"
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="white"),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=30, r=30, t=30, b=30),
        height=350,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

with col_right:
    st.markdown("#### 🔮 Prediction")

    if predict_clicked or "last_pred" not in st.session_state:
        prediction = float(model.predict(input_df.values)[0])
        st.session_state["last_pred"] = prediction
    else:
        prediction = st.session_state["last_pred"]

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='color:#00e5ff; margin-bottom:0;'>{prediction:.2f}</h2>"
        f"<p style='color:#b0bec5;'>Predicted Score</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={'suffix': "", 'font': {'color': 'white', 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "#00c6ff"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 40], 'color': "#5c1a1a"},
                {'range': [40, 70], 'color': "#7a5c00"},
                {'range': [70, 100], 'color': "#0b5c2f"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': prediction
            }
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    if prediction < 40:
        st.warning("⚠️ Predicted performance is on the lower side — consider more study hours or attendance.")
    elif prediction < 70:
        st.info("ℹ️ Predicted performance is moderate — there's room for improvement.")
    else:
        st.success("✅ Predicted performance looks strong!")

    st.caption(
        f"Note: training labels observed in the range **{TRAIN_TARGET_MIN}–{TRAIN_TARGET_MAX}**, "
        "so predictions will typically fall near that range regardless of the gauge scale shown above."
    )

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#78909c; font-size:0.85rem;'>"
    "Built with ❤️ using Streamlit &nbsp;|&nbsp; Model: KNeighborsRegressor (scikit-learn)"
    "</p>",
    unsafe_allow_html=True,
)
