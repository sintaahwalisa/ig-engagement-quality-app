import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# App Config
# ===============================
st.set_page_config(
    page_title="Engagement Quality Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("## Engagement Quality Assessment")


# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ===============================
# Model Feature Schema (LOCKED)
# ===============================
MODEL_FEATURES = [
    "likes_per_10k_reach",
    "comments_per_10k_reach",
    "shares_per_10k_reach",
    "saves_per_10k_reach",
    "active_passive_ratio",
    "log_reach_win",
    "caption_bucket_medium",
    "hashtag_bucket_optimal"
]

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.markdown("### Post Inputs")

st.sidebar.markdown("**Engagement Signals**")

likes = st.sidebar.number_input("Likes per 10K Reach", 0.0, 500.0, 50.0)
comments = st.sidebar.number_input("Comments per 10K Reach", 0.0, 100.0, 10.0)
shares = st.sidebar.number_input("Shares per 10K Reach", 0.0, 100.0, 5.0)
saves = st.sidebar.number_input("Saves per 10K Reach", 0.0, 200.0, 20.0)

active_passive_ratio = st.sidebar.number_input(
    "Active / Passive Engagement Ratio", 0.0, 5.0, 0.5
)

log_reach = st.sidebar.number_input(
    "Log Reach", 8.0, 15.0, 12.0
)

st.sidebar.markdown("**Content Structure**")
# ===============================
# User-Friendly Categorical Inputs
# ===============================
caption_length = st.sidebar.selectbox(
    "Caption Length",
    ["Short", "Medium", "Long"]
)

hashtag_usage = st.sidebar.selectbox(
    "Hashtag Usage",
    ["Suboptimal", "Optimal"]
)

# ===============================
# Mapping UI → Model Features (CRITICAL FIX)
# ===============================
# Model only understands:
# - caption_bucket_medium (1 = Medium, 0 = Short/Long)
# - hashtag_bucket_optimal (1 = Optimal, 0 = Suboptimal)

caption_bucket_medium = 1 if caption_length == "Medium" else 0
hashtag_bucket_optimal = 1 if hashtag_usage == "Optimal" else 0

# ===============================
# Prepare Input (NUMERIC ONLY, SCHEMA SAFE)
# ===============================
input_data = pd.DataFrame([{
    "likes_per_10k_reach": float(likes),
    "comments_per_10k_reach": float(comments),
    "shares_per_10k_reach": float(shares),
    "saves_per_10k_reach": float(saves),
    "active_passive_ratio": float(active_passive_ratio),
    "log_reach_win": float(log_reach),
    "caption_bucket_medium": float(caption_bucket_medium),
    "hashtag_bucket_optimal": float(hashtag_bucket_optimal)
}], columns=MODEL_FEATURES)

# ===============================
# Prediction
# ===============================
prob = model.predict_proba(input_data)[0][1]
prediction = "HIGH" if prob >= 0.5 else "LOW"
st.write("Raw probability:", prob)


col1, col2 = st.columns([2, 1])

with col1:
    st.metric(
        label="Engagement Quality",
        value=prediction
    )

with col2:
    st.metric(
        label="Confidence",
        value=f"{prob:.1%}"
    )

st.divider()
# ===============================
# SHAP (LOCAL ONLY — SAFE MODE)
# ===============================
st.markdown("### Key Drivers")
st.caption(
    "The chart below explains which factors most influenced this assessment."
)

# Build SHAP explainer (NO background data → no dtype issues)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(input_data)

# Handle binary classifier output safely
if isinstance(shap_values, list):
    shap_class1 = shap_values[1]
else:
    shap_class1 = shap_values[:, :, 1]

fig, ax = plt.subplots()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_class1[0],
        base_values=explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        data=input_data.iloc[0],
        feature_names=MODEL_FEATURES
    ),
    show=False
)

st.pyplot(fig)


st.divider()
# ===============================
# Footer
# ===============================
st.caption(
    "Methodology: Behavioral feature engineering • Interpretable machine learning"
)
