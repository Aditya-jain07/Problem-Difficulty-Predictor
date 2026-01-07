# app.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


import streamlit as st

from predict import predict_difficulty

from config import MODELS_DIR
from preprocess import KEYWORDS

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Problem Difficulty Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Problem Difficulty Predictor")
st.write(
    "Enter a programming problem description below to predict its "
    "difficulty level and score."
)

st.markdown("---")

# ===============================
# Input Fields
# ===============================
description = st.text_area(
    "📝 Problem Description",
    height=200,
    placeholder="Describe the problem statement here..."
)

input_desc = st.text_area(
    "📥 Input Description",
    height=150,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "📤 Output Description",
    height=150,
    placeholder="Describe the output format..."
)

st.markdown("---")

# ===============================
# Prediction Button
# ===============================
if st.button("🔍 Predict Difficulty"):
    if not description.strip() or not input_desc.strip() or not output_desc.strip():
        st.error("Please fill in all fields before predicting.")
    else:
        with st.spinner("Predicting difficulty..."):
            label, score = predict_difficulty(
                description,
                input_desc,
                output_desc
            )

        st.success("Prediction completed!")

        st.markdown("### 🧠 Prediction Result")
        st.write(f"**Predicted Difficulty:** `{label.capitalize()}`")
        st.write(f"**Predicted Difficulty Score:** `{score}`")

        st.caption(
            "ℹ️ The difficulty score is a fine-grained estimate. "
            "For borderline problems, the predicted class and score may not align exactly."
        )


st.markdown("---")
st.caption("Built using TF-IDF, Logistic Regression & Linear SVR")
