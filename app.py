import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ---------------- SAFE BASE PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD SCALER (REQUIRED) ----------------
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ---------------- LOAD MODELS SAFELY ----------------
linear_model = pickle.load(open(os.path.join(BASE_DIR, "linear_model.pkl"), "rb"))

ridge_model = None
lasso_model = None

if os.path.exists(os.path.join(BASE_DIR, "ridge_model.pkl")):
    ridge_model = pickle.load(open(os.path.join(BASE_DIR, "ridge_model.pkl"), "rb"))

if os.path.exists(os.path.join(BASE_DIR, "lasso_model.pkl")):
    lasso_model = pickle.load(open(os.path.join(BASE_DIR, "lasso_model.pkl"), "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price Prediction", layout="centered")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1568605114967-8130f3a36994");
    background-size: cover;
}
.block-container {
    background-color: rgba(0,0,0,0.65);
    padding: 30px;
    border-radius: 15px;
}
h1, h2, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADING ----------------
st.markdown("<h1 style='text-align:center;'>🏡 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- USER INPUTS ----------------
CRIM = st.slider("Crime Rate", 0.0, 100.0, 5.0)
ZN = st.slider("Residential Land Zone (%)", 0.0, 100.0, 20.0)
INDUS = st.slider("Industrial Area (%)", 0.0, 30.0, 5.0)
CHAS = st.selectbox("Near Charles River", [0, 1])
NOX = st.slider("Nitric Oxide Concentration", 0.3, 1.0, 0.5)
RM = st.slider("Average Rooms", 3.0, 10.0, 6.0)
AGE = st.slider("Age of House (%)", 0.0, 100.0, 50.0)
DIS = st.slider("Distance to Employment Centers", 1.0, 12.0, 4.0)
RAD = st.slider("Accessibility Index", 1, 24, 4)
TAX = st.slider("Property Tax Rate", 150, 800, 300)
PTRATIO = st.slider("Pupil–Teacher Ratio", 12.0, 25.0, 18.0)
B = st.slider("Black Population Index", 0.0, 400.0, 350.0)
LSTAT = st.slider("Lower Status Population (%)", 1.0, 40.0, 10.0)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Price"):

    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                             DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    input_scaled = scaler.transform(input_data)

    results = []

    # Linear Regression (always available)
    linear_pred = linear_model.predict(input_scaled)[0]
    results.append(["Linear Regression", round(linear_pred, 2), f"${linear_pred*1000:,.0f}"])

    # Ridge Regression (if exists)
    if ridge_model:
        ridge_pred = ridge_model.predict(input_scaled)[0]
        results.append(["Ridge Regression", round(ridge_pred, 2), f"${ridge_pred*1000:,.0f}"])
    else:
        results.append(["Ridge Regression", "Not Available", "—"])

    # Lasso Regression (if exists)
    if lasso_model:
        lasso_pred = lasso_model.predict(input_scaled)[0]
        results.append(["Lasso Regression", round(lasso_pred, 2), f"${lasso_pred*1000:,.0f}"])
    else:
        results.append(["Lasso Regression", "Not Available", "—"])

    df = pd.DataFrame(
        results,
        columns=["Model", "Predicted Value ($1000s)", "Predicted Price ($)"]
    )

    st.success("🏠 Predicted House Prices")
    st.table(df)
