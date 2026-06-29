import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = tf.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the bank.")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

products = st.selectbox(
    "Number of Products",
    [1,2,3,4]
)

credit_card = st.selectbox(
    "Has Credit Card",
    ["Yes","No"]
)

active = st.selectbox(
    "Is Active Member",
    ["Yes","No"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# -----------------------------
# Encoding
# -----------------------------

Germany = 1 if country == "Germany" else 0
Spain = 1 if country == "Spain" else 0

Male = 1 if gender == "Male" else 0

HasCrCard = 1 if credit_card == "Yes" else 0

IsActiveMember = 1 if active == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    features = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        products,
        HasCrCard,
        IsActiveMember,
        salary,
        Germany,
        Spain,
        Male
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    probability = prediction[0][0]

    st.markdown("---")

    if probability > 0.5:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

