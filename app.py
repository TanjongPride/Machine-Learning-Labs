import streamlit as st
import joblib
import numpy as np

# Load the models
lr = joblib.load("models/logistic_regression.pkl")
rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

# Page config
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="centered")

# Title
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will **churn or stay**.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

st.divider()

# Encode inputs exactly like training
gender_enc = 1 if gender == "Male" else 0
contract_enc = {"Month-to-Month": 0, "One-Year": 1, "Two-Year": 2}[contract_type]
internet_enc = 0 if internet_service == "DSL" else 1
tech_enc = 1 if tech_support == "Yes" else 0

# Build input array
input_data = np.array([[age, gender_enc, tenure, monthly_charges,
                        contract_enc, internet_enc, total_charges, tech_enc]])

# Select model
model_map = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb
}
selected_model = model_map[model_choice]

# Predict button
if st.button("🔮 Predict Churn", use_container_width=True):
    prediction = selected_model.predict(input_data)[0]
    probability = selected_model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ This customer is **LIKELY TO CHURN**")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("**💡 Recommendation:** Offer a discount or upgrade to a longer contract.")
    else:
        st.success(f"✅ This customer is **LIKELY TO STAY**")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("**💡 Recommendation:** Customer is stable. Keep engagement high.")