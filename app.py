import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the car details below to predict its estimated price.")

# Load Trained Model
@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Form Inputs in Sidebar
st.sidebar.header("Car Details")

# Added Brand Input (matching brands in your dataset)
brand = st.sidebar.selectbox("Brand", ["Maruti", "Hyundai", "Tata", "Honda", "Mahindra", "Other"])

mfg_year = st.sidebar.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2018, step=1)
engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1197, step=50)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
ownership = st.sidebar.number_input("Ownership Count", min_value=1, max_value=5, value=1, step=1)

# Categorical Inputs (keep as original strings for OneHotEncoder)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
spare_key = st.sidebar.selectbox("Spare Key Available", ["Yes", "No"])

imperfections = st.sidebar.slider("Number of Imperfections", 0, 20, 2)
repainted_parts = st.sidebar.slider("Number of Repainted Parts", 0, 15, 1)

# Prediction Logic
if st.sidebar.button("Predict Price"):
    # DataFrame columns must include 'Brand' and match training features
    input_df = pd.DataFrame([{
        'Brand': brand,
        'Manufacturing_year': mfg_year,
        'Engine capacity': engine_capacity,
        'KM driven': km_driven,
        'Ownership': ownership,
        'Spare key': spare_key,
        'Transmission': transmission,
        'Fuel type': fuel_type,
        'Imperfections': imperfections,
        'Repainted Parts': repainted_parts
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: **₹{prediction:,.2f}**")
    
    st.write("### Input Feature Summary")
    st.dataframe(input_df)
