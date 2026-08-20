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

mfg_year = st.sidebar.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2018, step=1)
engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1197, step=50)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
ownership = st.sidebar.selectbox("Ownership Count", [1, 2, 3, 4, 5])

# Categorical Inputs
fuel_type_input = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission_input = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
spare_key_input = st.sidebar.selectbox("Spare Key Available", ["Yes", "No"])

imperfections = st.sidebar.slider("Number of Imperfections", 0, 20, 2)
repainted_parts = st.sidebar.slider("Number of Repainted Parts", 0, 15, 1)

# Prediction Logic
if st.button("Predict Price"):
    # Preprocessing to match the .map() transformations in the notebook
    spare_key = 1 if spare_key_input == "Yes" else 0
    transmission = 1 if transmission_input == "Manual" else 0
    
    fuel_map = {"Petrol": 1, "Diesel": 0, "CNG": 2}
    fuel_type = fuel_map[fuel_type_input]

    # DataFrame with feature order matching X_train
    input_df = pd.DataFrame([{
        'Manufacturing_year': mfg_year,
        'Engine capacity': engine_capacity,
        'Spare key': spare_key,
        'Transmission': transmission,
        'KM driven': km_driven,
        'Ownership': ownership,
        'Fuel type': fuel_type,
        'Imperfections': imperfections,
        'Repainted Parts': repainted_parts
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: **₹{prediction:,.2f}**")
    
    st.write("### Input Feature Summary")
    st.dataframe(input_df)
