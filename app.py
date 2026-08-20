import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")
st.write("Fill in the details below to estimate the price of a used car using the Random Forest model.")

# --- Load Model ---
@st.cache_resource
def load_artifacts():
    # Load your trained Random Forest Model
    model = joblib.load("car_price_model.pkl")
    return model

try:
    model = load_artifacts()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- User Inputs ---
st.header("Car Details")

col1, col2 = st.columns(2)

with col1:
    mfg_year = st.number_input(
        "Manufacturing Year", 
        min_value=1990, 
        max_value=2026, 
        value=2018, 
        step=1
    )
    
    engine_capacity = st.number_input(
        "Engine Capacity (cc)", 
        min_value=500, 
        max_value=6000, 
        value=1200, 
        step=50
    )
    
    km_driven = st.number_input(
        "KM Driven", 
        min_value=0, 
        max_value=500000, 
        value=30000, 
        step=1000
    )
    
    imperfections = st.number_input(
        "Imperfections Count", 
        min_value=0, 
        max_value=20, 
        value=2, 
        step=1
    )

    repainted_parts = st.number_input(
        "Repainted Parts", 
        min_value=0, 
        max_value=20, 
        value=0, 
        step=1
    )

with col2:
    # Spare key: Yes -> 1, No -> 0
    spare_key_input = st.selectbox("Spare Key Available?", options=["Yes", "No"])
    spare_key = 1 if spare_key_input == "Yes" else 0

    # Transmission: Manual -> 1, Automatic -> 0
    transmission_input = st.selectbox("Transmission", options=["Manual", "Automatic"])
    transmission = 1 if transmission_input == "Manual" else 0

    ownership = st.selectbox("Ownership (Owner Count)", options=[1, 2, 3, 4, 5])

    # Fuel type: Petrol -> 1, Diesel -> 0, CNG -> 2
    fuel_input = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
    fuel_mapping = {'Petrol': 1, 'Diesel': 0, 'CNG': 2}
    fuel_type = fuel_mapping[fuel_input]

# --- Predict Button & Logic ---
st.markdown("---")

if st.button("Predict Price", type="primary"):
    # Construct DataFrame with exact column names used during training
    input_data = pd.DataFrame([{
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

    try:
        # Pass input_data as a DataFrame (do not use .values)
        prediction = model.predict(input_data)[0]
        
        st.success(f"### Estimated Price: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
