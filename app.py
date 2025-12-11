import streamlit as st
import numpy as np
import joblib
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔮 Glass Type Prediction ")
st.write("Enter chemical composition to predict glass type.")
RI = st.number_input("Refractive Index (RI)", 1.4, 2.0)
Na = st.number_input("Sodium (Na)", 0.0, 20.0)
Mg = st.number_input("Magnesium (Mg)", 0.0, 10.0)
Al = st.number_input("Aluminum (Al)", 0.0, 5.0)
Si = st.number_input("Silicon (Si)", 60.0, 80.0)
K = st.number_input("Potassium (K)", 0.0, 10.0)
Ca = st.number_input("Calcium (Ca)", 0.0, 15.0)
Ba = st.number_input("Barium (Ba)", 0.0, 5.0)
Fe = st.number_input("Iron (Fe)", 0.0, 5.0)

if st.button("Predict Glass Type"):
    features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

   # Optional: Glass type description
    glass_types = {
        1: "Building Windows (Float Processed)",
        2: "Building Windows (Non-float Processed)",
        3: "Vehicle Windows",
        5: "Containers",
        6: "Tableware",
        7: "Headlamps"
    }
    description = glass_types.get(prediction, "Unknown Type")

    st.success(f"Predicted Glass Type: **{prediction}** - {description}")