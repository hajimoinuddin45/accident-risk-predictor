import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model_loader import model, scaler

st.set_page_config(page_title="Accident Risk Predictor", layout="centered")

st.title("🚦 Road‑Traffic Accident Risk Predictor")
st.write("Enter road, weather and time details to estimate accident severity risk and see key factors.")

# --- Input widgets ---
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=34.05)
    temp = st.slider("Temperature (°F)", 0.0, 120.0, 70.0)
    visibility = st.slider("Visibility (mi)", 0.0, 20.0, 10.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    hour = st.slider("Hour of Day", 0, 23, 14)
with col2:
    lng = st.number_input("Longitude", value=-118.24)
    wind = st.slider("Wind Speed (mph)", 0.0, 50.0, 5.0)
    pressure = st.slider("Pressure (inHg)", 25.0, 35.0, 30.0)
    weather_code = st.number_input("Weather Condition code", value=10, step=1)
    day = st.slider("Day of Week (0=Mon)", 0, 6, 2)

city_code = st.number_input("City code", value=100, step=1)
state_code = st.number_input("State code", value=5, step=1)

if st.button("Predict"):
    # Arrange features in the SAME order used during training
    feature_names = [
        'Start_Lat','Start_Lng','Temperature(F)','Humidity(%)',
        'Visibility(mi)','Wind_Speed(mph)','Pressure(in)',
        'Weather_Condition','City','State','Hour','Day'
    ]
    values = np.array([[lat, lng, temp, humidity, visibility,
                        wind, pressure, weather_code,
                        city_code, state_code, hour, day]])
    
    # Scale & predict
    X_scaled = scaler.transform(values)
    probs = model.predict(X_scaled)[0]
    pred_class = np.argmax(probs)
    severity_map = {0:"Severity 1 (Low)", 1:"Severity 2 (Moderate)",
                    2:"Severity 3 (Serious)", 3:"Severity 4 (High)"}
    
    st.subheader(f"🛑 Predicted Risk: **{severity_map[pred_class]}**")
    st.write({severity_map[i]: f"{probs[i]*100:.1f} %" for i in range(len(probs))})

    # Inline SHAP bar for this single instance
    with st.spinner("Calculating local feature explanations…"):
        # Use the same single row as background — fast enough
        background = X_scaled
        explainer = shap.KernelExplainer(model.predict, background)

        # Explain just this one row
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

        # Draw a bar plot and get the figure object
        fig = shap.bar_plot(
            shap_values[0][0],        # SHAP values for this row
            feature_names,            # labels
            show=False,
            max_display=8             # top 8 factors
        )

        st.pyplot(fig)               # display in Streamlit
     # pass figure to Streamlit
