import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model_loader import model, scaler

# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Accident Risk Predictor",
                   layout="centered")

st.title("🚦 Road‑Traffic Accident Risk Predictor")
st.write(
    "Enter road, weather and time details to estimate accident severity "
    "risk and see the key contributing factors."
)

# -------------------- Input widgets -----------------------
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
    weather_code = st.number_input("Weather Condition code", value=10, step=1)
    day = st.slider("Day of Week (0=Mon)", 0, 6, 2)

city_code = st.number_input("City code", value=100, step=1)
state_code = st.number_input("State code", value=5, step=1)

# -------------------- Predict button ----------------------
if st.button("Predict"):
    feature_names = [
        "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
        "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)",
        "Weather_Condition", "City", "State", "Hour", "Day"
    ]
    values = np.array([[
        lat, lng, temp, humidity, visibility,
        wind, pressure, weather_code,
        city_code, state_code, hour, day
    ]])

    # Scale & predict
    X_scaled = scaler.transform(values)
    probs = model.predict(X_scaled)[0]
    pred_class = np.argmax(probs)
    severity_map = {
        0: "Severity 1 (Low)",
        1: "Severity 2 (Moderate)",
        2: "Severity 3 (Serious)",
        3: "Severity 4 (High)"
    }

    st.subheader(f"🛑 Predicted Risk: **{severity_map[pred_class]}**")
    st.write({severity_map[i]: f"{probs[i]*100:.1f} %" for i in range(len(probs))})

    # -------------------- SHAP explanation -----------------
    with st.spinner("Calculating local feature explanations…"):

        # 1️⃣ Build a small background sample for KernelExplainer
        background = shap.sample(X_scaled, 50, random_state=0)
        def predict_proba_fn(x):
            # Keras returns a tf‑tensor; convert to numpy
            return model.predict(x)
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background)

        # 2️⃣ SHAP values (list, one array per class)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

        # 3️⃣ Select the SHAP vector for this sample & class
        sample_index = 0
        class_index = pred_class
        sv = shap_values[class_index][sample_index]   # shape (n_features,)

        # 4️⃣ Wrap in a SHAP Explanation so bar() knows the labels
        expl = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value[class_index],
            data=values[sample_index],
            feature_names=feature_names
        )

        # 5️⃣ Plot — shap.plots.bar now needs only (expl, max_display)
        fig = shap.plots.bar(expl, max_display=8)

        st.pyplot(fig)
