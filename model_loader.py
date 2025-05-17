import tensorflow as tf
import joblib

MODEL_PATH = "accident_risk_model.h5"
SCALER_PATH = "scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
