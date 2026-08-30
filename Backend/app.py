from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent / "lstm_model"


@st.cache_resource
def load_predict_fn():
    model = tf.saved_model.load(str(MODEL_DIR))
    return model.signatures["serving_default"]


st.title("Aplikasi Prediksi Cuaca")
st.write("Masukkan data di bawah ini untuk memprediksi cuaca.")

tn = st.number_input("Temperatur Minimum (Tn)")
tx = st.number_input("Temperatur Maksimum (Tx)")
rr = st.number_input("Curah Hujan (RR)")

if st.button("Prediksi"):
    predict_fn = load_predict_fn()
    input_data = np.array([[tn, tx, rr]], dtype=np.float32)
    input_data_reshaped = input_data.reshape(-1, 1, 3)
    input_tensor = tf.convert_to_tensor(input_data_reshaped)

    prediction = predict_fn(input_tensor)
    output_key = list(prediction.keys())[0]
    result = prediction[output_key].numpy()[0]

    st.subheader("Hasil Prediksi:")
    st.write(f"Prediksi temperatur minimum: {result[0]:.2f}")
    st.write(f"Prediksi temperatur maksimum: {result[1]:.2f}")
    st.write(f"Prediksi curah hujan: {result[2]:.2f}")
