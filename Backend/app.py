import streamlit as st
import tensorflow as tf
import numpy as np
import urllib

# Load model yang sudah disimpan
model = tf.saved_model.load('lstm_model')
# Fungsi prediksi sederhana untuk model yang di-load
predict_fn = model.signatures['serving_default']

st.title("Aplikasi Prediksi Kondisi Cuaca")
st.write("Masukkan data di bawah ini untuk memprediksi kondisi cuaca.")

# Input dari user
tn = st.number_input("Temperatur Minimum (Tn)")
tx = st.number_input("Temperatur Maksimum (Tx)")
rr = st.number_input("Curah Hujan (RR)")

if st.button("Prediksi"):
    # Siapkan data input (n_timesteps=1, n_features=3)
    input_data = np.array([[tn, tx, rr]], dtype=np.float32)
    input_data_reshaped = input_data.reshape(-1, 1, 3)

    # Konversi ke tensor
    input_tensor = tf.convert_to_tensor(input_data_reshaped)

    # Prediksi (menyesuaikan dengan key output model Keras/TF)
    prediction = predict_fn(input_tensor)
    # Mengambil output pertama (biasanya 'dense_...' atau key pertama)
    output_key = list(prediction.keys())[0]
    result = prediction[output_key].numpy()[0]

    st.subheader("Hasil Prediksi:")
    st.write(f"Prediksi temperatur minimum: {result[0]:.2f}")
    st.write(f"Prediksi temperatur maksimum: {result[1]:.2f}")
    st.write(f"Prediksi curah hujan: {result[2]:.2f}")

print("Password untuk Tunnel:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())