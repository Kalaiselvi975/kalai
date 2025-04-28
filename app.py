# Install necessary libraries (for Google Colab or fresh environments)
# !pip install streamlit tensorflow scikit-learn

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load the saved scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = tf.keras.models.load_model('mlp_model.h5')

# UI Design
st.title("🧠 Parkinson's Disease Prediction App")
st.write("Enter the voice measurements below to predict if a person has Parkinson's Disease.")

# Function to take user input
def user_input_features():
    MDVP_Fo_Hz = st.number_input('Average vocal fundamental frequency (MDVP:Fo(Hz))', min_value=0.0)
    MDVP_Fhi_Hz = st.number_input('Maximum vocal fundamental frequency (MDVP:Fhi(Hz))', min_value=0.0)
    MDVP_Flo_Hz = st.number_input('Minimum vocal fundamental frequency (MDVP:Flo(Hz))', min_value=0.0)
    MDVP_Jitter_percent = st.number_input('Jitter (percent)', min_value=0.0)
    MDVP_Jitter_Abs = st.number_input('Jitter (Abs)', min_value=0.0)
    MDVP_RAP = st.number_input('RAP', min_value=0.0)
    MDVP_PPQ = st.number_input('PPQ', min_value=0.0)
    Jitter_DDP = st.number_input('DDP', min_value=0.0)
    MDVP_Shimmer = st.number_input('Shimmer', min_value=0.0)
    MDVP_Shimmer_dB = st.number_input('Shimmer (dB)', min_value=0.0)
    Shimmer_APQ3 = st.number_input('Shimmer APQ3', min_value=0.0)
    Shimmer_APQ5 = st.number_input('Shimmer APQ5', min_value=0.0)
    MDVP_APQ = st.number_input('MDVP:APQ', min_value=0.0)
    Shimmer_DDA = st.number_input('Shimmer DDA', min_value=0.0)
    NHR = st.number_input('NHR', min_value=0.0)
    HNR = st.number_input('HNR', min_value=0.0)
    RPDE = st.number_input('RPDE', min_value=0.0)
    DFA = st.number_input('DFA', min_value=0.0)
    spread1 = st.number_input('spread1', min_value=-10.0)
    spread2 = st.number_input('spread2', min_value=-10.0)
    D2 = st.number_input('D2', min_value=0.0)
    PPE = st.number_input('PPE', min_value=0.0)

    data = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz,
                      MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP,
                      MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                      Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
                      NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
    return data

# User input
input_data = user_input_features()

# Prediction
if st.button('Predict Parkinson\'s Status'):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction = (prediction > 0.5).astype(int)
    
    if prediction[0][0] == 1:
        st.error('⚠️ The model predicts that the person **may have Parkinson\'s Disease**.')
    else:
        st.success('✅ The model predicts that the person is **healthy**.')

# Footer
st.write("---")
st.caption("Developed with ❤️ using Streamlit")
