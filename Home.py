import streamlit as st
import pandas as pd
import numpy as np
import joblib
import firebase_admin
import sys
import json

from firebase_admin import credentials, storage
from google.oauth2 import service_account
from google.cloud import storage

icon_path = "ML.png"
st.set_page_config(page_title="Risk Analyser", page_icon = icon_path)

service_account_key = {
    "type": st.secrets.type,
    "project_id": st.secrets.project_id,
    "private_key_id": st.secrets.private_key_id,
    "private_key": st.secrets.private_key,
    "client_email": st.secrets.client_email,
    "client_id": st.secrets.client_id,
    "auth_uri": st.secrets.auth_uri,
    "token_uri": st.secrets.token_uri,
    "auth_provider_x509_cert_url": st.secrets.auth_provider_x509_cert_url,
    "client_x509_cert_url": st.secrets.client_x509_cert_url,
}

cred = credentials.Certificate(service_account_key)

if not firebase_admin._apps: # if firebase initialized already, skip this
    firebase_admin.initialize_app(cred, {'storageBucket': 'ml-take-home-assessment.appspot.com'})

credentials = service_account.Credentials.from_service_account_info(service_account_key)

files = [
    'MoneyLionRiskAnalyzer.pkl',
    'label_encoders.pkl',
    'target_encoder.pkl',
    'anon_ssn_danger.txt'
]

for file_name in files:
    storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob(file_name).download_to_filename(file_name)

st.image(icon_path, width=70)
st.title('MoneyLion Take Home Assessment - Loan Risk Predictor')

def predict(features):
    model = joblib.load('MoneyLionRiskAnalyzer.pkl')
    le = joblib.load('label_encoders.pkl')
    te = joblib.load('target_encoder.pkl')

    column_names = ['fraud_score', 'loanAmount', 'fraud_made', 'paid_off', 'state', 'apr', 'payFrequency', 'days']
    feature_array = np.array([features])
    features_df = pd.DataFrame(feature_array, columns=column_names)

    for column, encoder in le.items():
        if column in features_df.columns:
            features_df[column] = encoder.transform(features_df[column].astype(str))
    
    features_df = features_df.values
    prediction = model.predict(features_df)
    prediction = prediction[0]

    prediction_decode = (prediction > 0.5).astype(int) # Round up the result to either 0 or 1

    # prediction_decode = te.inverse_transform(prediction_decode) #Commented as we are using our custom decoder with custom messages
    if (prediction_decode == 0):
        st.warning("This loan might be risky, kindly review more.")
        st.warning(f"This loan is only {prediction*100:.2f} % safe, proceed with caution", icon="⚠️")
    else:
        st.success("Good news, this loan is generally safe")
        st.success(f"This loan is {prediction*100:.2f} % safe", icon="✅")
    st.write("(Note: Higher score means the model found it to be safer)")

SSN_input = st.text_input('SSN (Social Security Number)')

col1, col2 = st.columns([1,1])
with col1:
    state_input = st.selectbox(
    "State*",
    (
        'IL', 'CA', 'MO', 'NV', 'IN', 'TX', 'UT', 'FL', 'TN', 'MI', 'RI',
        'OH', 'OK', 'NJ', 'VA', 'LA', 'PA', 'SC', 'NC', 'WI', 'NE', 'ID',
        'CT', 'CO', 'WA', 'AL', 'WY', 'DE', 'NM', 'MS', 'KY', 'GA', 'IA',
        'AZ', 'MN', 'SD', 'HI', 'KS', 'ND', 'AK', 'NY', 'MD'
    ),
    index=None,
    placeholder="Select a state",
    )

    frequency_input = st.selectbox(
    "Payment frequency*",
    (
        'Weekly',
        'Bi-Weekly', 
        'Semi-Monthly', 
        'Monthly', 
        'Irregular' 
    ),
    index=None,
    placeholder="Select a frequency",
    )

    loan_amount_input = st.number_input('Loan Amount ($)*', min_value=0)

with col2:
    fraud_score_input = st.number_input('Clarity Fraud Report score*', min_value=0)
    APR_input = st.number_input('APR*', min_value=0)

    column1, column2 = st.columns([1,1])
    with column1:
        tenure_num_input = st.number_input('Loan tenure*', min_value=1)
    with column2:
        tenure_details = st.selectbox(
        "",
        (
            'Days',
            'Week(s)', 
            'Month(s) (30 days)', 
        )
        )

paid_off_input = st.slider('How many MoneyLion loans this applicant has paid off in the past?*', 0, 210, 0)
fraud_made_input = st.number_input('Number of fraud inquiry in the past year*', min_value=0)

if frequency_input == "Weekly":
    freq = "W"
elif frequency_input == "Bi-Weekly":
    freq = "B"
elif frequency_input == "Semi-Monthly":
    freq = "S"
elif frequency_input == "Monthly":
    freq = "M"
else:
    freq = "I"

if tenure_details == "Days":
    tenure = tenure_num_input
elif tenure_details == "Week(s)":
    tenure = tenure_num_input*7
else:
    tenure = tenure_num_input*30

if st.button("Predict risk"):
    if SSN_input:
        with open('anon_ssn_danger.txt', 'r') as file:
            ssn_list = [line.strip() for line in file]

        if SSN_input in ssn_list:
            st.warning('Proceed with caution, SSN has been associated with bad loans prior!', icon="⚠️")
        else:
            st.info('SSN not found in danger list', icon="✅")

    features = [fraud_score_input, loan_amount_input, fraud_made_input, paid_off_input, state_input, APR_input, freq, tenure]
    if any(f is None for f in features):
        st.warning("Oops, some of the mandatory fields are missing, kindly fill in the rest to finish the analysation", icon="☹️")
    else:
        predict(features)


