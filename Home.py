import streamlit as st
import pandas as pd
import numpy as np
import joblib
import firebase_admin
import sys

from firebase_admin import credentials, storage
from google.oauth2 import service_account
from google.cloud import storage

icon_path = "ML.png"
firebase_cred = "ml-take-home-assessment-644f6706de7d.json"

st.set_page_config(page_title="Risk Analyser", page_icon = icon_path)

if not firebase_admin._apps: # if firebase initialized already, skip this
    cred = credentials.Certificate(firebase_cred)
    firebase_admin.initialize_app(cred, {'storageBucket': 'ml-take-home-assessment.appspot.com'})

credentials = service_account.Credentials.from_service_account_file(firebase_cred)

files = [
    'MoneyLionRiskAnalyzer.pkl',
    'label_encoders.pkl',
    'target_encoder.pkl'
]

for file_name in files:
    storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob(file_name).download_to_filename(file_name)

st.image(icon_path, width=70)
st.title('MoneyLion Take Home Assessment - Loan Risk Predictor')

def predict(features):
    model = joblib.load('MoneyLionRiskAnalyzer.pkl')
    le = joblib.load('label_encoders.pkl')
    te = joblib.load('target_encoder.pkl')

    column_names = ['fraud_score', 'loan_amount', 'fraud_made', 'paid_off', 'state', 'APR', 'payFrequency']
    feature_array = np.array([features])
    features_df = pd.DataFrame(feature_array, columns=column_names)

    for column, encoder in le.items():
        if column in features_df.columns:
            features_df[column] = encoder.transform(features_df[column].astype(str))
    
    features_df = features_df.values
    prediction = model.predict(features_df)
    st.write(prediction)

    prediction_decode = (prediction > 0.5).astype(int) # Round up the result to either 0 or 1

    # prediction_decode = te.inverse_transform(prediction_decode) #Commented as we are using our custom decoder with custom messages

    if (prediction_decode == 0):
        st.warning("This loan might be risky, kindly review more.")
        st.warning(f"This loan is {prediction*100:.2f} % safe, proceed with warning", icon="⚠️")
    else:
        st.success("Good news, this loan is generally safe")
        st.success(f"This loan is {prediction*100:.2f} % safe", icon="✅")
    st.markdown("Higher score equals to less risk")

st.markdown("\n\nKindly fill in the loan applicant's data\n")

SSN_input = st.text_input('SSN (Social Security Number)')

col1, col2 = st.columns([1,1])
with col1:
    state_input = st.selectbox(
    "State",
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
    "Payment frequency",
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

    loan_amount_input = st.number_input('Loan Amount ($)', min_value=0)

with col2:
    fraud_score_input = st.number_input('Clarity Fraud Report score', min_value=0)
    APR_input = st.number_input('APR (%)', min_value=0)

paid_off_input = st.slider('How many MoneyLion loans this applicant has paid off in the past?', 0, 210, 0)
fraud_made_input = st.number_input('Number of fraud inquiry in the past year', min_value=0)

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

features = [fraud_score_input, loan_amount_input, fraud_made_input, paid_off_input, state_input, APR_input, freq]

if st.button("Predict risk"):
    predict(features)


