import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle
import numpy as np
import preprocessing_utils

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Function untuk 
def prediction():
    def feature_selection(df):
        df = df.copy()
        if "Transaction Date and Time" in df.columns:
            df["Transaction Date and Time"] = pd.to_datetime(df["Transaction Date and Time"], errors="coerce")
            df["Hour"] = df["Transaction Date and Time"].dt.hour
            df["DayOfWeek"] = df["Transaction Date and Time"].dt.dayofweek
            df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
            def get_time_of_day(hour):
                if pd.isnull(hour):
                    return None
                elif 6 <= hour < 18:
                    return "Morning"
                else:
                    return "Night"
            df["TimeOfDay"] = df["Hour"].apply(get_time_of_day)
        return df

    # Title
    st.title("Credit Card Fraud 💳")

    # User Input
    with st.form(key = "Player"):
        # Name Input
        time = st.text_input("Transaction Date and Time:",
                        placeholder= "YYYY-MM-DD HH-MM-SS", help="Isi Sesuai Format")
        # Usia input
        amount = st.number_input("Transaction Amount:",
                        placeholder= 0, min_value=0, help="Dalam INR")
        name = st.text_input("Cardholder Name:",
                        placeholder="John Doe", help="Nama CC")
        card_number = st.text_input("Card Number (Hashed or Encrypted):",
                        placeholder="d4f7a9b2c6e3f1",
                        help="Hash / Encrypted")
        merchant_name = st.text_input("Merchant Name:",
                        placeholder="Amazon India")
        mcc = st.number_input("Merchant Category Code (MCC):", placeholder=5311)
        loc = st.text_input("Transaction Location (City or ZIP Code):", placeholder="Mumbai")
        curr = st.selectbox("Transaction Currency:",["INR", "EUR", "USD"])
        card_type = st.selectbox("Card Type:", ['MasterCard', 'American Express', 'Visa'])
        exp_date = st.text_input("Card Expiration Date:", placeholder="2022-01", help= "YYYY-MM")
        cvv = st.text_input("CVV Code (Hashed or Encrypted)", placeholder="x9z3a1", help= "YYYY-MM")
        resp_code = st.selectbox("Transaction Response Code:",[0,5,12])
        trans_id = st.text_input("Transaction ID:",placeholder="TXN874561")
        prev_trans = st.number_input("Previous Transactions:",min_value=0, max_value=3, help= "Jika lebih dari 3x, masukan saja 3")
        source = st.selectbox("Transaction Source:",["Online", "In-Person"])
        ip_address = st.text_input("IP Address:")
        device = st.selectbox("Device Information:", ['Tablet', 'Mobile', 'Desktop'])
        trans_source = st.selectbox("Transaction Source", ["Online", "In-Person"])
        username = st.text_input("User Account Information:", placeholder="Masukan Username anda")
        notes = st.text_input("Transaction Notes", placeholder="Masukan catatan")
        submit = st.form_submit_button("Predict")

    if submit:

        data_inf = {
    'Transaction Date and Time': time,  
    'Transaction Amount': amount,                        
    'Cardholder Name': name,                    
    'Card Number (Hashed or Encrypted)': card_number, 
    'Merchant Name': merchant_name,                      
    'Merchant Category Code (MCC)': mcc,               
    'Transaction Location (City or ZIP Code)': loc,  
    'Transaction Currency': curr,                      
    'Card Type': card_type,                                
    'Card Expiration Date': exp_date,            
    'CVV Code (Hashed or Encrypted)': cvv,       
    'Transaction Response Code': resp_code,                    
    'Transaction ID': trans_id,                        
    'Previous Transactions': prev_trans,                       
    'Transaction Source': trans_source,                   
    'IP Address': ip_address,                      
    'Device Information': device,                
    'User Account Information': username,
    'Transaction Notes': notes
    }

        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        preds = best_model.predict(data_inf)

        st.write("# Prediction: ", "Fraud" if preds == 1 else "Not Fraud")

if __name__ == '__main__':
    prediction()
