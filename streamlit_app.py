import streamlit as st
import dill as pickle
import sys
import types
import pandas as pd
import preprocessing_utils

sys.modules["__main__"].feature_selection = preprocessing_utils.feature_selection

with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

import eda, prediction

with st.sidebar:
    st.title("Page Navigation")
    page = st.radio("Page: ",("EDA", "Model Demo"))

if page == "EDA":
    eda.eda()
else:
    prediction.prediction()
