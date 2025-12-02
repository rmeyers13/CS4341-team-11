import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import pickle

st.title("RiskMap: Accident Prediction")

# Load model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# User inputs
weather = st.selectbox("Weather", ["Clear", "Rain", "Snow"])
light = st.selectbox("Light Level", ["Daylight", "Dark - lighted roadway"])
road = st.selectbox("Road Condition", ["Dry", "Wet", "Snow/Ice"])

# Simple prediction (mock)
if st.button("Predict Risk"):
    st.write("High risk areas shown in red on map")

    # Show map
    m = folium.Map(location=[42.3249, -71.4008], zoom_start=12)
    folium.Marker([42.3249, -71.4008], popup="High Risk Zone").add_to(m)
    folium_static(m)