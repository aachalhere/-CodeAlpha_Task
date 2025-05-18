import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/60/46/ff/6046ff0749e72af99ffdad4c43d6977c.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.8);
}

h1, h2, h3, .stTextInput, .stNumberInput, .stSelectbox {
    color: black;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Data
df = pd.read_csv("car data.csv")
df = df.drop('Owner', axis=1)
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)

X = df.drop(['Selling_Price', 'Car_Name'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

st.title("🚗 Car Price Prediction ")

# User Input
car_names = df['Car_Name'].unique()
selected_car = st.sidebar.selectbox("Select Your Car", sorted(car_names))

year = st.sidebar.slider("Year of Purchase", 1990, 2025, 2015)
present_price = st.sidebar.number_input("Present Price (in Lakhs)", 0.0, 50.0, step=0.5, key="price")
driven_kms = st.sidebar.number_input("Kilometers Driven", 0, 500000, step=500, key="kms")
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'], key="fuel")
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'], key="seller")
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'], key="trans")

input_dict = {
    'Year': year,
    'Present_Price': present_price,
    'Driven_kms': driven_kms,
    'Fuel_Type_Diesel': 1 if fuel_type == 'Diesel' else 0,
    'Fuel_Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
    'Selling_type_Individual': 1 if seller_type == 'Individual' else 0,
    'Transmission_Manual': 1 if transmission == 'Manual' else 0
}

input_data = pd.DataFrame([input_dict])
prediction = model.predict(input_data)[0]

st.subheader(f" Predicted Selling Price for {selected_car}:")
st.markdown(
    f"<h2 style='color: blue;'>₹ {round(prediction, 2)} Lakhs</h2>",
    unsafe_allow_html=True
)

