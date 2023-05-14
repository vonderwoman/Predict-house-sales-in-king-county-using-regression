import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import pickle

# Load the pre-trained model
with open('polynomial_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
df = pd.read_csv('kc_house_data.csv')

# Define the features used for prediction
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'yr_built', 'zipcode']

# Create a polynomial feature transformer
polyfeat = PolynomialFeatures(degree=3)
X_poly = polyfeat.fit_transform(df[features])

# Fit the model on the entire dataset
model.fit(X_poly, df['price'])

# Create a function to predict house prices
def predict_price(features):
    X = polyfeat.transform([features])
    predicted_price = model.predict(X)
    return predicted_price[0]

# Streamlit app
def main():
    st.title('House Price Prediction')
    st.markdown('Enter the features of the house and get the predicted price.')

    # Create input fields for the features
    bedrooms = st.number_input('Number of bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Number of bathrooms', min_value=0.5, max_value=8.0, value=2.5)
    sqft_living = st.number_input('Living area (in square feet)', min_value=500, max_value=10000, value=2000)
    sqft_lot = st.number_input('Lot area (in square feet)', min_value=500, max_value=100000, value=5000)
    floors = st.number_input('Number of floors', min_value=1, max_value=5, value=1)
    waterfront = st.selectbox('Waterfront', ['No', 'Yes'])
    view = st.number_input('View rating (0-4)', min_value=0, max_value=4, value=0)
    grade = st.number_input('Grade (1-13)', min_value=1, max_value=13, value=7)
    yr_built = st.number_input('Year built', min_value=1900, max_value=2023, value=2000)
    zipcode = st.number_input('Zipcode', min_value=10000, max_value=99999, value=98001)

    # Convert categorical feature to numeric
    waterfront = 1 if waterfront == 'Yes' else 0

    # Create a feature vector
    input_features = [bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, grade, yr_built, zipcode]

    # Predict the house price
    predicted_price = predict_price(input_features)

    # Display the predicted price
    st.subheader('Predicted House Price')
    st.write('${:,.2f}'.format(predicted_price))

if __name__ == '__main__':
    main()
