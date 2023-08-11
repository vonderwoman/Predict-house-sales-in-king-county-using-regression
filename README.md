# Predict-house-sales-in-king-county-using-regression

![image](https://miro.medium.com/v2/resize:fit:1183/1*SjQxZnK2N35-Hmom3bwyPg.jpeg)

[![Open In Jupyter Notebook](https://img.shields.io/badge/Open%20in-Jupyter%20Notebook-blue)](https://mybinder.org/v2/gh/username/repo/master?filepath=notebook.ipynb)     [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Anaconda-Server Badge](https://anaconda.org/anaconda/python/badges/version.svg)](https://anaconda.org/anaconda/python)

This case-study aims to predict house sales in king county, Washington State, using regression

# Background
This dataset contains house sale prices for King County,which includes Seattle.It includes houses sold between May 2014, and May 2015 

# Dataset Descriptions

The kc_house_data dataset includes the following features:

  - id: Unique identifier for each house
  - date: Date of the house sale
  - price: Sale price of the house (target variable)
  - bedrooms: Number of bedrooms in the house
  - bathrooms: Number of bathrooms in the house
  - sqft_living: Living area in square feet
  - sqft_lot: Lot area in square feet
  - floors: Number of floors in the house
  - waterfront: Whether the house has a waterfront view (0: No, 1: Yes)
  - view: Overall view rating of the house (0-4)
  - grade: Overall grade given to the house (1-13)
  - yr_built: Year the house was built
  - zipcode: Zipcode of the house
  
## Project Structure

The project is structured as follows:

 - kc_house_data.csv: The dataset file containing the house sales data.
 - polynomial_regression_model.pkl: The pre-trained polynomial regression model for house price prediction.
 - app.py: A Streamlit app that allows users to predict house prices and explore the dataset.


## Associated Tasks
  - Importing necessary libraries: The code imports various libraries such as numpy, pandas, scikit-learn, matplotlib.pyplot, seaborn, and mpl_toolkits.mplot3d.

  - Creating an evaluation metrics DataFrame: An empty DataFrame named "evaluation" is created to store evaluation metrics for different models.

 - Reading and exploring the data: The code reads the data from a CSV file named "kc_house_data.csv" into a pandas DataFrame named df. It then displays the first few rows of the        DataFrame.

 - Splitting the data into training and testing sets: The code uses the train_test_split function from scikit-learn to split the dataset df into training and testing subsets.

 - Simple linear regression: The code performs simple linear regression using the LinearRegression class from scikit-learn. It fits the linear regression model to the training data and makes predictions on the test data. Evaluation metrics such as mean squared error (MSE) and R-squared values are calculated.

 - Data visualization: The code creates a scatter plot of the test data points along with the predicted regression line.

- Complex models: The code defines a function to calculate the adjusted R-squared value. It then creates two complex models using additional features and multiple linear regression. Evaluation metrics for the complex models are calculated and displayed. Data visualization using boxplots is also performed to compare features against price.

- Polynomial regression: The code applies polynomial feature transformation to the complex models using degrees 2 and 3. It fits polynomial regression models to the transformed features and target variable. Evaluation metrics for the polynomial regression models are calculated and displayed. The best model is selected based on the R-squared value.

- Saving the model: The code saves the best polynomial regression model to a file named "polynomial_regression_model.pkl" using the pickle module.

# Some Visualization Output

## Prediction of Regression line
![Predicted Regression Line](https://github.com/vonderwoman/Predict-house-sales-in-king-county-using-regression/blob/main/Output/Predicted%20Regression%20Line.png)

## Pearson Correlation Matrix
![Pearson Correlation Matrix](https://github.com/vonderwoman/Predict-house-sales-in-king-county-using-regression/blob/main/Output/Pearson%20Correlation%20Matrix.png)

## Correlation of bedroom/bathroom features with the target variable
![Correlation of bedroom/bathroom features with the target variable](https://github.com/vonderwoman/Predict-house-sales-in-king-county-using-regression/blob/main/Output/Boxplot-for-some-features.png)

# Instructions

To run this project, follow these steps:

    Clone the repository:

bash

git clone https://github.com/your-username/your-repo.git

    Install the required dependencies:

bash

pip install -r requirements.txt

    Place the kc_house_data.csv file in the project directory.

    Run the Streamlit app:

bash

streamlit run app.py

    Access the app in your browser by clicking on the provided URL.

    In the Streamlit app, enter the features of a house to get the predicted price using the pre-trained model.

    Explore the dataset using the sidebar options in the app, such as displaying the raw data, correlation heatmap, and feature distribution.

# Resources

    Streamlit Documentation
    Scikit-Learn Documentation
    Seaborn Documentation

# App Deployed on Streamlit
![app_deployed](https://github.com/vonderwoman/Predict-house-sales-in-king-county-using-regression/blob/main/asset/gif2_streamlit.gif)


# License

This project is licensed under the MIT License.

Feel free to fork and modify this project according to your needs. Happy predicting!    
