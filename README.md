House Price Prediction | Machine Learning Project       Demo Link: https://house-price-prediction-1527.streamlit.app/

An end-to-end Machine Learning regression project that predicts house prices using Linear Regression, Ridge Regression, and Lasso Regression.
The project covers data preprocessing, model training, evaluation, comparison, serialization, and deployment using Streamlit.

🔍 Problem Statement

Predict the median house price based on socio-economic and environmental features using supervised machine learning techniques.

🧠 Skills & Technologies Used (ATS Keywords)

Python

Machine Learning

Linear Regression

Ridge Regression (L2 Regularization)

Lasso Regression (L1 Regularization)

Feature Scaling (StandardScaler)

Model Evaluation (R², Adjusted R², RMSE)

Model Serialization (Pickle)

Streamlit Deployment

Pandas, NumPy, Scikit-learn

Git & GitHub

📂 Project Structure
House Price Prediction/
│
├── app.py                          # Streamlit inference application
├── House_Price_Prediction.ipynb    # Model training & analysis
├── linear_model.pkl                # Trained Linear Regression model
├── ridge_model.pkl                 # Trained Ridge Regression model
├── lasso_model.pkl                 # Trained Lasso Regression model
├── scaler.pkl                      # Feature scaler
├── HousingData.csv                 # Dataset (optional)
└── README.md

📊 Dataset Overview

Type: Structured tabular data

Target Variable:

MEDV – Median house value (in $1000s)

Sample Features:

CRIM – Crime rate

RM – Average number of rooms

AGE – Age of the house

TAX – Property tax rate

LSTAT – Lower status population percentage

⚙️ Machine Learning Pipeline
1️⃣ Data Preprocessing

Missing value handling using median imputation

Feature scaling using StandardScaler

2️⃣ Model Training

Linear Regression (baseline)

Ridge Regression (reduces overfitting)

Lasso Regression (feature selection)

3️⃣ Model Evaluation

R² Score

Adjusted R²

Root Mean Squared Error (RMSE)

4️⃣ Model Serialization

Trained models and scaler saved using pickle

Enables reuse without retraining

📈 Model Comparison

The application compares predictions from all three models to demonstrate the effect of regularization on output values.

Model	Purpose
Linear Regression	Baseline prediction
Ridge Regression	Controls multicollinearity
Lasso Regression	Feature selection & sparsity
🖥️ Deployment (Streamlit)

The Streamlit web application:

Accepts user input for house features

Applies the same preprocessing pipeline

Predicts house prices in real time

Displays predictions from Linear, Ridge, and Lasso models in a table format.


Predictions are shown in $1000s

Example: 23.33 → $23,330
