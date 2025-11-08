# House Price Prediction ML Model

A machine learning model with a user-friendly web interface for predicting house prices based on various features.

## Features

- 🏠 Accurate house price prediction using machine learning
- 🎯 Multiple model comparison (Random Forest, Gradient Boosting, XGBoost, Ridge, Lasso)
- 🚀 Hyperparameter tuning for optimal performance
- 💻 Beautiful web UI built with Streamlit
- 📊 Real-time price predictions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

First, train the ML model using the housing data:
```bash
python train_model.py
```

This will:
- Load and preprocess the Housing.csv data
- Train multiple models and compare their performance
- Perform hyperparameter tuning on the best model
- Save the trained model and preprocessors

### Step 2: Run the Web Application

Start the Streamlit web interface:
```bash
streamlit run app.py
```

The application will open in your default web browser where you can:
- Input house features (area, bedrooms, bathrooms, etc.)
- Get instant price predictions
- View model information

## Model Performance

The model uses Ridge Regression with optimized hyperparameters:
- **R² Score**: ~0.65
- **RMSE**: ~1,336,000
- **MAE**: ~978,000

## Features Used for Prediction

- Area (sq ft)
- Number of Bedrooms
- Number of Bathrooms
- Number of Stories
- Parking Spaces
- Main Road (yes/no)
- Guest Room (yes/no)
- Basement (yes/no)
- Hot Water Heating (yes/no)
- Air Conditioning (yes/no)
- Preferred Area (yes/no)
- Furnishing Status (furnished/semi-furnished/unfurnished)

## Files

- `train_model.py` - Model training script
- `app.py` - Streamlit web application
- `Housing.csv` - Training data
- `model.pkl` - Trained model (generated after training)
- `scaler.pkl` - Feature scaler (generated after training)
- `furnishing_encoder.pkl` - Label encoder (generated after training)
- `requirements.txt` - Python dependencies

## Notes

- Make sure to train the model before running the web application
- The model is trained on the provided Housing.csv dataset
- Predictions are in Indian Rupees (₹)
