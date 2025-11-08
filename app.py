import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-price {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 1rem;
    }
    .stSlider > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏠 House Price Prediction</p>', unsafe_allow_html=True)
st.markdown("---")

# Check if model exists
if not os.path.exists('model.pkl'):
    st.error("❌ Model not found! Please run 'train_model.py' first to train the model.")
    st.stop()

# Load the model and preprocessors
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('furnishing_encoder.pkl')
        use_scaling = joblib.load('use_scaling.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, le, use_scaling, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: {e}")
        st.stop()
        return None, None, None, None, None

# Load model
model, scaler, le, use_scaling, feature_names = load_model()

if model is None:
    st.stop()

# Sidebar for input
st.sidebar.header("🏠 House Features")

# Input fields
area = st.sidebar.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=6000, step=100)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=5, value=2)
stories = st.sidebar.slider("Stories", min_value=1, max_value=5, value=2)
parking = st.sidebar.slider("Parking Spaces", min_value=0, max_value=5, value=2)

mainroad = st.sidebar.selectbox("Main Road", ["yes", "no"], index=0)
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"], index=1)
basement = st.sidebar.selectbox("Basement", ["yes", "no"], index=1)
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"], index=1)
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"], index=0)
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"], index=1)
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"], index=1)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 House Details")
    
    # Display input features in a nice format
    features_dict = {
        "Area": f"{area:,} sq ft",
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking Spaces": parking,
        "Main Road": mainroad.title(),
        "Guest Room": guestroom.title(),
        "Basement": basement.title(),
        "Hot Water Heating": hotwaterheating.title(),
        "Air Conditioning": airconditioning.title(),
        "Preferred Area": prefarea.title(),
        "Furnishing Status": furnishingstatus.replace("-", " ").title()
    }
    
    # Create two columns for features display
    col1a, col1b = st.columns(2)
    
    with col1a:
        for i, (key, value) in enumerate(list(features_dict.items())[:6]):
            st.metric(label=key, value=value)
    
    with col1b:
        for i, (key, value) in enumerate(list(features_dict.items())[6:]):
            st.metric(label=key, value=value)

with col2:
    st.header("💡 Prediction")
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad == 'yes' else 0,
            'guestroom': 1 if guestroom == 'yes' else 0,
            'basement': 1 if basement == 'yes' else 0,
            'hotwaterheating': 1 if hotwaterheating == 'yes' else 0,
            'airconditioning': 1 if airconditioning == 'yes' else 0,
            'parking': parking,
            'prefarea': 1 if prefarea == 'yes' else 0,
            'furnishingstatus': le.transform([furnishingstatus])[0]
        }
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        
        # Make prediction
        try:
            if use_scaling:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_df)[0]
            
            # Store prediction in session state
            st.session_state.prediction = prediction
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.stop()

# Display prediction
if 'prediction' in st.session_state:
    st.markdown("---")
    prediction = st.session_state.prediction
    
    # Format prediction
    formatted_price = f"₹{prediction:,.0f}"
    
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Predicted House Price")
    st.markdown(f'<p class="prediction-price">{formatted_price}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional info
    st.info("💡 This prediction is based on the trained machine learning model. Actual prices may vary based on market conditions and other factors.")

# Information section
st.markdown("---")
st.header("📖 About This Model")

with st.expander("ℹ️ Model Information", expanded=False):
    st.write("""
    This house price prediction model uses machine learning to estimate house prices based on various features:
    
    - **Area**: Total area of the house in square feet
    - **Bedrooms**: Number of bedrooms
    - **Bathrooms**: Number of bathrooms
    - **Stories**: Number of floors/stories
    - **Parking**: Number of parking spaces
    - **Main Road**: Whether the house is connected to the main road
    - **Guest Room**: Whether the house has a guest room
    - **Basement**: Whether the house has a basement
    - **Hot Water Heating**: Whether the house has hot water heating
    - **Air Conditioning**: Whether the house has air conditioning
    - **Preferred Area**: Whether the house is in a preferred area
    - **Furnishing Status**: Level of furnishing (furnished, semi-furnished, unfurnished)
    
    The model was trained on historical house price data and uses advanced machine learning algorithms
    to provide accurate predictions.
    """)

with st.expander("📊 Model Performance", expanded=False):
    st.write("""
    The model performance metrics are displayed after training. The model uses cross-validation
    to ensure reliable predictions and is optimized for accuracy.
    
    To see detailed performance metrics, check the output of the training script.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Created By Aman Raj</p>", unsafe_allow_html=True)
