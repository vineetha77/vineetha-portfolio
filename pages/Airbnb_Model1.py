#  streamlit run C:\DataScience_Projects\Airbnb\airbnb_price_prediction\Airbnb_Model1.py


import streamlit as st
import shap
import pickle
import numpy as np
import json
import xgboost
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Configure Streamlit page layout
st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

# Get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(BASE_DIR, 'xgb_airbnb_model.pkl')
#JSON_PATH = os.path.join(BASE_DIR, 'xgb_airbnb_columns_list.json')
MODEL_PATH = 'xgb_airbnb_model.pkl'
JSON_PATH = 'xgb_airbnb_columns_list.json'

# Load model and selected features
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(JSON_PATH, 'r') as f:
    selected_features = json.load(f)

# Cached SHAP explainer for performance
@st.cache_resource
def get_shap_explainer(_model):
    explainer = shap.Explainer(_model)
    return explainer

# App title
st.markdown("<h1 style='text-align: center; color: darkblue;'>🏠 Airbnb Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Section: Listing Details
st.markdown("### ✨ **Enter Listing Details**")
col1, col2 = st.columns(2)

with col1:
    room_type = st.selectbox("**Room Type**", ['Entire home/apt', 'Private room', 'Shared room'])
    accommodates = st.number_input("**Accommodates**", min_value=1, max_value=20, value=1)
    bathrooms = st.number_input("**Bathrooms**", min_value=1, max_value=10, step=1)

with col2:
    bedrooms = st.number_input("**Bedrooms**", min_value=0, max_value=10, step=1)
    num_reviews = st.number_input("**Number of Reviews**", min_value=0, max_value=1000, step=1)
    city = st.selectbox("**City**", ['NYC', 'LA', 'SF', 'Chicago', 'DC'])

st.markdown("---")

# Build input dictionary
input_dict = {
    'accommodates': accommodates,
    'bathrooms': bathrooms,
    'bedrooms': bedrooms,
    'number_of_reviews': num_reviews,
    f'room_type_{room_type}': 1,
    f'city_{city}': 1
}

# Initialize all features with 0 and update with input values
input_data = {feature: 0 for feature in selected_features}
input_data.update(input_dict)

input_vector = np.array([input_data[feat] for feat in selected_features]).reshape(1, -1)

# Predict and explain
if st.button("🔍 **Predict Price**"):
    log_price = model.predict(input_vector)[0]
    actual_price = np.exp(log_price)

    # Show prediction results
    st.markdown("### 🧾 **Prediction Results**")
    col1, col2 = st.columns(2)
    col1.metric("🧮 **Log Price**", f"{log_price:.2f}")
    col2.metric("💰 **Estimated Price**", f"${actual_price:.2f}")
    st.success("✅ Prediction completed successfully!")

    # SHAP Explanation
    st.markdown("### 🧠 **Why this prediction?**")
    explainer = get_shap_explainer(model)
    shap_values = explainer(input_vector)

    # Convert to DataFrame for display
    shap_df = pd.DataFrame({
        'Feature': selected_features,
        'SHAP Value': shap_values.values[0],
        'Feature Value': input_vector[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    st.write("Top contributing features:")
    st.dataframe(shap_df.head(10).style.background_gradient(cmap='RdBu', subset=['SHAP Value']))

    # Waterfall Plot
    st.markdown("### 📊 **Impact Visualization (Waterfall Plot)**")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)