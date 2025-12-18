import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()

# Load encoders and scaler
@st.cache_resource
def load_encoders():
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

encoders = load_encoders()
label_encoder_gender = encoders['label_encoder_gender']
one_hot_encoder_geo = encoders['one_hot_encoder_geography']
scaler = encoders['scaler']

# Streamlit app title
st.title('🏦 Customer Churn Prediction')
st.write('Enter customer details to predict churn probability')

# Create input form
st.sidebar.header('Customer Information')

# Input fields
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1)
geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=40, step=1)
tenure = st.sidebar.slider('Tenure (years)', min_value=0, max_value=10, value=3, step=1)
balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0)
num_of_products = st.sidebar.slider('Number of Products', min_value=1, max_value=4, value=2, step=1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

# Prepare input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Display input data
st.subheader('Customer Details')
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Geography:** {geography}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**Tenure:** {tenure} years")

with col2:
    st.write(f"**Balance:** ${balance:,.2f}")
    st.write(f"**Number of Products:** {num_of_products}")
    st.write(f"**Has Credit Card:** {'Yes' if has_cr_card == 1 else 'No'}")
    st.write(f"**Is Active Member:** {'Yes' if is_active_member == 1 else 'No'}")
    st.write(f"**Estimated Salary:** ${estimated_salary:,.2f}")

# Prediction button
if st.button('Predict Churn', type='primary'):
    # Prepare data for prediction
    input_df = pd.DataFrame([input_data])
    
    # Encode Gender
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
    
    # One-hot encode Geography
    geo_encoded = one_hot_encoder_geo.transform(input_df[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())
    
    # Concatenate encoded geography with input data
    input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]
    
    # Display results
    st.subheader('Prediction Results')
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Churn Probability', f'{prediction_proba:.2%}')
    
    with col2:
        churn_prediction = 'Yes' if prediction_proba > 0.5 else 'No'
        st.metric('Will Churn?', churn_prediction)
    
    with col3:
        risk_level = 'High' if prediction_proba > 0.7 else 'Medium' if prediction_proba > 0.4 else 'Low'
        st.metric('Risk Level', risk_level)
    
    # Progress bar for probability
    st.write('### Churn Probability')
    st.progress(float(prediction_proba))
    
    # Interpretation
    st.write('---')
    if prediction_proba > 0.5:
        st.error(f'⚠️ This customer has a **{prediction_proba:.1%}** probability of churning. Consider retention strategies.')
    else:
        st.success(f'✅ This customer has a **{prediction_proba:.1%}** probability of churning. Low risk.')

# Footer
st.write('---')
st.caption('Customer Churn Prediction Model | Built with TensorFlow & Streamlit')
