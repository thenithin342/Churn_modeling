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
    return load_model('regression_model.h5')

model = load_trained_model()

# Load encoders and scaler
@st.cache_resource
def load_encoders():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        one_hot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return label_encoder_gender, one_hot_encoder_geo, scaler

label_encoder_gender, one_hot_encoder_geo, scaler = load_encoders()

# Streamlit app title
st.title('💰 Salary Prediction App')
st.write('Predict estimated salary based on customer features')

# Create input form
st.sidebar.header('Customer Information')

# Input fields
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650, step=1)
geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=35, step=1)
tenure = st.sidebar.slider('Tenure (years)', min_value=0, max_value=10, value=5, step=1)
balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
num_of_products = st.sidebar.slider('Number of Products', min_value=1, max_value=4, value=2, step=1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
exited = st.sidebar.selectbox('Exited', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

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
    'Exited': exited
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
    st.write(f"**Exited:** {'Yes' if exited == 1 else 'No'}")

# Prediction button
if st.button('Predict Salary', type='primary'):
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
    prediction = model.predict(input_scaled, verbose=0)
    predicted_salary = prediction[0][0]
    
    # Display results
    st.subheader('Prediction Results')
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Predicted Salary', f'${predicted_salary:,.2f}')
    
    with col2:
        salary_category = 'High' if predicted_salary > 100000 else 'Medium' if predicted_salary > 50000 else 'Low'
        st.metric('Salary Category', salary_category)
    
    with col3:
        monthly_salary = predicted_salary / 12
        st.metric('Monthly Salary', f'${monthly_salary:,.2f}')
    
    # Visualization
    st.write('---')
    st.write('### Salary Breakdown')
    
    # Create a simple comparison
    salary_data = {
        'Category': ['Predicted Annual', 'Predicted Monthly', 'Average Bank Salary'],
        'Amount': [predicted_salary, monthly_salary, 100000]
    }
    salary_df = pd.DataFrame(salary_data)
    
    st.bar_chart(salary_df.set_index('Category')['Amount'])
    
    # Additional insights
    st.write('---')
    st.info(f'💡 **Insight:** Based on the customer profile, the estimated annual salary is **${predicted_salary:,.2f}**')

# Footer
st.write('---')
st.caption('Salary Prediction Model | Built with TensorFlow & Streamlit')
