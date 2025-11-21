import pickle
import streamlit as st
import pandas as pd

df = pd.read_csv('data.csv')
st.title('Laptop Price Predictor')

company = st.selectbox('Company', df['Company'].unique())
product = st.selectbox('Product', df[df['Company'] == company]['Product'].unique())
typename = st.selectbox('Typename', df['TypeName'].unique())
inches = st.slider('Screen Size', min_value = 10, max_value = 18, value = 14)
resolution = st.selectbox('Screen Resolution', df[df['Company'] == company]['ScreenResolution'])
cpu = st.selectbox('CPU', df[df['Company'] == company]['Cpu'].unique())
ram = st.slider('RAM', min_value = 2, max_value = 64, value = 4)
memory = st.selectbox('Memory', df[df['Company'] == company]['Memory'].unique())
gpu = st.selectbox('GPU', df[df['Company'] == company]['Gpu'].unique())
os = st.selectbox('OS', df[df['Company'] == company]['OpSys'].unique())
wt = st.slider('Weight', min_value = 0.5, max_value = 5.0, value = 2.0, step = 0.1)

input_df = pd.DataFrame({
    'Company' : [company],
    'Product' : [product],
    'TypeName' : [typename],
    'Inches': [inches],
    'ScreenResolution' : [resolution],
    'Cpu' : [cpu],
    'Ram' : [ram],
    'Memory' : [memory],
    'Gpu' : [gpu],
    'OpSys' : [os],
    'Weight' : [wt]
})

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

output_df = model.predict(input_df)

if st.button('Show Prediction'):

    st.success(f'Predicted Price is {output_df[0]:.2f}')
