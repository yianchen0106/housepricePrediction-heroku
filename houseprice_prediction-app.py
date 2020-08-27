import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR

st.write("""
# House prices Prediction App

This app predicts the **USA house** prices!

Data obtained from the [Kaggle data](https://www.kaggle.com/shree1992/housedata?select=output.csv).
""")

st.sidebar.header('User Input Features')
def user_input_features():
    bedrooms = st.sidebar.selectbox('Bedrooms',(1,2,3,4))
    bathrooms = st.sidebar.selectbox('Bathrooms',(1,2,3,4,5,6))
    floors = st.sidebar.selectbox('Floors',(1,1.5,2,2.5) )
    waterfront = st.sidebar.selectbox('Waterfront',(0,1))
    view = st.sidebar.selectbox('View',(0,1,2,3,4))
    condition = st.sidebar.selectbox('Condition(1~5)', (0,1,2,3,4,5))
    yr_built = st.sidebar.selectbox('Build Time(yr)',list(reversed(range(1900,2015))))



    sqft_living = st.sidebar.slider('living areas(sqft)', 800,5660,1000)
    sqft_above = st.sidebar.slider('Area above 1F(sqft)', 550,5010,)
    sqft_basement = st.sidebar.slider('basement area(sqft)', 0,2150,500)


    data = {'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_above':sqft_above,
            'sqft_basement':sqft_basement,
            'yr_built':yr_built}
    features = pd.DataFrame(data, index=[0])
    return features

df_in = user_input_features()


df = pd.read_csv('house_price_cleaned01.csv', index_col=0)
df_price = df['price']
df.drop(columns=['price'], inplace=True)
df_new=pd.concat([df_in, df], axis=0)
df_new.reset_index(drop=True)

df_new = round(df_new, 2)

df_x = (df_new-df_new.min(axis=0))/(df_new.max(axis=0)-df_new.min(axis=0))

x = df_x[:1]


st.subheader('Standardized User Input features')
st.write(x)

load_reg = pickle.load(open('house_svr.pkl', 'rb'))

pred = load_reg.predict(x)

st.subheader('Prediction Price')
price = pred
price = pred*(df_price.max(axis=0)-df_price.min(axis=0))+df_price.min(axis=0)
st.write(price)
