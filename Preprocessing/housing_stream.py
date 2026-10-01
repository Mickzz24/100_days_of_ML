import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.joblib")
preprocessing = joblib.load("preprocessing.joblib")

st.title("Housing Price Analyzer")
st.write("Find the best price according to your preferences")

df = pd.read_csv("/Users/theerdhasjoy/BIA/ML/Dataset/Housing.csv")

X = df.drop('price', axis= 1)

column_names = X.columns
# st.write(column_names)

user_input = {}

for col in column_names:

    unique_values = X[col].unique()
    selected = st.selectbox(col, unique_values)

    user_input[col] = selected

predictor = st.button("Predict price")

if predictor:
    input_df = pd.DataFrame([user_input])
    # st.write(input_df)

    input_df[preprocessing["num_cols"]] = input_df[
    preprocessing["num_cols"]
    ].fillna(preprocessing["num_fill"])
 
    input_df[preprocessing["cat_cols"]] = input_df[
        preprocessing["cat_cols"]
        ].fillna(preprocessing["cat_fill"])

    for col, mapping in preprocessing["encoders"].items():
        
        input_df[col] = input_df[col].map(mapping)
 
    input_df = input_df[preprocessing["columns"]]
    
    prediction = model.predict(input_df)
    
    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")
 



