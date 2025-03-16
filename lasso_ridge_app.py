import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_absolute_error,r2_score,explained_variance_score,mean_squared_error
from sklearn.datasets import load_diabetes

import streamlit as st 
import pickle

# adding pymongo to store and retreive from mongo db
# these 2 lines are copied from mongo_connect.py which is copied from cluster code in mongodb web
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# these 2 lines are copied from mongo_connect.py which is copied from cluster code in mongodb web
uri = "mongodb+srv://ineuron_user:1985suresh@cluster0.xhhwv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# create a database inside mongo db , its named as student
db = client['diabates']
# create a collection inside the database , its named as student_pred
# next step is we have to store the input and output in the collection
collection = db['diabates_pred']


def load_model():
    # to read the physical pickle file , open it in read binary mode
    with  open("ridge_lasso.pkl",'rb') as file:
        # file is going to return 3 things... as we stored 3 things
        # save them on 3 different variables..
        ridge_model,lasso_model,scaler = pickle.load(file)
    return ridge_model,lasso_model,scaler


def preprocesssing_input_data(diabetes, scaler):
    # user wouldnt know what kind of transformation is being done within the code
    # so we need to do the same kind of transformation that we did in the pickle file 
    # this function will take data , scaler object 
    diabetes_df = pd.DataFrame(diabetes.data,columns = diabetes.feature_names ) # converted to data frame
    print(diabetes_df.head())

    diabetes_df['target'] = diabetes.target
    print(diabetes_df.head())

    y = diabetes_df['target']
    x = diabetes_df.drop(columns=['target'])
    print("diabetes_df",diabetes_df.shape) # shape is 442,11
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    # using scaler transform for both train & test
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled,x_test_scaled,y_train,y_test

def predict_data(data): # data is the input dataset
    # predict function will take data and predict the y value
    # we are calling load_model here which will return 3 values
    ridge_model,lasso_model,scaler = load_model()
    # then we preprocess the values using the other function
    x_train_scaled,x_test_scaled,y_train,y_test = preprocesssing_input_data(data,scaler)

    ridge_model.fit(x_train_scaled,y_train)
    lasso_model.fit(x_train_scaled,y_train)

    # prediction using the processed_data
    ridge_prediction = ridge_model.predict(x_test_scaled)
    lasso_prediction = lasso_model.predict(x_test_scaled)
   
    print("ridge_prediction",ridge_prediction)
    print("shape of ridge_prediction", ridge_prediction.shape)
    print("lasso_prediction",lasso_prediction)
    print("shape of lasso_prediction", lasso_prediction.shape)

    evaluate_model(ridge_prediction,lasso_prediction, x_test_scaled, y_test)

    return ridge_prediction,lasso_prediction

def evaluate_model(ridge_prediction,lasso_prediction, x_test_scaled, y_test):
    # Calculate metrics
    # ridge
    mse_ridge = mean_squared_error(y_test, ridge_prediction)
    rmse_ridge = np.sqrt(mse_ridge)
    mae_ridge = mean_absolute_error(y_test, ridge_prediction)
    r2_ridge = r2_score(y_test, ridge_prediction)
    explained_var_ridge = explained_variance_score(y_test, ridge_prediction)

    print("mse_ridge",mse_ridge)
    print("rmse_ridge",rmse_ridge)
    print("mae_ridge",mae_ridge)
    print("r2_ridge",r2_ridge)
    print("explained_var_ridge",explained_var_ridge)

    # lasso
    mse_lasso = mean_squared_error(y_test, lasso_prediction)
    rmse_lasso = np.sqrt(mse_lasso)
    mae_lasso = mean_absolute_error(y_test, lasso_prediction)
    r2_lasso = r2_score(y_test, lasso_prediction)
    explained_var_lasso = explained_variance_score(y_test, lasso_prediction)

    print("mse_lasso",mse_lasso)
    print("rmse_lasso",rmse_lasso)
    print("mae_lasso",mae_lasso)
    print("r2_lasso",r2_lasso)
    print("explained_var_lasso",explained_var_lasso)
    

    # Calculate Adjusted R-squared ridge & lasso
    n = len(y_test)
    k = x_test_scaled.shape[1]
    adj_r2_ridge = 1 - ((1 - r2_ridge) * (n - 1) / (n - k - 1))
    adj_r2_lasso = 1 - ((1 - r2_lasso) * (n - 1) / (n - k - 1))

    print("adj_r2_ridge",adj_r2_ridge)
    print("adj_r2_lasso",adj_r2_lasso)


def main():

    st.title("diabates prediction")
    st.write("enter your data to get a prediction for diabates")

    age = st.number_input("age",min_value = 1, max_value = 100 , value = 35)
    sex = st.number_input("sex",min_value = 1, max_value = 3 , value = 1)
    bmi = st.number_input("bmi",min_value = 1, max_value = 30 , value = 15)
    bp = st.number_input("bp",min_value = 1, max_value = 30 , value = 15)
    s1 = st.number_input("s1",min_value = 1, max_value = 100 , value = 11)
    s2 = st.number_input("s2",min_value = 1, max_value = 100 , value = 12)
    s3 = st.number_input("s3",min_value = 1, max_value = 100 , value = 13)
    s4 = st.number_input("s4",min_value = 1, max_value = 100 , value = 14)
    s5 = st.number_input("s5",min_value = 1, max_value = 100 , value = 15)
    s6 = st.number_input("s6",min_value = 1, max_value = 100 , value = 16)

    # on the click of a button , we need to predict the output..
    if st.button("predict-your_score"):
        user_data = {
            # map the original column name with the variables we created for UI
            # in a key value pair..
            "age":age,
            "sex":sex,
            "bmi":bmi,
            "bp":bp,
            "s1":s1,
            "s2":s2,
            "s3":s3,
            "s4":s4,
            "s5":s5,
            "s6":s6
        }

        diabetes = load_diabetes() # available in sklearn
        ridge_prediction,lasso_prediction = predict_data(diabetes)
        
        #ridge_prediction,lasso_prediction = predict_data(user_data)
        st.success(f"your prediction result is {ridge_prediction} ")
        st.success(f"your prediction result is {lasso_prediction}")

        # storing all input and output as one dictionary to store it in mongo db
        user_data['ridge_prediction'] = round(float(ridge_prediction[0]),2)
        user_data['lasso_prediction'] = round(float(lasso_prediction[0]),2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)

        #diabetes = load_diabetes() # available in sklearn
        #ridge_prediction,lasso_prediction = predict_data(diabetes)
        

if __name__ == "__main__":
    main()


'''
# this code does not do train test split..
def load_model():
    # to read the physical pickle file , open it in read binary mode
    with  open("ridge_lasso.pkl",'rb') as file:
        # file is going to return 3 things... as we stored 3 things
        # save them on 3 different variables..
        ridge_model,lasso_model,scaler = pickle.load(file)
    return ridge_model,lasso_model,scaler

def preprocesssing_input_data(data, scaler):
    # user wouldnt know what kind of transformation is being done within the code
    # so we need to do the same kind of transformation that we did in the pickle file 
    # this function will take data , scaler object 
    diabetes_df = pd.DataFrame(data)


    diabetes_df_transformed = scaler.transform(diabetes_df)
    print(diabetes_df_transformed)
    return diabetes_df_transformed


def predict_data(data): # data is the input dataset
    # predict function will take data and predict the y value
    # we are calling load_model here which will return 3 values
    ridge_model,lasso_model,scaler = load_model()
    # then we preprocess the values using the other function
    processed_data = preprocesssing_input_data(data,scaler)
    print(processed_data)
    


    # prediction using the processed_data
    ridge_prediction = ridge_model.predict(processed_data)
    lasso_prediction = lasso_model.predict(processed_data)

    print("ridge_prediction",ridge_prediction)
    print("lasso_prediction",lasso_prediction)
    # returns the prediction..
    return ridge_prediction,lasso_prediction

def main():

    diabetes = load_diabetes() # available in sklearn
    diabetes_df = pd.DataFrame(diabetes.data,columns = diabetes.feature_names ) # converted to data frame
    print(diabetes_df.head())

    diabetes_df['target'] = diabetes.target
    print(diabetes_df.head())
    y = diabetes_df['target']

    x = diabetes_df.drop(columns=['target'])
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


    prediction = predict_data(x)

if __name__ == "__main__":
    main()

'''