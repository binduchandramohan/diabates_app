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
db = client['diabates_1']
# create a collection inside the database , its named as student_pred
# next step is we have to store the input and output in the collection
collection = db['diabates_pred_1']


def train_model(diabetes_df):

    y = diabetes_df['target']
    x = diabetes_df.drop(columns=['target'])
    print("diabetes_df",diabetes_df.shape) # shape is 442,11
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    scaler = StandardScaler()
    # using scaler transform for both train & test
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ridge_model = Ridge()
    lasso_model = Lasso()

    ridge_model.fit(x_train_scaled,y_train)
    lasso_model.fit(x_train_scaled,y_train)

    evaluate_model(ridge_model,lasso_model, x_test_scaled, y_test)

    with open("ridge_lasso_1.pkl" , 'wb') as file :
        pickle.dump((ridge_model,lasso_model,scaler),file)

    return ridge_model,lasso_model,scaler

def evaluate_model(ridge_model,lasso_model, x_test_scaled, y_test):

    # predict for test scaled data 
    ridge_prediction = ridge_model.predict(x_test_scaled)
    lasso_prediction = lasso_model.predict(x_test_scaled)

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


def preprocesssing_input_data(user_data, scaler):
    # user wouldnt know what kind of transformation is being done within the code
    # so we need to do the same kind of transformation that we did in the pickle file 
    # this function will take data , scaler object to do standardization
    df = pd.DataFrame([user_data])
    user_data_transformed = scaler.transform(df)
    print("user_data_transformed",user_data_transformed)
    return user_data_transformed



def predict_data(ridge_model,lasso_model,data): # data is the input dataset
    # predict function will take data and predict the y value    
    # prediction using the processed_data

    ridge_prediction = ridge_model.predict(data)
    lasso_prediction = lasso_model.predict(data)
   
    print("ridge_prediction",ridge_prediction)
    print("shape of ridge_prediction", ridge_prediction.shape)
    print("lasso_prediction",lasso_prediction)
    print("shape of lasso_prediction", lasso_prediction.shape)

    return ridge_prediction,lasso_prediction



def main():

    diabetes = load_diabetes() # available in sklearn
    diabetes_df = pd.DataFrame(diabetes.data,columns = diabetes.feature_names ) # converted to data frame
    print(diabetes_df.head())

    diabetes_df['target'] = diabetes.target
    print(diabetes_df.head())

    # train the model with actual data 
    ridge_model,lasso_model,scaler = train_model(diabetes_df)
    
    st.title("diabates prediction")
    st.write("enter your data to get a prediction for diabates")


    age = st.number_input("age",min_value=-0.000000, step=0.001, max_value=0.06, value=0.038076, format="%.5f")
    sex = st.number_input("sex",min_value=-0.000000, step=0.001, max_value=0.06, value=0.050680, format="%.5f")
    bmi = st.number_input("bmi",min_value=-0.000000, step=0.001, max_value=0.07, value=0.061696, format="%.5f")
    bp = st.number_input("bp",min_value=-0.000000, step=0.001, max_value=0.06, value=0.021872, format="%.5f")
    s1 = st.number_input("s1",min_value=-0.000000, step=0.001, max_value=0.06, value=-0.044223, format="%.5f")
    s2 = st.number_input("s2",min_value=-0.000000, step=0.001, max_value=0.06, value=-0.034821, format="%.5f")
    s3 = st.number_input("s3",min_value=-0.000000, step=0.001, max_value=0.06, value=-0.043401, format="%.5f")
    s4 = st.number_input("s4",min_value=-0.000000, step=0.001, max_value=0.06, value=-0.002592, format="%.5f")
    s5 = st.number_input("s5",min_value=-0.000000, step=0.001, max_value=0.06, value=0.019908, format="%.5f")
    s6 = st.number_input("s6",min_value=-0.000000, step=0.001, max_value=0.06, value=-0.017646, format="%.5f")

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

        user_data_transformed = preprocesssing_input_data(user_data, scaler)

        ridge_prediction,lasso_prediction = predict_data(ridge_model,lasso_model,user_data_transformed)
        
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