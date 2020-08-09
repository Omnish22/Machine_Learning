import numpy as np 
import pandas as pd 

# ========================================================

# DATA IMPORT AND SPLITTING

train_df = pd.read_csv("kc_house_train_data.csv")
test_df = pd.read_csv('kc_house_test_data.csv')

X_train = train_df['sqft_living'].values.reshape(-1,1)
y_train = train_df['price'].values.reshape(-1,1)

X_test = test_df['sqft_living'].values.reshape(-1,1)
y_test = test_df['price'].values.reshape(-1,1)

# ================================================

def LinearParameters(x,y):
    ''' this function will take one feature as an input
        and give parameters i.e. slope and intercept '''
    
    sum_yx = sum(y * x)
    sum_y = sum(y)
    sum_x = sum(x)
    sum_sq_x = sum(x ** 2)

    slope = ((sum_yx - (sum_y * sum_x))/(sum_sq_x - (sum_x * sum_x)/len((x)))).tolist()
    intercept = (sum(y)/len(y) - slope * sum(x)/len(x)).tolist()

    return (*intercept,*slope)

# print(LinearParameters(X_train,y_train))


def Prediction(x_train,y_train,x_test,y_test):
    ''' in this function u have to give training sets and testing sets  
        and in return it will give parameters and prediction value or values
        and rss error '''

    w0,w1=LinearParameters(x_train,y_train)
    y_predict = w0 + w1 * x_test
    
    if type(x_test) is int or float:
        error = (y_predict - y_test) ** 2
    else:
        error = sum((y_predict -y_test)**2)

    return f'SLOPE : {w1} \n INTERCEPT : {w0} \n prediction is : {y_predict} \n error is : {error}'

print(Prediction(X_train,y_train,X_test,y_test))

