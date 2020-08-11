import numpy as np 
import pandas as pd 
import math

train_df = pd.read_csv("kc_house_train_data.csv")
test_df = pd.read_csv("kc_house_test_data.csv")
# print(train_df.info())

# ===========================================================

# ADDING GIVEN COLUMNS TO BOTH DATA SETS
# bathrooms * bathrooms
train_df['bedrooms_squared'] = train_df['bedrooms'] **2
test_df['bedrooms_squared'] = test_df['bedrooms'] **2

# bathrooms * bedrooms
train_df['bed_bath_rooms'] = train_df['bedrooms'] * train_df['bathrooms']
test_df['bed_bath_rooms'] = test_df['bedrooms'] * test_df['bathrooms']

# log(sqft_living)
train_df['log_sqft_living'] = train_df['sqft_living'].apply(lambda x : np.log(x))
test_df['log_sqft_living'] = test_df['sqft_living'].apply(lambda x : np.log(x))

# lat + long
train_df['lat_plus_long'] = train_df['lat'] + train_df['long']
test_df['lat_plus_long'] = test_df['lat'] + test_df['long']

# ==================================================================================

def MultipleRegressionTraining(data,InputFeatures,OutputFeatures):
    ''' this function is use for getting all weights in model '''
    x = data[InputFeatures].values
    y = data[OutputFeatures].values
    w = (np.linalg.inv(x.T @ x) @ x.T ) @ y
    return {'coefficients': w, 'RSS': (y - x @ w).T  @ (y- x @ w)}


def MultipleRegressionPrediction(TrainData,TestData,InputFeatures,OutputFeatures):
    ''' this function will take training data testing data ,list of input features 
        list of output feature and return predicted value and rss '''

    w = (MultipleRegressionTraining(TrainData,InputFeatures,OutputFeatures))
    w = w['coefficients']
    x = TestData[InputFeatures].values
    y_actual = TestData[OutputFeatures].values

    predict_y = x @ w 
    rss = (y_actual - predict_y).T @ (y_actual - predict_y)
    return {'y_predict': predict_y, 'rss': rss}

# =================================================================================================

#           QUESTIONS
        # ===============

# ques 1 : What is the mean value (arithmetic average)
#          of the 'bedrooms_squared' feature on TEST data? (round to 2 decimal places)
BedSq_avg = np.mean(test_df['bedrooms_squared'])
print(round(BedSq_avg,2))

#--------------------------------------------------------------------

# ques 2 : What is the mean value (arithmetic average)
#          of the 'bed_bath_rooms' feature on TEST data? (round to 2 decimal places)
print(round(np.mean(test_df['bed_bath_rooms']),2))

# -------------------------------------------------------------------------
# Question 3
# What is the mean value (arithmetic average) 
# of the 'log_sqft_living' feature on TEST data? (round to 2 decimal places)
print(round(np.mean(test_df['log_sqft_living']),2))

# --------------------------------------------------------------------------

# What is the mean value (arithmetic average)
#  of the 'lat_plus_long' feature on TEST data? (round to 2 decimal places)
print(round(np.mean(test_df['lat_plus_long']),2))

# ------------------------------------------------------------------------


# model 1
# Question 5
# What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 1?
f1=['sqft_living','bedrooms','bathrooms','lat','long']
y=['price']

model1=MultipleRegressionTraining(train_df,f1,y)
print(model1)

# -----------------------------------------------------------------------------

# model2
# What is the sign (positive or negative) for the coefficient/weight for 'bathrooms' in model 2?

f2 = ['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']

model2= MultipleRegressionTraining(train_df,f2,y)
print(model2)

# -------------------------------------------------------------------------------

# model3
f3 = ['sqft_living','bedrooms','bathrooms',
'lat','long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']

model3 = MultipleRegressionTraining(train_df,f3,y)
print(model3)

# Question 7
# Which model (1, 2 or 3) has lowest RSS on TRAINING Data?

# Which model (1, 2 or 3) has lowest RSS on TESTING Data?

rss1 = MultipleRegressionPrediction(train_df,test_df,f1,y)['rss']
rss2 = MultipleRegressionPrediction(train_df,test_df,f2,y)['rss']
rss3 = MultipleRegressionPrediction(train_df,test_df,f3,y)['rss']

print(rss1,rss2,rss3)