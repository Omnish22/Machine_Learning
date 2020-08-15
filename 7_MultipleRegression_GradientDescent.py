import numpy as np 
import pandas as pd 

# =====================================

def AddOnes(df,features):
    ''' this function will add array of one to handle first weight 
        after that features will be add to counter other weights '''
    featurearray = df[features]
    ones = np.ones((df.shape[0],1))
    x = np.column_stack((ones,featurearray))
    return x

def GradientDescent(df,features,output,weights,stepsize,tolerance=0.1):
    ''' this function will return updated weights '''

    x = AddOnes(df,features) 
    y = df[output].values
    weights = np.array(weights).reshape(-1,1)

    converge = False
    while not converge:
        y_predict = x @ weights
        error = y_predict - y

        # gradient of cost function 
        GradientSumSquares=0
        for i in range(len(weights)):
            # to calculate derivative just dot product of error and features
            derivative=2 * error.T  @ x[:,i].reshape(-1,1)
            # work of rss is done by gradient  
            GradientSumSquares += derivative**2
            z=stepsize * derivative        
            weights[i] -= np.array([z]).reshape(1,)

        # gradient mag
        GradientMagnitude = np.sqrt(GradientSumSquares)
        if GradientMagnitude < tolerance:
            converge = True
        
    return weights


# -----------------------------------------------------
# upload data

df= pd.read_csv("kc_house_train_data.csv")
testData= pd.read_csv("kc_house_test_data.csv")

# -------- MODEL 1 -------------------------------------


features = ['sqft_living']
output = ['price']
weights = np.array([-47000.,1.])
stepsize=7e-12
tolerance = 2.5e7

w = GradientDescent(df,features,output,weights,stepsize,tolerance)



# Q 1:- 
print(w)
print('\n')

# Q2 :- 
x_test = AddOnes(testData,features)
y_test = testData[output].values.reshape(-1,1)
pred_y = x_test @ w
print(pred_y[:1],y_test[0])

print('\n')

# ---------------- MODEL 2 -----------------------------------
f2 = ['sqft_living','sqft_living15']
o2 = ['price']
initial_weights = np.array([-100000.,1.,1.])
stepsize = 4e-12
tolerance = 1e9

w2 = GradientDescent(df,f2,o2,initial_weights,stepsize,tolerance)
x_test2 = AddOnes(testData,f2)
pred_y2 = x_test2 @ w2
print(pred_y2[0])

print('\n')
# Q4
print(y_test[0],pred_y[0],pred_y2[0])

print('\n')

# Q5
rss1 = np.sum((pred_y-y_test)**2)
rss2 = np.sum((pred_y2 - y_test)**2)
print(rss1,rss2)