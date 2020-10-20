import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ======================================================

def plot(x,y,xvar,yvar,color='blue'):    
    plt.scatter(x,y,c=color)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.show()

def PolyData(data,feature,degree):
    x = data[feature]
    x.columns=['power1']
    if degree>1:
        for i in range(2,degree+1):
            x[f'power{i}'] = x.iloc[:,0]**i
    return x.values


def Regression(x,y):
    v = 1
    me = []
    learning_rate = 1e-6
    epochs = 3000
    m = x.shape[1]
    nx = x.shape[0]

    # Initialize weights and bias
    w = np.zeros((m,1))
    b = np.zeros(y.shape)

    # Optimization using gradient descent
    for epoch in range(epochs):
        y_pred = x @ w + b
        error = y - y_pred
        MeanError = 1/m * np.sum(error)

        dw = -2 * x.T @ error 
        db =  2 * error

        w_ = w*(1-2*learning_rate*v)- learning_rate * dw
        b_ = b - learning_rate * db

        w = w_
        b = b_
    return w,b

# ====================================================
#   DATA PROCESSING


df = pd.read_csv("gm_2008_region.csv")

print(df.head())
df.info()

xvar = 'fertility'
yvar = 'life'

df = df.sort_values([xvar,yvar],ascending=True)

power=16
X = PolyData(df,[xvar],power)
Y = df[yvar].values.reshape(-1,1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
Y = scaler_x.fit_transform(Y)

print('=========================================')
print(f'shape of X is {X.shape}')
print(f'shape of Y is {Y.shape}')
# ===========================================================

lr = LinearRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)
error1 = mean_squared_error(Y,y_pred)

plt.scatter(X[:,0],Y)
plt.plot(X[:,0],y_pred,c='orange')
plt.show()

# ===========================================================

#   Ridge Regression
w,b = Regression(X,Y)
y_hat = X @ w + b
error2 = mean_squared_error(Y,y_hat)

print('============================================')
print(f'library error {error1}')
print(f'our error {error2}')

print('===========================================')
plt.scatter(X[:,0],Y)
plt.plot(X[:,0],y_hat,c='orange')
plt.show()