import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing  import MinMaxScaler
# ================================


def PolyData(data,feature,degree):
    x = data[feature]
    x.columns=['power1']
    if degree>1:
        for i in range(2,degree+1):
            x[f'power{i}'] = x.iloc[:,0]**i
    return x

def regression(X,y):
    iterations = 500
    learningrate = 0.0005
    n = X.shape[1] # features
    m = X.shape[0] # examples
    
    # initializing weights
    w = np.zeros((n,1))
    b = np.zeros((m,1))
    
    errorlist = list()
    # OPTIMIZATION
    for i in range(iterations):
        y_pred = X @ w + b  # prediction of y        
        error = y - y_pred
        gradcost_w = -2*X.T @ error
        errorlist.append(np.sum(gradcost_w))
        gradcost_b = -2 * error
        w = w - learningrate * gradcost_w
        b = b - learningrate * gradcost_b
        
        if i% 100 ==0:
            print(f'error at {i+1} iteration is  {errorlist[-1]}')
        
    return w,b, y_pred


# =================================================================
# LOAD DATA
df = pd.read_csv("kc_house_data.csv")
df = df.sort_values(["sqft_living",'price'],ascending=True)



# PROCESS DATA
poly_data1 = PolyData(df,['sqft_living'],4)
poly_data1['price'] = df['price']



X=poly_data1.drop('price',axis=1).values
Y = poly_data1['price'].values.reshape(-1,1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
Y= scaler_y.fit_transform(Y)


print(X.shape,Y.shape) #(21613, 1) (21613,)
Y = Y.reshape(-1,1)

# =================================================================
W,b,y_pred= regression(X,Y)

plt.scatter(X[:,0],Y)
plt.plot(X[:,0],y_pred,c='red')
# plt.show()