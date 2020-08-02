import pandas as pd 
import numpy as np 

df= pd.read_csv('dataset.csv')
print(df)

X=df['X'].values
y=df['y'].values

# initialize parameters and keep w0 constant 
# we want to visualise RSS with respect to one dimension of parameter 
# for 2 dimension that is for both w0 and w1, RSS will become like mesh grid
w0=0.
w1=-40.

RSS= {} # keys will be RSS and values will be w1 i.e. slope
epochs= 50
w1steps=2.
for epoch in range (epochs):
    SumSqErrors = 0
    for i in range(len(X)):  # to loop over all data points to calculate sum of square of error
        y_predicted = w0 + w1 * X[i]
        error= y[i] - y_predicted
        SqError= error**2
        SumSqErrors += SqError
    # add this Error sq sum as key and corresponding parameters (w0,w1) as values
    RSS[SumSqErrors] = w1
    # chnage w1
    w1= w1 + w1steps

import matplotlib.pyplot as plt 

plt.scatter(RSS.values(),RSS.keys())
plt.xlabel('w1')
plt.ylabel('RSS value')
plt.show()

print(RSS.keys())