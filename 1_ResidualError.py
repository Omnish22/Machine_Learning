import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df=pd.read_csv('Position_Salaries.csv')
print(df)

# SELECTTING FEATURES AND CONVERT INTO NUMPY 
X = df.iloc[:,1].values
y = df.iloc[:,2].values

# suppose that these parameters are selected
x_range=[np.min(X),np.max(X)]
w0=40
w1=52000
x= np.linspace(x_range[0],x_range[1],num=10)
y_predict=w0+w1*x
plt.plot(x,y_predict,c='red')


plt.scatter(X,y)
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()


# RSS

difference= y - y_predict # difference between actual and predict value
residual=0 # initialize rss value
for i in range(len(difference)):
    residual += i**2
print(residual)
