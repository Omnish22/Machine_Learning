# this code is for finding values of parameters without using optimization technique
# here parameters are of a line which have intercept as w0 and slope as w1.
# this algo is based on formulaes of w0 and w1 which we get after finding gradient of RSS

import numpy as np
import matplotlib.pyplot as plt 

X= np.array([0,1,2,3,4])
y= np.array([1,3,7,13,21])

plt.scatter(X,y)


N=len(X)
# FINDING SLOPE
SumProd_yx=0 # first multiply y and x and then take sum of all y*x
for i,j in zip(X,y):
    product= i*j # multiply
    SumProd_yx += product # sum

ProdSum_yx=sum(X) * sum(y) # first sum all y and x and then multiply result

SumSq_x=0 # this is sum of squares of x i.e x1^2 + x2^2 ....+ xn^2
for i in X:
    SumSq_x += i**2

SumProd_x= sum(X)*sum(X) # product of sum of all X

numerator = SumProd_yx - ProdSum_yx/N
denomenator = SumSq_x - SumProd_x/N

w1 = numerator / denomenator


w0 = sum(y)/N - (w1 * (sum(X)/N)) # using : y = intercept + (slope * X) .... w0 is intercept

y_pred= [w0+w1*i for i in X]
plt.scatter(X,y_pred,c='black')

x= [i for i in range(min(X)-1,max(X)+1)]
y_line = [w0 +w1*i for i in range(min(X)-1,max(X)+1)]
plt.plot(x,y_line,c='r')
plt.show()