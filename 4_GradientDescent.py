import matplotlib.pyplot as plt 
import math

# set the data points
X = [0,1,2,3,4]
y = [1,3,7,13,21]

plt.scatter(X,y)
plt.xlabel('Input')
plt.ylabel('Output')
# plt.show()

# initialize values
w0 = 0. # initial intercept
w1 = 0. # initial slope
StepSize = 0.05
T = 0.01 # tolerance or threshold value for magnitude of gradient
k=0
epochs = 100 # number of iterations 
# rss = {}

for epoch in range(epochs):
    k = k+1
    # nested loop for calculating prediction
    y_pred = [w0+(w1*i) for i in X ]
    error = [y_pred[i]-y[i] for i in range(len(y))]
    
    # Optimized intercept
    InterceptGradient =sum(error)
    InterceptAdjustment = StepSize * InterceptGradient
    NewIntercept = w0 - InterceptAdjustment

    # Optimized slope
    SlopeGradient = sum([i*j for i,j in zip(X,error)]) # sum(error *  t)
    SlopeAdjustment = StepSize * SlopeGradient
    NewSlope = w1 - SlopeAdjustment

    # magnitude of gradient
    GradMag = math.sqrt(InterceptGradient**2 + SlopeGradient**2)

    # Update intercept and slope
    w0 = NewIntercept
    w1= NewSlope

    # to check weather magnitude of gradient is greater or smaller than threshold value set
    if GradMag < T:
        break
    else:
        continue


print(k)
print(w0,w1)



