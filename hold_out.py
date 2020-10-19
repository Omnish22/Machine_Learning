import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Social_Network_Ads.csv')
df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

params = {
    'C':np.logspace(-5,8, 15),
    'penalty':['l1','l2']
}

X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

lr = LogisticRegression()
gridcv = GridSearchCV(lr,params,cv=5)
gridcv.fit(X_train,Y_train)

print(f'best parameters are {gridcv.best_params_}')
print(f'best score is {gridcv.best_score_}')