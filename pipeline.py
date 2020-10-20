import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# ==================================

df = pd.read_csv("gm_2008_region.csv")
X = df['fertility'].values.reshape(-1,1)
y = df['life'].values.reshape(-1,1)

steps = [
    ('imputation',Imputer(missing_values='NaN',strategy='mean',axis=0)),
    ('scaler',StandardScaler()),
    ('elasticnet',ElasticNet())
]

pipeline = Pipeline(steps)

parameters = {
    'elasticnet__l1_ratio': np.linspace(0,1,30)
}

gcv = GridSearchCV(pipeline,parameters,cv=5)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

gcv.fit(X_train,y_train)

score = gcv.score(X_test,y_test)
best_hypaparameters = gcv.best_params_

y_predict = gcv.predict(X_test)

plt.scatter(X_test,y_test)
plt.plot(X_test,y_predict,c='orange')
plt.show()