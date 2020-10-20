import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Social_Network_Ads.csv')
df.info()

X = df[['Age','EstimatedSalary']]
y = df[['Purchased']]

steps = [
    ('scaler',StandardScaler()),
    ('classifier',SVC())
]

param = {
    'classifier__C':[1,10,100],
    'classifier__gamma':[0.1,0.01]
}

pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

grcv = GridSearchCV(pipeline,param,cv=5)
grcv.fit(X_train,y_train)
y_pred = grcv.predict(X_test)


clf = SVC()
clf.fit(X_train,y_train)
print(f'score is {grcv.score(X_test,y_test)}')
print(f'score of orginal data {clf.score(X_test,y_test)}')