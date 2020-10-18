import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Social_Network_Ads.csv')
df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

k = np.arange(1,21,1)
params = {'n_neighbors':k}
knn = KNeighborsClassifier()

gridsearch = GridSearchCV(knn,params,cv=5)
gridsearch.fit(x,y)

print(gridsearch.best_score_)
print(gridsearch.best_params_)