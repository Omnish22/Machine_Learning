import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np 

df = pd.read_csv('gm_2008_region.csv')
df.info()

x = df['fertility'].values.reshape(-1,1)
y = df['life'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)

params = {
    'l1_ratio': np.linspace(0,1,20)
}

model = ElasticNet()
randomcv = RandomizedSearchCV(model,params,cv=5)
randomcv.fit(X_train,y_train)

y_pred = randomcv.predict(X_test)
score = randomcv.score(X_test,y_test)
mse = mean_squared_error(y_test,y_pred)

print(f'R squared score is {score}')
print(f'mse is {mse}')