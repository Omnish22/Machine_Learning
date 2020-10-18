import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('Social_Network_Ads.csv')
df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

params = {
    "max_depth":[3,None],
    "max_features":randint(1,2),
    "min_samples_leaf": randint(1,9),
    "criterion": ['gini','entropy']
}

classifier = DecisionTreeClassifier()
randomize = RandomizedSearchCV(classifier,params,cv=5)
randomize.fit(x,y)
print(f'best parameters are {randomize.best_params_}')
print(f'best score is {randomize.best_score_}')