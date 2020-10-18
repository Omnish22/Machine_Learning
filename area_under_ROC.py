import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score , roc_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
import numpy as np

# ======= DATA ==========
df = pd.read_csv('Social_Network_Ads.csv')
df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# --------------------------------------------------------------------------------------

# ======= model ==========
lr = LogisticRegression()
lr.fit(X_train,Y_train)
y_predict = lr.predict(x_test)
# ------------------------------

y_pred_proba = lr.predict_proba(x_test)[:,0]

# ===== CURVE =====
fp, tp, threshold = roc_curve(y_test,y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fp,tp)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.show()

# ======= AREA =========
# using roc_auc_score
area1 = roc_auc_score(y_test,y_pred_proba)
print(f'area under roc curve using roc_auc_score is {area1}')

# using cross_val_score
area2 = cross_val_score(lr,x_test,y_test,cv=5,scoring='roc_auc')
print(f'area under roc curve using cross val score is {area2}')
print(np.mean(np.array(area2)))