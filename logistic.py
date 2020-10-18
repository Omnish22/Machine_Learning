import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report, roc_curve
import matplotlib.pyplot as plt 


# =============================================================
# data preprocesseing
df = pd.read_csv("Social_Network_Ads.csv")
df.info()

X = df[['Age','EstimatedSalary']]
y = df['Purchased']

X,x,Y,y = train_test_split(X,y,test_size=0.4,random_state=42)

# ================================================================

# model 
lr = LogisticRegression()
lr.fit(X,Y)
y_pred = lr.predict(x)

# ===============================

# performance
cm = confusion_matrix(y,y_pred)
cr = classification_report(y,y_pred)

print(cm)
print(cr)

y_pred_proba = lr.predict_proba(x)[:,0]
y_pred_proba2 = lr.predict_proba(x)[:,1]

fpr, tpr, threshold = roc_curve(y,y_pred_proba)
fpr2, tpr2, threshold2 = roc_curve(y,y_pred_proba2)
# ====================================================
# PLOT ROC CURVE
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr)
plt.plot(fpr2,tpr2,'g')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

plt.show()