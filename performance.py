import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Social_Network_Ads.csv')
df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']



X_train,x_test,Y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

cr = classification_report(y_test,y_pred)
print(cr)