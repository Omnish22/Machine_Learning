import random 
import numpy as np 
import pandas as pd
from matplotlib import style 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
style.use('ggplot')

X_train,_ = make_blobs(n_samples=500,centers=3,n_features=2,random_state=20)
# print(type(X_train[0]))
df = pd.DataFrame(dict(x1=X_train[:,0],x2=X_train[:,1]))
plt.scatter(df.iloc[:,0],df.iloc[:,1])
# plt.show()

def init_centroids(df,k):
    centroids = []
    random.seed(0)
    samples = random.sample(range(len(df)),k)
    for i in samples:
        centroids.append(df.iloc[i,:].values)
    return np.asarray(centroids)

# print(init_centroids(df,3))
def dist(x,c):
    return np.sqrt(sum((x-c)**2))

def assignClusters(df,in_centroids):
    cluster = []
    for i in range(len(df)):
        distance = []
        for j in range(len(in_centroids)):
            distance.append(dist(df.iloc[i,:],in_centroids[j]))
        cluster.append(np.argmin(distance))
    return cluster 


def new_centroid(df,k,clusters):
    new_centroid = []
    for i in range(k):
        position = []
        for j in range(len(df)):
            if clusters[j]==i:
                position.append(df.iloc[j,:])
            
        new_centroid.append(np.mean(position,axis=0))
    return np.asarray(new_centroid)


def show_cluster(df,cluster,centroids):    
    df = pd.DataFrame(dict(x1=df.iloc[:,0],x2=df.iloc[:,1],label=cluster))
    groups = df.groupby('label')
    colors={0:'blue',1:'orange',2:'green',3:'yellow',4:'purple',5:'cyan',6:'aquamarine'}
    fig,ax = plt.subplots(figsize=(8,8))
    for key,group in groups:
        group.plot(ax=ax,kind='scatter',x='x1',y='x2',label='label', color=colors[key])
    ax.scatter(centroids[:,0],centroids[:,1],marker='*',s=150)
    plt.show()


def measure_change(Pcentroid, Ncentroid):
    res = 0 
    for a,b in zip(Pcentroid,Ncentroid):
        print("previous centroid \n",Pcentroid)
        print("New centroid \n",Ncentroid)
        res+=dist(a,b)
        print("residual ",res)
    return res 


def K_Means(df,k,iterations=5):
    Pcentroid = init_centroids(df,k)
    cluster = [0]*len(df)
    centroid_change = 100
    # iterations = 
    while centroid_change>0.001:
    # for i in range(iterations):
        cluster = assignClusters(df,Pcentroid)
        show_cluster(df,cluster,Pcentroid)
        Ncentroid = new_centroid(df,k,cluster)
        centroid_change = measure_change(Pcentroid,Ncentroid)
        print(centroid_change)
        Pcentroid = Ncentroid

    return cluster 

cluster = K_Means(df,3,4)
