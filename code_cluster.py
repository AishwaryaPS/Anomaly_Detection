#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("-"*30)
print("Clustering Model")
print("-"*30)
print()

import pandas as pd
import numpy as np

kdd_10 = pd.read_csv("Final3.csv")

from numpy import zeros
alength= len(kdd_10)
protocol_type={'udp': zeros(alength,dtype= int), 'tcp': zeros(alength,dtype= int), 'icmp': zeros(alength,dtype= int), 'other': zeros(alength,dtype= int)}
result= zeros(alength, dtype= int)
for i in range(alength):
    protocol_type[kdd_10.iloc[i]["protocol_type"]][i]=1    

kdd_10['tcp']  =  pd.Series(protocol_type["tcp"], index=kdd_10.index)
kdd_10['udp']  =  pd.Series(protocol_type["udp"], index=kdd_10.index)
kdd_10['icmp'] =  pd.Series(protocol_type["icmp"], index=kdd_10.index)

kdd_10         =  kdd_10.drop("protocol_type", axis=1)
kdd_10         =  kdd_10.drop("service", axis=1)
kdd_10         =  kdd_10.drop("flag", axis=1)

kdd_10.isnull().values.any()
kdd_10.fillna(0, inplace=True)
kdd_10.isnull().values.any()

result = kdd_10['result']
label = kdd_10['label']
kdd_10.drop(['result','label'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
kdd_norm = sc.fit_transform(kdd_10)
kdd_norm = pd.DataFrame(data=kdd_norm,columns=kdd_10.columns)
kdd_norm['result']=result


from sklearn.cluster import KMeans
i=0
def clustering(x):
    global i
    i+=1
    kmeans = KMeans(n_clusters=1, random_state=0).fit(x.drop("result",axis=1))
    #print(kmeans.cluster_centers_,i)
    return kmeans.cluster_centers_[0]
    

from sklearn.model_selection import train_test_split
train, test=train_test_split(kdd_norm, test_size=0.2, random_state=1)
kdd_centroids=train.groupby("result").apply(lambda x: clustering(x))
kdd_centroids=kdd_centroids.to_dict()


from sklearn.metrics.pairwise import euclidean_distances
results=[]
def closestCluster(x):
    minDist= np.inf
    closestCluster= 0
    #distances=[]
    for key, value in kdd_centroids.items():
        dist=euclidean_distances([value], [x])
        #distances.append(dist)
        if(dist<minDist):
            closestCluster= key
            minDist= dist
    return closestCluster

for i in range(len(test)):
    results.append(closestCluster(test.drop("result",axis=1).iloc[i]))
    #print(i)
print(results)

from sklearn import metrics
print("Accuracy: ",round(metrics.accuracy_score(test["result"], results),2))

print()
print("-"*30)
print("End of Clustering Model")
print("-"*30)