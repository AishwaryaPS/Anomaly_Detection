
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.model_selection import StratifiedKFold

#Reading the file

kdd_10 = pd.read_csv("C:\\Users\\sunri\\Downloads\\study material\\sem 5\\Data Analytics\\Project\\Final3.csv")


# In[2]:

from numpy import zeros

#Pre-processing (one-hot encoding, dropping categorical variables, removing nulls)

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
kdd_10         =  kdd_10.drop("label", axis=1)

kdd_10.isnull().values.any()
kdd_10.fillna(0, inplace=True)
print("Are null values present: ",kdd_10.isnull().values.any())


# In[3]:

#The counts of various classes in the result class

classesOfResults= set(kdd_10['result'])
print("The counts for each intrusion class(or normal) in the dataset are:\n ",kdd_10.groupby('result')['result'].count())


# In[4]:

#normalisation of the data

from sklearn.preprocessing import StandardScaler
sc     =  StandardScaler()  
kdd_norm  =  sc.fit_transform(kdd_10.drop("result",axis=1))
kdd_norm = pd.DataFrame(data=kdd_norm,columns=kdd_10.drop("result",axis=1).columns)
kdd_norm.loc[:,"result"]=kdd_10.loc[:,"result"]


# In[5]:

from sklearn.cluster import KMeans

#Function applied on a group by object to make it into a single cluster and returning the cluster center
#This way, a cluster is created for each class

def clustering(x):
    kmeans = KMeans(n_clusters=1, random_state=0).fit(x.drop("result",axis=1))
    return kmeans.cluster_centers_[0]
    


# In[17]:

from sklearn.model_selection import train_test_split

#creating a esting and training dataset

train, test=train_test_split(kdd_norm, test_size=0.7, random_state=7)
test1, test2=train_test_split(test, test_size=0.5, random_state=10)   
classesAbsent=(set(test1['result'])-set(train['result']))
for i in classesAbsent:
    test2 = test2.drop(test2[test2.result == i].index)
train=train.append(test2)
test=test1
kdd_centroids=train.groupby("result").apply(lambda x: clustering(x))
kdd_centroids=kdd_centroids.to_dict()
print("Number of classes in the training data: ",len(kdd_centroids),"\nHence, ",len(classesOfResults)-len(kdd_centroids)," intrusion classes are absent")
print("\nThe intusion classes not present in the training but present in the testing are: ",classesAbsent)
testClustersCount=test.groupby('result')['result'].count()
print("\nThe actual count of each class in the test data: \n",testClustersCount)


# In[11]:

from sklearn.metrics.pairwise import euclidean_distances

l=list(kdd_centroids.values())
distMatrix=euclidean_distances(l,l)
distMatrix=[[round(j) for j in i] for i in distMatrix]
print("Distance Matrix: ")
print(pd.DataFrame(distMatrix))
print("Based on the above distance matrix of distances between centroids, and some trial and error, 18 seems to be an optimal threshold value to classify something as part of none of the aforementioned clusters.")


# In[12]:

#Defining Function to find the closest class cluster for each row in the test data
def closestCluster(x):
    threshold=18
    minDist= 1000000
    closestCluster= 0
    distances={}
    for key, value in kdd_centroids.items():
        dist=euclidean_distances([value], [x])
        distances[key]=dist[0][0]
        if(dist<minDist):
            closestCluster= key
            minDist= dist[0][0]
    if(minDist<=threshold):
        return [closestCluster,distances]
    else:
        return ["new cluster",distances]

distList=[]
results=[]
newClusterCount=[]
testwithoutRes=test.drop("result",axis=1)
newClusterNeeded=[]
for i in range(len(test)):
    val=closestCluster(testwithoutRes.iloc[i])
    if(val[0]=="new cluster"):
        newClusterNeeded.append(testwithoutRes.iloc[i])
        results.append("Unknown Intrusion")
    else:
        results.append(val[0])
    distList.append([val[1]])
        
  
print(results)


# In[33]:

mismatched=[]
for i in range(len(test)):
    if results[i]!= test['result'].iloc[i]:
        mismatched.append(i)
print("\nTotal Number of mismatched classes: ",len(mismatched))

answerClustersCount= pd.DataFrame({'result':results}).groupby('result')['result'].count()
type(answerClustersCount)


# In[15]:

from sklearn import metrics
from sklearn.metrics import accuracy_score
metrics.accuracy_score(test["result"], results)


# In[13]:

'''import numpy as np

newClusterNeeded = pd.DataFrame(newClusterNeeded)
print(newClusterNeeded)
cost = np.zeros(5)
#for i in range(len(newClusterNeeded)):
#    for j in range(len(newClusterNeeded.columns)):
#        newClusterNeeded.iloc[i,j]= round(newClusterNeeded.iloc[i,j],4)
for n in range(2,5):
    unknownCluster = KMeans(n_clusters=n, random_state=0).fit(newClusterNeeded)
    print(unknownCluster.cluster_centers_)
    cost[n] = unknownCluster.append(sum(np.min(cdist(newClusterNeeded, unknownCluster.cluster_centers_, 'euclidean'), axis=1)) / newClusterNeeded.shape[0])
 '''


# In[193]:

'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('MSE')
plt.show()
'''


# In[221]:

print(newClusterNeeded)


# In[222]:

p=21163
print(results[p],distList[p],test['result'].iloc[p])


# In[223]:

for z in range(len(test['result'])):
    if test['result'].iloc[z] in classesAbsent:
        print(z)


# In[224]:

unknowns=[]
for z in range(len(results)):
    if results[z] == "Unknown Intrusion":
        unknowns.append(z)


# In[225]:

unknowns.index(23935)


# In[226]:

print(min(distList[21163][0].values()))
print(results[23935])


# In[227]:

test['result'].iloc[19545]


# In[228]:

count=0
for i in results:
    if i== "Unknown Intrusion":
        count+=1


# In[ ]:




# In[229]:

count


# In[ ]:



