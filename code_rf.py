#!/usr/bin/env python3
# -*- coding: utf-8 -*-


print("-"*30)
print("Random Forest Model")
print("-"*30)
print()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from numpy import zeros
from sklearn.decomposition import PCA
import itertools

# Data preprocessing
kdd_10 = pd.read_csv("Final3.csv")
kdd_10=kdd_10.drop('service',axis=1)
kdd_10=kdd_10.drop('flag',axis=1)
alength= len(kdd_10)
protocol_type={'udp': zeros(alength,dtype= int), 'tcp': zeros(alength,dtype= int), 'icmp': zeros(alength,dtype= int), 'other': zeros(alength,dtype= int)}
for i in range(alength):
    protocol_type[kdd_10.iloc[i]["protocol_type"]][i]=1    

# One hot encoding
kdd_10['tcp']  =  pd.Series(protocol_type["tcp"], index=kdd_10.index)
kdd_10['udp']  =  pd.Series(protocol_type["udp"], index=kdd_10.index)
kdd_10['icmp'] =  pd.Series(protocol_type["icmp"], index=kdd_10.index)
kdd_10         =  kdd_10.drop("protocol_type", axis=1)
kdd_10['new_result']=1
kdd_10.loc[kdd_10['result']=='normal.','new_result']=0
result=kdd_10['result']
kdd_10['label'] = "normal"

# Classifying into 4 main attack domains
kdd_10.loc[kdd_10['result'].isin(["ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."]),'label']="rtl"
kdd_10.loc[kdd_10['result'].isin(["back.","land.","neptune.","pod.","smurf.","teardrop."]),'label']="dos"
kdd_10.loc[kdd_10['result'].isin(["ipsweep.","nmap.","portsweep.","satan."]),'label']="probe"
kdd_10.loc[kdd_10['result'].isin(["buffer_overflow.","loadmodule.","perl.","rootkit."]),'label']="u2r"

# Dropping categorical variables
kdd_10=kdd_10.drop('result',axis=1)
kdd_10=kdd_10.drop('new_result',axis=1)

label=kdd_10['label']
kdd_10=kdd_10.drop('label',axis=1)


# PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pca = PCA(n_components=20)

kdd_10.isnull().values.any()
kdd_10.fillna(0, inplace=True)
kdd_10.isnull().values.any()


x = StandardScaler().fit_transform(kdd_10)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
#print(principalDf)
colnames = principalDf.columns
#print(colnames)

# Correlation matrix
import seaborn as sns
# get_ipython().magic('matplotlib inline')
plt.figure(figsize=(10,10)) 

sns.heatmap(kdd_10.corr())



# Function to plot confusion_matrix
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




def perf_measure(ans, conf_list):
    TP = []
    FP = []
    TN = []
    FN = []
    s=0
    total=0
    for i in range(len(conf_list)):
        for j in range(len(conf_list)):
            total+=ans[i][j]
    for i in range(len(conf_list)):
        TP.append(ans[i][i])
        s=0
        for j in range(len(conf_list)):
            if(j!=i):
                s+=ans[j][i]
        FP.append(s)
        FN.append(sum(ans[i])-TP[i])
        TN.append(total-TP[i]-FP[i]-FN[i])
    return(TP, FP, TN, FN)
    



# Random forest model
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import pydot

# 5 fold cross validation
skf=StratifiedKFold(n_splits=5,shuffle=True)

# Removing columns that have 0 importance in the model
pruned_kdd=kdd_10.drop(['duration','land','urgent','num_failed_logins','root_shell','su_attempted','num_file_creations','num_root','num_shells'],axis=1)
pruned_kdd=pruned_kdd.drop(['num_access_files','num_outbound_cmds','is_host_login','is_guest_login','srv_diff_host_rate'],axis=1)


colnames = list(pruned_kdd.columns)
final_accuracy =0.0
final_precision = 0.0
final_recall = 0.0
list_of_all_outputs =list(range(len(pruned_kdd.tcp)))

conf_list=["normal","rtl","dos","u2r","probe"]
picCount=0
for train_indices,test_indices in skf.split(pruned_kdd,label):
    temp = list(test_indices)
    print("Training set length: ",len(train_indices), " and Testing set length: ",len(test_indices))
    
    # Train and test features
    train = pruned_kdd.take(train_indices)
    test = pruned_kdd.take(test_indices)
    
    # Train and test targets
    train_label = label.take(train_indices)
    test_label = label.take(test_indices)
    train.iloc[:,0:(len(train.columns)-1)] = train.iloc[:,0:(len(train.columns)-1)].apply(pd.to_numeric)
    test.iloc[:,0:(len(train.columns)-1)] = test.iloc[:,0:(len(train.columns)-1)].apply(pd.to_numeric)
    # Normalizing the data
    sc = StandardScaler()  
    train_norm  =  sc.fit_transform(train.iloc[:,0:(len(train.columns)-1)])  
    test_norm   =  sc.transform(test.iloc[:,0:(len(train.columns)-1)])
    
    train = pd.DataFrame(data=train_norm,columns=train.iloc[:,0:(len(train.columns)-1)].columns)
    test = pd.DataFrame(data=test_norm,columns=test.iloc[:,0:(len(test.columns)-1)].columns)
    
    
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train, train_label)
    
    # Visualising a tree   
    estimator = classifier.estimators_[5]

    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', feature_names = test.columns.values, class_names = list(set(label)), rounded = True, proportion = False, filled = True)
    
    # Convert to png
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    
    graph.write_png('tree'+str(picCount)+'.png')
    from IPython.display import Image
    Image(filename = 'tree'+str(picCount)+'.png')
    picCount+=1
    
    # Predicting for test dataset
    predicted = classifier.predict(test)
    
    # Appending predictions to final list
    c=0
    for i in temp:
        list_of_all_outputs[i]=predicted[c]
        c+=1
    
    # checking importance of variables (used it to drop the not important attributes)
    importances = list(classifier.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(colnames, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    accuracy = metrics.accuracy_score(test_label, predicted)
    print("Accuracy: ",round(accuracy*100,2))
    print()

# Confusion matrix
ans = confusion_matrix(label, list_of_all_outputs,labels=conf_list)

tp,fp,tn,fn=perf_measure(ans,conf_list)
    
plot_confusion_matrix(ans,conf_list)
for i in range(len(conf_list)):
    accuracy=(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) * 100
    precision = tp[i] / (tp[i]+fp[i]) * 100
    recall = tp[i] / (tp[i]+fn[i]) * 100
    print("Precision of ",conf_list[i],": ",precision,sep="")
    print("Recall of ",conf_list[i],": ",recall,sep="")
    print("Accuracy of ",conf_list[i],": ",accuracy,sep="")
    print()
    final_precision+=precision
    final_recall+=recall
    final_accuracy+=accuracy

final_accuracy=final_accuracy/5
final_precision=final_precision/5
final_recall=final_recall/5

# Appending the categorical columns
kdd_10['result']=result
kdd_10['label']=label
kdd_10['predicted']=list_of_all_outputs

# Getting a binary confusion matrix for normal vs attack
actual=list(range(len(result)))
predicted=list(range(len(result)))
for i in range(len(label)):
    if(label[i]=="normal"):
        actual[i]="normal"
    else:
        actual[i]="attack"
    if(list_of_all_outputs[i]=="normal"):
        predicted[i]="normal"
    else:
        predicted[i]="attack"

bin_ans=confusion_matrix(actual,predicted,labels=["normal","attack"])
plot_confusion_matrix(bin_ans, ["normal","attack"])

# print(feature_importances)
labs=[]
vals=[]
c=0
for i in feature_importances:
    if(c<15):
        labs.append(i[0])
        vals.append(i[1])
    c+=1
index = np.arange(len(labs))
plt.figure(figsize=(20,10))
plt.bar(index, vals)
plt.xlabel('Categories', fontsize=25)
plt.ylabel('Importance', fontsize=25)
plt.xticks(index, labs, fontsize=15, rotation=60)
plt.title('Importance Values',fontsize=25)
plt.show()
# Only top 15 features are shown to make the plotting more understandable. The other attributes had minimal importance

print()
print("Final Accuracy: ",round(final_accuracy,2), "%")
print()

print()
print("-"*30)
print("Random Forest Model")
print("-"*30)