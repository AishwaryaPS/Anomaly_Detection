#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import code_rf as RandomForest
import code_nn as Nnet
import code_cluster as Clustering
from sklearn import metrics


df = RandomForest.kdd_10
df2 = Nnet.kdd_10
df3 = Clustering.kdd_norm
result=[]
print(set(RandomForest.list_of_all_outputs))
for i in range(len(df['predicted'])):
    print(i)
    #if(df['predicted'][i]==df2['predictions'][i]):
    #    result.append(df['predicted'][i])
    if(not(df['predicted'][i]=="normal" and df2['predictions'][i]=="normal")):
        result.append(Clustering.closestCluster(df3.drop("result",axis=1).iloc[i]))
    else:
        result.append("normal.")

accuracy = metrics.accuracy_score(df['result'], result)
print("The 24 class accuracy is ",round(accuracy*100,2),"%")

result_5=[]
for i in range(len(df['predicted'])):
    print(i)
    if(df['predicted'][i]==df2['predictions'][i]):
        result_5.append(df['predicted'][i])
    elif(not(df['predicted'][i]=="normal" and df2['predictions'][i]=="normal")):
        result_5.append(Clustering.closestCluster(df3.drop("result",axis=1).iloc[i]))
    else:
        result_5.append("normal.")
        
label_final = ["normal" if x=='normal.' 
         else "rtl" if x in ["ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."] 
         else "dos" if x in ["back.","land.","neptune.","pod.","smurf.","teardrop."] 
         else "probe" if x in ["ipsweep.","nmap.","portsweep.","satan."] 
         else "u2r" if x in ["buffer_overflow.","loadmodule.","perl.","rootkit."]
         else x for x in result_5]

accuracy_5 = metrics.accuracy_score(df['label'], label_final)
print("The 24 class accuracy is ",round(accuracy*100,2),"%")
print()
print("The 5 class accuracy is ",round(accuracy_5*100,2),"%")

label_2 = ['normal' if x=='normal'
           else 'attack' for x in label_final]
true_label_2 = ['normal' if x=='normal'
           else 'attack' for x in df['label']]

print(metrics.confusion_matrix(true_label_2, label_2))
accuracy_2 = metrics.accuracy_score(true_label_2, label_2)
print("The 2 class accuracy is ",round(accuracy_2*100,2),"%")