#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:27:49 2018

@author: adityapandey
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


print("-"*30)
print("Neural Network Model")
print("-"*30)
print()

#PreProcessing
kdd_10 = pd.read_csv("Final3.csv")
kdd_10=kdd_10.drop('service',axis=1)
kdd_10=kdd_10.drop('flag',axis=1)
alength= len(kdd_10)
protocol_type={'udp': np.zeros(alength,dtype= int), 'tcp': np.zeros(alength,dtype= int), 'icmp': np.zeros(alength,dtype= int), 'other': np.zeros(alength,dtype= int)}
for i in range(alength):
    protocol_type[kdd_10.iloc[i]["protocol_type"]][i]=1    

kdd_10['tcp']  =  pd.Series(protocol_type["tcp"], index=kdd_10.index)
kdd_10['udp']  =  pd.Series(protocol_type["udp"], index=kdd_10.index)
kdd_10['icmp'] =  pd.Series(protocol_type["icmp"], index=kdd_10.index)
kdd_10         =  kdd_10.drop("protocol_type", axis=1)
kdd_10['new_result']=1
kdd_10.loc[kdd_10['result']=='normal.','new_result']=0

kdd_10.isnull().values.any()
kdd_10.fillna(0, inplace=True)
kdd_10.isnull().values.any()

#Basic Data Analysis
def dataSetAnalysis(df):
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)
    
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)
    
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)
    
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)
    
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 30)

dataSetAnalysis(kdd_10)


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

#Creating Matrix for input to Model
seed = 7
np.random.seed(seed)

dataset = kdd_10.values

result = kdd_10['result']
label = kdd_10['label']
kdd_10.drop(['label','result','new_result'], inplace=True, axis=1)
columns = kdd_10.columns

from sklearn import preprocessing

scaler = preprocessing.RobustScaler()
kdd_10_n = scaler.fit_transform(kdd_10)
kdd_10_n = pd.DataFrame(kdd_10_n, columns=columns)
kdd_10_n.head(5)

#Convert result variable to dummy variables (i.e. one hot encoded)
encoder = LabelEncoder()
encoder.fit(label)
encoded_lab = encoder.transform(label)
dummy_lab = np_utils.to_categorical(encoded_lab)

from ann_visualizer.visualize import ann_viz;

#Create model
def nn_model():
    print("Model Running")
    model = Sequential()
    model.add(Dense(25, input_dim=41, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(5, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    ann_viz(model, title="Neural Network", filename="/Users/adityapandey/Desktop/5th Sem Project/Data Analytics/Nnet.gv")
    return model

estimator = KerasClassifier(build_fn=nn_model, epochs=20, batch_size=5, verbose=0)

#K-Fold Cross Validation
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
pred_results = cross_val_predict(estimator, kdd_10_n, dummy_lab, cv=kfold)

#Evaluating Results
results_labels = pd.Series(encoder.inverse_transform(pred_results))

from sklearn.metrics import confusion_matrix
label = list(label)
results_labels = list(results_labels)

ans = confusion_matrix(list(label), list(results_labels), labels = ["normal","rtl","dos","u2r","probe"])

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
    
final_accuracy =0.0
final_precision = 0.0
final_recall = 0.0
conf_list = ["normal","rtl","dos","u2r","probe"]
tp,fp,tn,fn=perf_measure(ans,conf_list)

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
print()
print("Final Accuracy: %.3f%%" % (final_accuracy))

#Plotting Results
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix:")
    else:
        print('Confusion matrix, without normalization:')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    time.sleep(1)  
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(ans, conf_list)

time.sleep(4)  

attack_label = ["normal" if x=="normal" else "attack" for x in label]
attack_results_label = ["normal" if x=='normal' else "attack" for x in results_labels]
ans_bin = confusion_matrix(attack_label, attack_results_label, labels = ["normal","attack"])
plot_confusion_matrix(ans_bin, ["normal","attack"])

kdd_10['result']=result
kdd_10['label']=label
kdd_10['predictions']=results_labels

print()
print("-"*30)
print("End of Neural Network Model")
print("-"*30)