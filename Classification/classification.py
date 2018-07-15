# -*- coding: utf-8 -*-
"""
Created on Tue May 29 02:27:32 2018

@author: hasib
"""
#from collections import Counter
#from imblearn.datasets import fetch_datasets
#from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import NearMiss
#from imblearn.metrics import classification_report_imbalanced
#from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
#from sklearn.ensemble import RandomForestClassifier
import numpy as np
#import numpy as np
from sklearn.metrics import confusion_matrix


np.random.seed(1337)  # for reproducibility
#from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score, precision_score,recall_score, f1_score

from dbn.tensorflow.models import SupervisedDBNClassification
from sklearn.preprocessing import MinMaxScaler as Scaler
import pandas as pd
#import tensorflow as tf
f = 1
dataframe = pd.read_csv("diabetesNoMissingFinal.csv") # Let's have Pandas load our dataset as a dataframe
dt = dataframe[0:769]
#dt = dt.drop(["PatientID"], axis = 1)
#dt[['PlasmaGlucose',"PlasmaGlucose","DiastolicBloodPressure","TricepsThickness","SerumInsulin"]] = dt[['PlasmaGlucose',"PlasmaGlucose","DiastolicBloodPressure","TricepsThickness","SerumInsulin"]].replace(0, np.NaN)

# fill missing values with mean column values
#dt.fillna(dt.median(), inplace=True)
#inputX = dt.loc[:, ["PlasmaGlucose", "DiastolicBloodPressure", "SerumInsulin", "BMI"]].values

inputX = dt.loc[:, ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI", "Age"]].values
#inputY = dataframe.loc[:, ["Diabetic"]].values
#inputY = np.transpose(inputY)
inputY = np.genfromtxt("diabetesNoMissingFinal.csv",delimiter=',', skip_header=True)
inputY = inputY[:,8]
inputY = inputY[0:769]




# Loading dataset
#diabetes = load_diabetes()
#X, Y = diabetes.data, diabetes.target
#print(X,Y)
X, Y = inputX, inputY

# Data scaling
#X = (X / 200).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
scaler = Scaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
print(len(X_train))
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[10,8],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.01 ,
                                         n_epochs_rbm= 10 , 
                                         n_iter_backprop=5000,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)


classifier.fit(X_train, Y_train)


# Test
Y_pred = classifier.predict(X_test)
a = accuracy_score(Y_test, Y_pred)
print('Done.\nAccuracy: %f' % a)
print('Done.\nPrecision: %f' % precision_score(Y_test, Y_pred))
print('Done.\nRecall: %f' % recall_score(Y_test, Y_pred))
print('Done.\nf1 score: %f' % f1_score(Y_test, Y_pred))
#print('Done.\nf1 score: %f' % classification_report(Y_test, Y_pred))
#print('Done.\nf1 score: %f' % confusion_matrix(Y_test, Y_pred))
cm1 = confusion_matrix(Y_test, Y_pred)
print(cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+1+cm1[1,1]+1)/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
#precision_score