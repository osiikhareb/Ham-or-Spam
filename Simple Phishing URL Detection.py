# -*- coding: utf-8 -*-
"""
Phishing URL detection 

@author: Osi
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost

# Read in training and test datasets
trainCSV = os.path.join("phishing-dataset","train.csv")
testCSV = os.path.join("phishing-dataset","test.csv")

traindf = pd.read_csv(trainCSV)
testdf = pd.read_csv(testCSV)

# Get features from train & test
X_train = traindf.values
X_test = testdf.values

# Label training and test datasets of the phishing web pages
y_train = traindf.pop("target").values
y_test = testdf.pop("target").values





## Random Forest
# Train Test and Asses using Random Forest
clf1 = RandomForestClassifier()
clf1.fit(X_train, y_train)
y_test_pred_rf = clf1.predict(X_test)

# Check results
print(accuracy_score(y_test, y_test_pred_rf))
print(confusion_matrix(y_test, y_test_pred_rf))




## XGBoost Classifier
# Train Test and Asses using Random Forest
'''Further/more extensive implementation required using GPU'''
clf2 = XGBoostClassifier()
clf2.fit(X_train, y_train)
y_test_pred_xg = clf2.predict(X_test)

# Check results
print(accuracy_score(y_test, y_test_pred_xg))
print(confusion_matrix(y_test, y_test_pred_xg))