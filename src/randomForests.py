#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:00:17 2020

Description: This script trains a random forests tree classifier using a specified csv
             file. In addition, the script will print out a report of the model.
             
@author: Coronado
@author: Bernard
"""

import pandas as pd
import numpy as mp

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import datetime

#get the current time to timestamp the classification results
now = datetime.datetime.now()

date_str = now.strftime("%Y-%m-%d %H:%M:%S")

###OTHER  DATA SETS
#filename = 'filtered_data_set.csv'
#filename = "pca_data_set.csv"

filenames = ["Hits_data_set.csv", "pca_Hits_data_set.csv", "2016-19_filtered_data_set.csv", "pca_2016-19_filtered_data_set.csv"]

filename = filenames[2]


##################### get and prepare the data #########################
#get the data
data = pd.read_csv(filename)

#name of the output
y_name = data.columns[len(data.columns.values)-1]

#get the inputs
# inputs = data.drop('popularity', axis = 1)
inputs = data.drop(y_name, axis = 1)


#get the outputs
#outputs = data['popularity']
outputs = data[y_name]


#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, train_size = 0.7,
                                                    shuffle = True)

##################### learn the Random Forests model #########################

#set values for parameters
estimators = 100
crit = "gini" #criterion for splitting

maxleafnodes = None
maxdepth = None

#create an instance of a random forest classifier given the parameters above
rfc = RandomForestClassifier(n_estimators = estimators, 
                             max_depth = maxdepth, max_leaf_nodes = maxleafnodes,
                             criterion = crit)

#train the random forest model
rfc.fit(X_train, Y_train)

#predict the output given the test data
rfc_predict = rfc.predict(X_test)

#obtain the accuracy
accuracy = accuracy_score(Y_test,rfc_predict)

#obtain the classification report for the trained model given the test output
#and the predicted values
class_report = classification_report(Y_test, rfc_predict)


###################### output Random Forest report #########################

print("###################### Report #########################")

print(confusion_matrix(Y_test,rfc_predict))

print('\n')

print('Random Forests Classification Report with ' + str(estimators) + ' estimators\n') 

print('File name:\t ' + filename)
print('Date:\t'+date_str + '\n')
print('Criterion:\t'+crit + '\n')
print('Max Depth: '+ str(maxdepth) + 
       '\t Max Leaf Nodes: ' + str(maxleafnodes) + '\n') 

print(class_report)

print('scikit accuracy score function:     ' + str(round(accuracy*100, 1) ) + '%\n\n')

print('Value counts for hit(1) and non-hit songs (0)')
print(Y_test.value_counts())