#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:54:08 2020

@author: Coronado
@author: Bernard
"""
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import datetime

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
outputs = data[y_name]

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, train_size = 0.7,
                                                    shuffle = True)

##################### learn the Logisitic Regression model #########################
# Default Parameters: (penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
# intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
# max_iter=100, multi_class='auto', verbose=0, warm_start=False, 
# n_jobs=None, l1_ratio=None)

#start an instance of a logistic regression model
logmodel = LogisticRegression()#fit_intercept = True)

#train the model with the training data
logmodel.fit(X_train,Y_train)

#obtain the predictions using the test data
logr_predict = logmodel.predict(X_test)

#print(confusion_matrix(Y_test,rfc_predict))
accuracy = accuracy_score(Y_test,logr_predict)

###################### output Logistic regression report #########################

print('\n')

print('Logistic Regression Report \n') 

print('File name:\t ' + filename)
print('Date:\t'+date_str + '\n')

print(classification_report(Y_test, logr_predict))

print('scikit accuracy score function:     ' + str(round(accuracy*100, 1) ) + '%\n\n')

print('Value counts for hit(1) and non-hit songs (0)')
print(Y_test.value_counts())

#determine the accuracy manually for comfirmation
correct = 0
total = 0
for x in range(len(Y_test)):
    if Y_test.values[x] == logr_predict[x]:
        correct +=1
    total+=1
    
print('manually calculated accuracy:     ' + str(round(correct/total*100, 1)) + '%\n\n')  
    