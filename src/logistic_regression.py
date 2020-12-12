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


#get the data
data = pd.read_csv('filtered_data_set.csv')


#get the inputs
inputs = data.drop('popularity', axis = 1)

#get the outputs
outputs = data['popularity']

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, train_size = 0.7,
                                                    shuffle = True)
#start an instance of a logistic regression model
logmodel = LogisticRegression()

#train the model with the training data
logmodel.fit(X_train,Y_train)

#obtain the predictions using the test data
predictions = logmodel.predict(X_test)

#print a classification report using the predictions and actual output
print('evaluation report')
print(classification_report(Y_test, predictions))

#determine the accuracy alone using scikit function
accuracy = accuracy_score(Y_test,predictions)
print('scikit accuracy score function:     ' + str(round(accuracy*100, 1) ) + '%\n\n')

#determine the accuracy manually for comfirmation
correct = 0
total = 0
for x in range(len(Y_test)):
    if Y_test.values[x] == predictions[x]:
        correct +=1
    total+=1
    
print('manually calculated accuracy:     ' + str(round(correct/total*100, 1)) + '%\n\n')  
    