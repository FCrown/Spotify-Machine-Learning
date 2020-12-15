#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:54:12 2020

Description: This script trains a decision tree classifier using a specified csv
             file. In addition, the script will print out a classification report.
             The tree may be visualized if the visualize is set to True. 
             
             Note: Adjust the depth of the tree at the declaration of the decision
                   tree instance when visualizing to reduce the size.
                   
@author: Coronado
@author: Bernard
"""

import pandas as pd
import numpy as mp

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from six import StringIO

from sklearn.tree import export_graphviz

import pydotplus
 
import matplotlib.image as mpimg

import datetime

#get the current time to timestamp the classification results
now = datetime.datetime.now()
date_str = now.strftime("%Y-%m-%d %H:%M:%S")

###OTHER  DATA SETS
#filename = 'filtered_data_set.csv'
#filename = "pca_data_set.csv"

###HITS DATA SETS
filename = "hits_pca_data_set"

visualize = False

#get the data
data = pd.read_csv(filename + ".csv")

#name of the output
y_name = 'hit'

#get the inputs
# inputs = data.drop('popularity', axis = 1)
inputs = data.drop(y_name, axis = 1)

#get the output y value
outputs = data[y_name]

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, train_size = 0.7,
                                                    shuffle = True)

#create an instance of a decsion tree classifier
dtree = DecisionTreeClassifier()#max_depth = 3) #set depth for visual

#train the decision  tree
dtree.fit(X_train, Y_train)

#predict the results using the trained model given the test data
dtree_predict = dtree.predict(X_test)

#obtain the accuracy score
accuracy = accuracy_score(Y_test,dtree_predict)

#obtain the classification report for the trained model given the test output
#and the predicted values
class_report = classification_report(Y_test, dtree_predict)
###################### output decision tree report #########################

#print the confusion matrix
print(confusion_matrix(Y_test, dtree_predict))

print('\n')

print('Decision Tree Report \n') 

print('File name:\t ' + filename + ".csv")
print('Date:\t'+date_str + '\n')

print(class_report)

print('scikit accuracy score function:     ' + str(round(accuracy*100, 1) ) + '%\n\n')

print('Value counts for hit(1) and non-hit songs (0)')
print(Y_test.value_counts())

###################### visualizing the decision tree #########################
if visualize:
    #create a savename for the image
    imageName = "DT_" + filename + ".png"
    
    dot_data = StringIO()
    
    export_graphviz(dtree, out_file = dot_data, feature_names = list(inputs.columns),
                    filled = True, rounded = True)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    graph.write_png(imageName)







