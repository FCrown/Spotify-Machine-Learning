#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:54:12 2020

Description: This script trains a decision tree classifier using a specified csv
             file. In addition, the script will print out a classification report.
             The tree may be visualized if the visualize is set to True. 
             
@author: Coronado
@author: Bernard
"""

import pandas as pd
import numpy as mp

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from six import StringIO

from sklearn.tree import export_graphviz

import pydotplus

import datetime

#get the current time to timestamp the classification results
now = datetime.datetime.now()
date_str = now.strftime("%Y-%m-%d %H:%M:%S")

filenames = ["Hits_data_set.csv", "pca_Hits_data_set.csv", "2016-19_filtered_data_set.csv", "pca_2016-19_filtered_data_set.csv"]

filename = filenames[2]


visualize = False #setting to true makes and saves an image of the tree

##################### get and prepare the data #########################
#get the data
data = pd.read_csv(filename)

#name of the output
y_name = data.columns[len(data.columns.values)-1]

#get the inputs
# inputs = data.drop('popularity', axis = 1)
inputs = data.drop(y_name, axis = 1)

#get the output y value
outputs = data[y_name]

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, train_size = 0.7,
                                                    shuffle = True)

##################### learn the decision tree model #########################
##Documentation: class sklearn.tree.DecisionTreeClassifier(*, 
#criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
#random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
#min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)

#create an instance of a decsion tree classifier
maxleafnodes = 100
depth = 10

#will not limit depth if a visual is not needed
if not visualize:
    dtree = DecisionTreeClassifier(max_depth = depth, max_leaf_nodes = maxleafnodes)
else:
    dtree = DecisionTreeClassifier(max_depth = 3) #set depth for visual

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

print("###################### Report #########################")
#print the confusion matrix
print(confusion_matrix(Y_test, dtree_predict))

print('\n')

print('Decision Tree Report \n') 

print('Depth: '+ str(dtree.tree_.max_depth) + '\t Node Count: ' + str(dtree.tree_.node_count) + 
      '\t Max Leaf Nodes: ' + str(maxleafnodes) + '\n') 

print('File name:\t ' + filename + ".csv")
print('Date:\t'+date_str + '\n')

print(class_report)

print('scikit accuracy score function:     ' + str(round(accuracy*100, 1) ) + '%\n\n')

print('Value counts for hit (1) and non-hit songs (0) in the test output')
print(Y_test.value_counts())

###################### visualizing the decision tree #########################
if visualize:
    #create a savename for the image
    imageName = "DT_" + filename + "_"+ date_str + ".png"
    
    dot_data = StringIO()
    
    export_graphviz(dtree, out_file = dot_data, feature_names = list(inputs.columns),
                    filled = True, rounded = True)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    graph.write_png(imageName)

