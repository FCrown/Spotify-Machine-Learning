#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:53:11 2020

@author: Coronado
@author: Bernard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:00:17 2020

@author: Coronado
@author: Bernard
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

#get the data
# data = pd.read_csv('filtered_data_set.csv')

filenames = ["Hits_data_set.csv", "2016-19_filtered_data_set.csv"]

filename = filenames[0]

savename = 'pca_'+filename

data = pd.read_csv(filename)

#name of the output
y_name = data.columns[len(data.columns.values)-1]


#get the inputs
inputs = data.drop(y_name, axis = 1)

#get the outputs
outputs = data[y_name]

#PCA 
pca = PCA()

pca.fit(inputs)


####################### Documentation Sklearn PCA #################################
# explained_variance_ : array, shape (n_components,) The amount of variance explained by each of the selected components.

# Equal to n_components largest eigenvalues of the covariance matrix of X.

# New in version 0.18.

# explained_variance_ratio_ : array, shape (n_components,) Percentage of variance explained by each of the selected components.

# If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.
#########################################################################


#see the individal components (eigen_vector in rows)
eigen_vectors =  pca.components_

#get the eigenvalues 
eigen_values = pca.explained_variance_

#get the explained variance ratio (sum_lambda_1_k)/total_sum_lambda)
expl_var_ratio = pca.explained_variance_ratio_



################### PCA output ################################################
#run PCA again to get k components with explained variance > 0.9
pca_k = PCA(n_components = 0.9)
pca_k.fit(inputs)

#transform the inputs
X_k_pca = pca_k.transform(inputs)

#turn new feature matrix into a dataframe
X_k_pca_df = pd.DataFrame(data = X_k_pca)

#get pca optimized data set
pca_data_set = X_k_pca_df.join(outputs)

#export the dataset
pca_data_set.to_csv(savename, index = False)

###################### Visualizations ######################################
##scatter plot
x_pca = pca.transform(inputs)

#plot the first two components and color the datapoints that are hits
plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c = data[y_name], cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.title(savename)
plt.ylabel('Second Principal Component')

df_comp = pd.DataFrame(pca.components_, columns = inputs.columns)

#produce a heatmap showing the principle components with respect to the 
#features
plt.figure(figsize = (12,8))
plt.title(savename + '\nAll PCA Components')
plt.tight_layout(pad=0)
sns.heatmap(df_comp, cmap = 'plasma')

## heat ma[]
df_k_comp = pd.DataFrame(pca_k.components_, columns = inputs.columns)

#produce a heatmap showing the principle components with respect to the 
#features
plt.figure(figsize = (12,8))
plt.title(savename + '\nPCA With k-Components (Explained variance >0.9)')
plt.tight_layout(pad=0)
sns.heatmap(df_k_comp, cmap = 'plasma')

































