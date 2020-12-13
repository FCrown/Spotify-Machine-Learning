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



#get the data
data = pd.read_csv('filtered_data_set.csv')


#get the inputs
inputs = data.drop('popularity', axis = 1)

#get the outputs
outputs = data['popularity']

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

#determine the number of k features needed 
s = 0
k = 0
for x in expl_var_ratio:
    k+=1
    s+=x
    if s>0.9:
        break

#run PCA again to get k components
pca = PCA(n_components = k)
pca.fit(inputs)

#transform the inputs
X_pca = pca.transform(inputs)

#turn new feature matrix into a dataframe
X_pca_df = pd.DataFrame(data = X_pca)

#get pca optimized data set
pca_data_set = X_pca_df.join(outputs)

#export the dataset
pca_data_set.to_csv("pca_data_set.csv", index = False)




































