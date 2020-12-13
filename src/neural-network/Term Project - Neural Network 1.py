#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix


# In[2]:


# Import data set csv
data_set = pd.read_csv('filtered_data_set.csv')


# In[3]:


# Set X and y columns
X = data_set[['valence','acousticness','danceability','duration_ms','energy','explicit',
             'instrumentalness','key','liveness','loudness','mode','speechiness','tempo']].values
y = data_set['popularity'].values


# In[4]:


# Create the X training and testing set, and Y training and testing set where 70% of the rows
# are for the training set and the rest to the testing set.
x_testing_set, x_training_set, y_testing_set, y_training_set = train_test_split(X, y, test_size=0.7,
                                                                                shuffle=True)


# In[5]:


# Create model we still construct sequentially 
model = Sequential()

# Add dense (every input connected to all units in hidden layer)
# Activation - sigmoid maps between 0 and 1. relu maps to 0 or 1
model.add(Dense(15, input_dim=13, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(10, activation='relu'))

# Output layer 
model.add(Dense(1, activation='sigmoid'))


# In[6]:


# Compile the model. 
# Optimizer - Adam is an efficient optimize to apply gradient descent to the model.
# Metrics - want the accuracy on how the model predicts 
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


# In[7]:


model.fit(x_training_set,y_training_set,epochs=1000, batch_size=64)


# In[8]:


# Get the predicted values with the testing set
test_predictions = model.predict(x_testing_set)


# In[9]:


# Resize to series
test_predictions = pd.Series(test_predictions.reshape(2961,))


# In[10]:


training_score = model.evaluate(x_training_set,y_training_set)
test_score = model.evaluate(x_testing_set,y_testing_set)
print(training_score)
print(test_score)


# In[11]:


# Find predict y values with the x testing set and find accuracy
ynew = model.predict_classes(x_testing_set)
correct=0
for i in range(0,len(ynew)):
    if(ynew[i]==y_testing_set[i]):
        correct = correct + 1
print("Accuracy=", correct/len(test_predictions))


# In[12]:


loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)),y=loss)
plt.title("Training Loss per Epoch");


# In[ ]:




