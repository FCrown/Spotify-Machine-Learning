import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Import data set csv
data_set = pd.read_csv('filtered_data_set.csv')

# Set X and y columns
X = data_set[['valence','acousticness','danceability','duration_ms','energy','explicit',
             'instrumentalness','key','liveness','loudness','mode','speechiness','tempo']]
y = data_set['popularity']

# Create the X training and testing set, and Y training and testing set where 70% of the rows
# are for the training set and the rest to the testing set.
x_testing_set, x_training_set, y_testing_set, y_training_set = train_test_split(X, y, test_size=0.7,
                                                                                shuffle=True)

# Create linear regression object
lm = LinearRegression()

# Fit the linear model to the training data
lm.fit(x_training_set,y_training_set)

# Find the predicted y values
predictions = lm.predict(x_testing_set)

# Compare the predicted outputs to a threshold value
scaled_predictions = []
for i in range(0,len(predictions)):
    if(predictions[i] >= 0.5):
        scaled_predictions.append(1)
    else:
        scaled_predictions.append(0)    

# Calculate Accuracy
accuracy_score(y_testing_set, scaled_predictions)




























