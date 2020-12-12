import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import data set csv
data_set = pd.read_csv('filtered_data_set.csv')

# Convert to np array
data_set = np.array(data_set)

# Determine the number of rows in the raw data.
num_rows = np.size(data_set, 0)

# Using the number of rows, create a column of 1s to place as the first column in the data set.
ones_col = (np.ones((num_rows, 1)))

# Set X to all values to our inputs.
X = np.concatenate((ones_col, data_set), axis=1)

# Set y to only the output column
y = data_set[:,-1]

# Drop the last column
X = X[:, :-1] 

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

# Calculate the accuracy
correct = 0
total = 0
for i in range(0, len(predictions)):
    if(y_testing_set[i] == scaled_predictions[i]):
        correct = correct + 1
    total = total + 1
print("Correct = ", correct, " Total = ", total, " ",((correct/total)*100), "%")

