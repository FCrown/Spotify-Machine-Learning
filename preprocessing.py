import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

# Import data set csv.
data_set = pd.read_csv('data.csv')

# Take only the songs from 2016 to 2020
filtered_data_set = data_set[data_set['year'].values > 2015]

# Drop uneeded columns from filtered data
filtered_data_set = filtered_data_set.drop(['year','id', 'name','artists','release_date'], axis=1)

# Normalize the x features to range 0 to 1.
scaler = MinMaxScaler();
filtered_data_set = scaler.fit_transform(filtered_data_set)

# Export data to csv
np.savetxt("filtered_data_set.csv", filtered_data_set, delimiter=",")
































