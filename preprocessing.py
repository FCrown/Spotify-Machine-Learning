#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:55:41 2020

@author: Coronado 
@author: Bernard
"""

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
scaled_filtered_ds = scaler.fit_transform(filtered_data_set)

filtered_df = pd.DataFrame(data = scaled_filtered_ds, columns = filtered_data_set.columns)

pop_col = filtered_df['popularity'] + 0.5 - filtered_df['popularity'].mean()


filtered_df['popularity'] = round(pop_col)

filtered_df = filtered_df[['valence','acousticness','danceability','duration_ms','energy','explicit',
                          'instrumentalness','key','liveness','loudness','mode','speechiness','tempo','popularity']]
# Export data to csv
np.savetxt("filtered_data_set.csv", filtered_df, delimiter=",")