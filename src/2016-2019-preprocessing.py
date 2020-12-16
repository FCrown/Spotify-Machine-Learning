#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:55:41 2020
@author: Coronado 
@author: Bernard
"""

import pandas as pd

import numpy as np
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import fitgaussian as fg

# Import data set csv.
data_set = pd.read_csv('data.csv')

#savename for the file
savename = "2016-19_filtered_data_set.csv"

# Take only the songs from 2016 to 2020
filtered_data_set = data_set[data_set['year'].values > 2015]

# Drop uneeded columns from filtered data
filtered_data_set = filtered_data_set.drop(['year','id', 'name','artists','release_date'], axis=1)

# Normalize the x features to range from 0 to 1 and reconstruct dataframe
scaler = MinMaxScaler();
scaled_filtered_ds = scaler.fit_transform(filtered_data_set)

filtered_df = pd.DataFrame(data = scaled_filtered_ds, columns = filtered_data_set.columns)


########################### Plot and gaussian Fit ##########################################
#only the songs 0.5*std are considered hits
pop_col = filtered_df['popularity'] #+ 0.5 - mu - (0.5*std)


counts = pop_col.value_counts().sort_index()

plt.plot(counts, '.')
plt.plot(counts [0.05:1], 'g.')

pop_col = pop_col[pop_col.values >= 0.05]

counts = counts[0.05:1]

#estimated height 
h = 500

#get the mean
mu = filtered_df['popularity'].mean()

#get the standard deviation
std =  filtered_df['popularity'].std()

#fit to a gaussian
g_params = fg.fitGaussian(mu,std,h,counts)

#plot the gaussian
plt.plot(counts.index, g_params[0], 'r')

#get the optimized parameters
mu, sigma, height = g_params[1]

##################### Set output to binary and export #####################################

#only get data with popularity greater than 0.05
filtered_df = filtered_df[filtered_df['popularity'].values > 0.05]

#only the songs 0.5*std are considered hits
pop_col = filtered_df['popularity'] + 0.5 - mu - (0.5*std)

#make popularity column 0's and 1's
filtered_df['popularity'] = round(pop_col)

#reorder columns and set popularity to the last column
filtered_df = filtered_df[['valence','acousticness','danceability','duration_ms','energy','explicit',
                          'instrumentalness','key','liveness','loudness','mode','speechiness','tempo','popularity']]
#export with labels
filtered_df.to_csv(savename, index = False)