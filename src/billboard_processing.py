#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:17:15 2020

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

import matplotlib.pyplot as plt

import seaborn as sns

import fitgaussian as fg

#Import the spotify data
spotify_dat = pd.read_csv('data.csv')

#import the billboard data
billboard_dat = pd.read_csv('billboardHot100_1999-2019.csv')

#save name of csv file
savename = "Hits_data_set.csv"

#filter out songs that fall out of 1999-2019 range
spotify_dat = spotify_dat[spotify_dat['year'].values >= 1999]

hit_data = spotify_dat[spotify_dat['name'].isin(billboard_dat['Name'])]

#construct a hit column to add to the scaled dataset
hit_data['hit'] = 1

#initialize hit column to zero
spotify_dat['hit'] = 0

#spotify songs that are billboard hits become 1
spotify_dat['hit'] = spotify_dat['hit'] + hit_data['hit']

#fill in all the nan values with 0
spotify_dat['hit'] = spotify_dat['hit'].fillna(0)

#prepare spotify data to be normalized
reduced_spot = spotify_dat.drop(['year','id', 'name','artists','release_date'], axis=1)


# Normalize the x features to range 0 to 1.
scaler = MinMaxScaler();
scaled_red_spot = scaler.fit_transform(reduced_spot)

#put the data into dataframe with the appropriate labels
filtered_df = pd.DataFrame(data = scaled_red_spot, columns = reduced_spot.columns)

#rearrange columns
filtered_df = filtered_df[['valence','acousticness','danceability','duration_ms','energy','explicit',
                         'instrumentalness','key','liveness','loudness','mode','speechiness','tempo','popularity','hit']]


########################### Plot and export ##########################################
plt.scatter(spotify_dat['year'], filtered_df['popularity'],c = filtered_df['hit'] , cmap = 'plasma')


#export with labels
filtered_df.to_csv(savename, index = False)

