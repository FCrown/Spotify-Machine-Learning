#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:16:06 2020

@author: Coronado
"""

import pandas as pd
import matplotlib.pyplot as plt

filenames = ["Hits_data_set.csv", "Hits_pca_data_set.csv", 'data.csv', 'billboardHot100_1999-2019.csv']

#Import the spotify data
data = pd.read_csv(filenames[0])

yname = 'hit'

pd.crosstab(data.drop(yname,axis = 1),data[yname]).plot(kind='bar')

