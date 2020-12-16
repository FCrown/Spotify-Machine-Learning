#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:05:31 2020

@author: Coronado
"""
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

import matplotlib.pyplot as plt

import seaborn as sns


filename = 'Hits_data_set.csv'

# Import data set csv.
df = pd.read_csv(filename)

savename = "hits_pair_plot.png"

sns_plot = sns.pairplot(df, hue = 'hit', palette = 'deep', plot_kws=dict(marker=".", size= 0.001))

sns_plot.savefig(savename)

