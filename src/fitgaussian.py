#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orignially Created on Thu Mar 26 23:32:05 2020

12-13-20: This file implements a previously gaussian fit function that was 
          constructed using the least squares method

@author: Coronado
@author: Bernard
"""

#import os
import pandas as pd
import numpy as np
from scipy import optimize

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters, y, x = None):
    
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)


#function fits a specified set of data given an initial guess for parameters
def fitGaussian(mu, sigma, height, inputData):
    xvals = inputData.index
    yvals = inputData.values
    

    def f(x): return height() * np.exp(-((x-mu())/sigma())**2)

    fit_gaussian = lambda x, mu, s, h: h * np.exp(-((x-mu)/s)**2)

    mu = Parameter(mu)
    sigma = Parameter(sigma)
    height = Parameter(height)

    fitResult = fit(f, [mu, sigma, height], yvals, xvals)

    fitParams = fitResult[0]

    #fitted guassian (x values, mu, sigma, height)
    fitData = fit_gaussian(xvals, fitParams[0], fitParams[1], fitParams[2])
   
    return [fitData,fitParams]


            
