#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:42:47 2024

@author: mrutala
"""
import pymc as pm
import numpy as np

def TruncatedNormal_Regression(x, y, bounds):
    with pm.Model() as model:
        slope = 0.0 # pm.Uniform('slope', -0.5, 0.5)
        #intercept = pm.Normal('intercept', 80, 16)
        intercept = pm.Uniform('intercept', 30, 200)
        #sigma = pm.HalfNormal('sigma', sigma=30)
        sigma = pm.Uniform('sigma', 0, 50)
        
        #   Assume the obs are normally distributed about a line defined by mu
        normal_dist = pm.Normal.dist(mu=slope * x + intercept, sigma=sigma)
        pm.Truncated("obs", normal_dist, lower=bounds[0], upper=bounds[1], observed=y)
        
    return model

def TruncatedNormal_Fit(x, y, bounds):
    model = TruncatedNormal_Regression(x, y, bounds)
    with model:
        fit = pm.sample()
        
    return fit

def TruncatedNormal_Informed_Regression(x, y, bounds):
    with pm.Model() as model:
        slope = 0.0 # pm.Uniform('slope', -0.5, 0.5)
        intercept = pm.Normal('intercept', np.mean(y), 25)
        #sigma = pm.HalfNormal('sigma', sigma=30)
        sigma = pm.Uniform('sigma', 0, 50)
        
        #   Assume the obs are normally distributed about a line defined by mu
        normal_dist = pm.Normal.dist(mu=slope * x + intercept, sigma=sigma)
        pm.Truncated("obs", normal_dist, lower=bounds[0], upper=bounds[1], observed=y)
        
    return model

def TruncatedNormal_Informed_Fit(x, y, bounds):
    model = TruncatedNormal_Informed_Regression(x, y, bounds)
    with model:
        fit = pm.sample()
        
    return fit