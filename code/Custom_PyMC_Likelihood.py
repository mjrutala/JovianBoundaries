#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:12:27 2024

@author: mrutala
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

import pandas as pd
import datetime as dt
import corner

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime

import BoundaryForwardModeling as JP_BFM
import BoundaryModels as BM


def my_model(params, coords):
    """
    

    Parameters
    ----------
    params : tuple
        constant parameters to plug into model
    coords : numpy array
        multi-dimensional coordinates for fitting

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    model_info = JP_BFM.boundary_model_init('Shuelike')
    
    r, t, p, p_dyn = coords
    
    coordinate_df = pd.DataFrame({'r': r, 
                                  't': t,
                                  'p': p,
                                  'p_dyn': p_dyn})
    
    result_df = JP_BFM.find_ModelOrbitIntersections(coordinate_df, 
                                                    boundary = 'bs',
                                                    model = model_info['model'], 
                                                    params = params)
    
    
    return result_df['within_bs'].to_numpy()

def my_model2(params_list, coords, model_number):
    
    #   Re-construct a DataFrame to feed into find_ModelOrbitIntersections()
    coords_df = pd.DataFrame({'r': coords[0], 't': coords[1], 'p': coords[2], 'p_dyn': coords[3]})
    
    #   Lookup model function from model number
    model_dict = BM.init(BM.lookup(model_number))
    
    result_df = JP_BFM.find_ModelOrbitIntersections(coords_df,
                                                    boundary='bs',
                                                    model = model_dict['model'],
                                                    params = params_list)
    
    return result_df['within_bs'].to_numpy()


def my_loglike(params, sigma, coords, data, model_number):
    
    model = my_model2(params, coords, model_number)
    #return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)
    # return -(0.5 / sigma**2) * ((data - model) ** 2)

    mask0 = data == 0
    
    #   Weights for the residuals: higher where there are fewer points
    weight_value = len(data[mask0]) / len(data)
    weight_value = 1/10
    
    weights = np.zeros(len(data))
    weights[mask0] = 1 - weight_value
    weights[~mask0] = weight_value
    
    residuals = (data - model) * weights
    
    return -(0.5 / sigma**2) * (np.abs(residuals))
    #return -(0.5 / sigma**2) * ((residuals) ** 2)

# define a pytensor Op for our likelihood function
class LogLike(Op):
    def make_node(self, params, sigma, coords, data, model_number) -> Apply:
        # Convert inputs to tensor variables
        param_list = []
        for param in params:
            param_list.append(pt.as_tensor(param))
            
        params = pt.as_tensor(params)
        # r0 = pt.as_tensor(r0)
        # r1 = pt.as_tensor(r1)
        # a0 = pt.as_tensor(a0)
        # a1 = pt.as_tensor(a1)
        #params = pt.as_tensor_variable(params)
        sigma = pt.as_tensor(sigma)
        # x = pt.as_tensor(x)
        coords = pt.as_tensor_variable(coords)
        data = pt.as_tensor(data)
        
        model_number = pt.as_tensor(model_number)
        
        # params = pt.as_tensor([r0, r1, a0, a1])
        inputs = [params, sigma, coords, data, model_number]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [data.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        params, sigma, coords, data, model_number = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        #params = [r0, r1, a0, a1]
        loglike_eval = my_loglike(params, sigma, coords, data, model_number)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

#   Data
location_df, coordinate_df = JP_BFM._ready_DataFrames(dt.datetime(2016, 6, 1), dt.datetime(2020, 6, 1), resolution=10)

model_name = 'Shuelike'
model_dict = BM.init(model_name)
model_number = 1 

coords = coordinate_df.loc[:, ['r', 't', 'p', 'p_dyn']].to_numpy()
sigma = 10
data = location_df['within_bs'].to_numpy()

# create our Op
loglike_op = LogLike()

def custom_dist_loglike(data, params, sigma, coords, model_number):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(params, sigma, coords, data, model_number)


with pm.Model() as potential_model:
    # uniform priors on m and c
    # r0 = pm.Uniform("r0", lower=0.0, upper=100.0)
    #r1 = pm.Uniform("r1", lower=-0.5, upper=0.25)
    # a0 = pm.Uniform("a0", lower=0.0, upper=5.0)
    # a1 = pm.Uniform("a1", lower=-10.0, upper=10.0)
    # r0 = pm.InverseGamma("r0", mu=60, sigma=30)
    # r1 = pm.Normal("r1", mu=-0.2, sigma=0.1)
    # a0 = pm.InverseGamma("a0", mu=2.0, sigma=10.0)
    # a1 = pm.Normal("a1", mu=0, sigma=0.5)
    # breakpoint()
    #constraint = pm.Deterministic('constrant', a0 > -a1)
    #theta = pt.as_tensor([m,c])
    
    params = []
    for param_name in model_dict['param_distributions'].keys():
        param_dist = model_dict['param_distributions'][param_name]
        param = param_dist(param_name, **model_dict['param_descriptions'][param_name])
        params.append(param)
    #params = [r0, r1, a0, a1]
    
    sigma_dist = pm.HalfNormal("sigma_dist", sigma=sigma)
    
    #   !!!! NOT WORKING YET
    coords_std = np.zeros(np.shape(coords))
    coords_std[:,0:3] = 1e-4
    coords_std[:,3] = 1e-4 # artificial pressure errors, none on xyz
    coords_dist = pm.TruncatedNormal('coords_dist', mu=coords.T, sigma=coords_std.T, lower=1e-4, shape=np.shape(coords.T))
    
    # coords_dist = coords.T
    
    # use a Potential instead of a CustomDist
    pm.Potential("likelihood", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    
    idata_potential = pm.sample(tune=0, draws=2000, chains=4)

# plot the traces
az.plot_trace(idata_potential);

figure = corner.corner(idata_potential, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,)

#   Sample a subset of all the MCMC models to calculate the MAE between the data and the model
#   This is used to compare different models to one another
stack = az.extract(idata_potential, num_samples=100)


def plot_FitSummary():

    fig, axs = plt.subplots(figsize=(6,4), nrows=2)

    model_info = BM.init(model_name)

    all_models = []
    mad_list = []
    for i in range(100):
    
        a0, a1, r0, r1 = stack.a0.values[i], stack.a1.values[i], stack.r0.values[i], stack.r1.values[i]
        
        result_df = JP_BFM.find_ModelOrbitIntersections(coordinate_df, 
                                                        boundary = 'bs',
                                                        model = model_info['model'], 
                                                        params = [r0, r1, a0, a1])
        all_models.append(result_df['within_bs'].to_numpy())
        mad_list.append(np.mean(np.abs(location_df['within_bs'] - result_df['within_bs'])))
        
    axs[1].plot(location_df.index, np.mean(all_models, 0), color='black', linewidth=1)
    axs[1].plot(location_df.index, location_df['within_bs'], color='C0', linewidth=1, zorder=0)
    
    axs[1].annotate('Cross-model Mean MAD = {:.3f} +/- {:.3f}'.format(np.mean(mad_list), np.std(mad_list)),
                    (0,1), (1,-1), xycoords='axes fraction', textcoords='offset fontsize')
    axs[1].set(ylim=(0,1.1))
    
    
    model_coords = pd.DataFrame({'t': np.linspace((-0.99)*np.pi, (0.99)*np.pi, 1000),
                                 'p': np.zeros(1000) + (1/2)*np.pi,
                                 'p_dyn': np.zeros(1000) + 0.05})
    for i in range(100):
        
        a0, a1, r0, r1 = stack.a0.values[i], stack.a1.values[i], stack.r0.values[i], stack.r1.values[i]
        
        
        model_coords['r'] = model_info['model'](parameters = (r0, r1, a0, a1), coordinates=model_coords.to_numpy().T)
        
        model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
            
        # model_x = model_coords['r'] * np.cos(model_coords['t'])
        # model_y = model_coords['r'] * np.sin(model_coords['t'])
        axs[0].plot(model_x, model_y, alpha=0.05, color='black')
        
        model_coords = model_coords.drop('r', axis = 'columns')
    
    #   params as a list in the correct order
    params = [stack.mean()[e].values for e in stack.mean()[['r0', 'r1', 'a0', 'a1']]]
    model_coords['r'] = model_info['model'](parameters = params, coordinates = model_coords.to_numpy().T)
    model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
    axs[0].plot(model_x, model_y, color='C0', label='Mean')
    model_coords = model_coords.drop('r', axis='columns')
    
    #   params as a list in the correct order
    params = [stack.median()[e].values for e in stack.median()[['r0', 'r1', 'a0', 'a1']]]
    model_coords['r'] = model_info['model'](parameters = params, coordinates = model_coords.to_numpy().T)
    model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
    axs[0].plot(model_x, model_y, color='C1', label='Median')    
    model_coords = model_coords.drop('r', axis='columns')
    
    axs[0].scatter(*location_df.query('within_bs == 0')[['x', 'y']].to_numpy().T, 
            color='C3', s=4, marker='x')
    
    axs[0].legend()
    
    axs[0].set(xlim = (200,-600), ylim=(-200,200), aspect=1)

plot_FitSummary()