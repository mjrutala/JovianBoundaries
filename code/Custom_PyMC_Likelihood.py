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


def my_loglike(params, sigma, coords, data):
    # We fail explicitly if inputs are not numerical types for the sake of this tutorial
    # As defined, my_loglike would actually work fine with PyTensor variables!
    
    # for param in (m, c, sigma, x, data):
    #     if not isinstance(param, (float, np.ndarray)):
    #         raise TypeError(f"Invalid input type to loglike: {type(param)}")
    
    model = my_model(params, coords)
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
    
    return -(0.5 / sigma**2) * ((residuals) ** 2)

# define a pytensor Op for our likelihood function
class LogLike(Op):
    def make_node(self, params, sigma, coords, data) -> Apply:
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

        # params = pt.as_tensor([r0, r1, a0, a1])
        inputs = [params, sigma, coords, data]
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
        params, sigma, coords, data = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        #params = [r0, r1, a0, a1]
        loglike_eval = my_loglike(params, sigma, coords, data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

#   Data
location_df, coordinate_df = JP_BFM._ready_DataFrames(dt.datetime(2016, 6, 1), dt.datetime(2020, 6, 1), resolution=10)


coords = coordinate_df.loc[:, ['r', 't', 'p', 'p_dyn']].to_numpy()
sigma = 10
data = location_df['within_bs'].to_numpy()

# create our Op
loglike_op = LogLike()

def custom_dist_loglike(data, params, sigma, coords):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(params, sigma, coords, data)

# # use PyMC to sampler from log-likelihood
# with pm.Model() as no_grad_model:
#     # uniform priors on m and c
#     r0 = pm.Uniform("r0", lower=20.0, upper=80.0)
#     r1 = pm.Uniform("r1", lower=-0.3, upper=-0.1)
#     a0 = pm.Uniform("a0", lower=0.0, upper=2.0)
#     a1 = pm.Uniform("a1", lower=0.0, upper=1.0)
    
#     #params = pt.as_tensor([r0, r1, a0, a1])
#     #breakpoint()
#     # use a CustomDist with a custom logp function
#     likelihood = pm.CustomDist(
#         "likelihood", r0, r1, a0, a1, sigma, coords.T, observed=data, logp=custom_dist_loglike, #signature='(),(),()->()'
#     )

# ip = no_grad_model.initial_point()

# with no_grad_model:
#     # Use custom number of draws to replace the HMC based defaults
#     idata_no_grad = pm.sample(3000, tune=1000)

# # plot the traces
# az.plot_trace(idata_no_grad);


with pm.Model() as potential_model:
    # uniform priors on m and c
    # r0 = pm.Uniform("r0", lower=0.0, upper=100.0)
    #r1 = pm.Uniform("r1", lower=-0.5, upper=0.25)
    # a0 = pm.Uniform("a0", lower=0.0, upper=5.0)
    # a1 = pm.Uniform("a1", lower=-10.0, upper=10.0)
    r0 = pm.InverseGamma("r0", mu=60, sigma=30)
    r1 = pm.Normal("r1", mu=-0.2, sigma=0.1)
    a0 = pm.InverseGamma("a0", mu=2.0, sigma=10.0)
    a1 = pm.Normal("a1", mu=0, sigma=0.5)
    
    #constraint = pm.Deterministic('constrant', a0 > -a1)
    #theta = pt.as_tensor([m,c])
    params = [r0, r1, a0, a1]

    # use a Potential instead of a CustomDist
    pm.Potential("likelihood", custom_dist_loglike(data, params, sigma, coords.T))

    idata_potential = pm.sample(tune=2000, draws=2000, chains=4)

# plot the traces
az.plot_trace(idata_potential);

figure = corner.corner(idata_potential, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,)

#   Sample a subset of all the MCMC models to calculate the MAE between the data and the model
#   This is used to compare different models to one another
stack = az.extract(idata_potential, num_samples=100)

fig, ax = plt.subplots()

model_info = JP_BFM.boundary_model_init('Shuelike')
# r, t, p, p_dyn = coords.T
# coordinate_df = pd.DataFrame({'r': r, 
#                               't': t,
#                               'p': p,
#                               'p_dyn': p_dyn})

all_models = []
mad_list = []
for i in range(100):
    
    a0, a1, r0, r1 = stack.a0.values[i], stack.a1.values[i], stack.r0.values[i], stack.r1.values[i]
    
    result_df = JP_BFM.find_ModelOrbitIntersections(coordinate_df, 
                                                    boundary = 'bs',
                                                    model = model_info['model'], 
                                                    params = [r0, r1, a0, a1])
    #ax.plot(coordinate_df.index, result_df['within_bs'], color='black', alpha=0.01, zorder=2)
    all_models.append(result_df['within_bs'].to_numpy())
    mad_list.append(np.mean(np.abs(location_df['within_bs'] - result_df['within_bs'])))
    
ax.plot(location_df.index, np.mean(all_models, 0), color='black', linewidth=1)
ax.plot(location_df.index, location_df['within_bs'], color='C0', linewidth=1, zorder=0)
    

fig, ax = plt.subplots()

model_coords = pd.DataFrame({'t': np.linspace((-3/4)*np.pi, (3/4)*np.pi, 1000),
                             'p': np.zeros(1000) + (1/2)*np.pi,
                             'p_dyn': np.zeros(1000) + 0.05})
for i in range(100):
    
    a0, a1, r0, r1 = stack.a0.values[i], stack.a1.values[i], stack.r0.values[i], stack.r1.values[i]
    
    
    model_coords['r'] = model_info['model'](parameters = (r0, r1, a0, a1), coordinates=model_coords.to_numpy().T)
    
    model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
        
    # model_x = model_coords['r'] * np.cos(model_coords['t'])
    # model_y = model_coords['r'] * np.sin(model_coords['t'])
    ax.plot(model_x, model_y, alpha=0.05, color='black')
    
    model_coords = model_coords.drop('r', axis = 'columns')

#   params as a list in the correct order
params = [stack.mean()[e].values for e in stack.mean()[['r0', 'r1', 'a0', 'a1']]]
model_coords['r'] = model_info['model'](parameters = params, coordinates = model_coords.to_numpy().T)
model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
ax.plot(model_x, model_y, color='C0', label='Mean')
model_coords = model_coords.drop('r', axis='columns')

#   params as a list in the correct order
params = [stack.median()[e].values for e in stack.median()[['r0', 'r1', 'a0', 'a1']]]
model_coords['r'] = model_info['model'](parameters = params, coordinates = model_coords.to_numpy().T)
model_x, model_y, model_z = BM.convert_SphericalSolarToCartesian(*model_coords[['r', 't', 'p']].to_numpy().T)
ax.plot(model_x, model_y, color='C1', label='Median')    
model_coords = model_coords.drop('r', axis='columns')

ax.legend()

ax.set(xlim = (-200,200), ylim=(-200,200))
