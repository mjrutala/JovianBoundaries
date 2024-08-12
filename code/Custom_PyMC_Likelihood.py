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

def my_model(r0, r1, a0, a1, coords):
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
                                                    params = [r0, r1, a0, a1])
    
    
    return result_df['within_bs'].to_numpy()


def my_loglike(r0, r1, a0, a1, sigma, coords, data):
    # We fail explicitly if inputs are not numerical types for the sake of this tutorial
    # As defined, my_loglike would actually work fine with PyTensor variables!
    
    # for param in (m, c, sigma, x, data):
    #     if not isinstance(param, (float, np.ndarray)):
    #         raise TypeError(f"Invalid input type to loglike: {type(param)}")
    model = my_model(r0, r1, a0, a1, coords)
    return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

# define a pytensor Op for our likelihood function
class LogLike(Op):
    def make_node(self, r0, r1, a0, a1, sigma, coords, data) -> Apply:
        # Convert inputs to tensor variables
        r0 = pt.as_tensor(r0)
        r1 = pt.as_tensor(r1)
        a0 = pt.as_tensor(a0)
        a1 = pt.as_tensor(a1)
        #params = pt.as_tensor_variable(params)
        sigma = pt.as_tensor(sigma)
        # x = pt.as_tensor(x)
        coords = pt.as_tensor_variable(coords)
        data = pt.as_tensor(data)

        inputs = [r0, r1, a0, a1, sigma, coords, data]
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
        r0, r1, a0, a1, sigma, coords, data = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = my_loglike(r0, r1, a0, a1, sigma, coords, data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

#   Data
location_df, coordinate_df = JP_BFM._ready_DataFrames(dt.datetime(2016, 6, 1), dt.datetime(2023, 6, 1), resolution=10)


coords = coordinate_df.loc[:, ['r', 't', 'p', 'p_dyn']].to_numpy()
sigma = 10
data = location_df['within_bs'].to_numpy()

# create our Op
loglike_op = LogLike()

def custom_dist_loglike(data, r0, r1, a0, a1, sigma, coords):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(r0, r1, a0, a1, sigma, coords, data)

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
    r0 = pm.Uniform("r0", lower=0.0, upper=100.0)
    r1 = pm.Uniform("r1", lower=-0.5, upper=0.25)
    a0 = pm.Uniform("a0", lower=0.0, upper=5.0)
    a1 = pm.Uniform("a1", lower=-10.0, upper=10.0)
    
    #theta = pt.as_tensor([m,c])

    # use a Potential instead of a CustomDist
    pm.Potential("likelihood", custom_dist_loglike(data, r0, r1, a0, a1, sigma, coords.T))

    idata_potential = pm.sample(tune=500, draws=1000, chains=10)

# plot the traces
az.plot_trace(idata_potential);

figure = corner.corner(idata_potential, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,)