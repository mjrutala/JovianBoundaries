#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:54:35 2024

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
import pickle
import pprint
import os

from pytensor.graph import Apply, Op
import xarray
from scipy.optimize import approx_fprime

from sklearn.metrics import confusion_matrix

import BoundaryForwardModeling as BFM
import BoundaryModels as BM
import JoyBoundaryCoords as JBC

import CrossingPreprocessingRoutines as preproc
import CrossingPostprocessingRoutines as postproc

# Load custom plotting style
try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

def HybridMCMC(boundary, model_name, 
               fraction,
               n_pressures, n_draws, n_chains, n_modes):
    "Shell to run HybridMCMC and save results"
    
    # Get the run start time for file-saving and make a corresponding dir
    now_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    savedir = '/Users/mrutala/projects/JupiterBoundaries/posteriors/{}/'.format(now_str)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    else:
        msg = "The path '{}' already exists-- and shouldn't! Returning..."
        print(msg.format(savedir))
        return
    
    # Dictionary of arguments to be passed
    args = {'boundary': boundary,                                               # Boundary param
            'model_name': model_name,                                           # Boundary param
            'spacecraft_to_use': ['Ulysses', 'Galileo', 'Cassini', 'Juno'],     # Data param
            'resolution': '10Min',                                              # Data param
            'fraction': fraction,                                               # Data param
            'n_pressures': n_pressures,                                         # MCMC param
            'n_draws': n_draws,                                                 # MCMC param
            'n_chains': n_chains,                                               # MCMC param 
            'n_modes': n_modes}                                                 # MCMC param 
    
    # Run the MCMC sampler
    posterior, model_dict = _HybridMCMC(**args)
    
    # Write to file
    stem = "{model_name}-{boundary}_".format(**args) + now_str
    
    # Inputs
    i_filepath = savedir + stem + "_input.txt"
    with open(i_filepath, 'w') as f:
        pprint.pprint(args, f)
        
    # Posterior
    p_filepath = savedir + stem + "_posterior.pkl"
    with open(p_filepath, 'wb') as f:
        pickle.dump(posterior, f)
    
    #   Plot a trace to check for convergence
    fig, axs = plt.subplots(nrows=len(posterior), ncols=2, figsize=(6.5, 2*len(posterior)))
    plt.subplots_adjust(bottom=0.025, left=0.05, top=0.975, right=0.95,
                       hspace=0.5, wspace=0.2)
    az.plot_trace(posterior, axes=axs, rug=True)
    plt.savefig(savedir + stem + "_traceplot.png", dpi=300)
    
    #   Plot a corner plot to investigate covariance
    figure = corner.corner(posterior, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,)
    figure.set_size_inches(len(posterior),len(posterior))
    plt.savefig(savedir + stem + "_cornerplot.png", dpi=300)
    
    plt.show()

    return posterior, model_dict

def _HybridMCMC(boundary, model_name,
                spacecraft_to_use, resolution, fraction, 
                n_pressures, n_draws, n_chains, n_modes):
    
    positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use)
    
    positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')
    
    # FOR TESTING, TO MAKE THE MCMC SAMPLER RUN FASTER
    positions_df = positions_df.sample(frac=fraction, axis='rows', replace=False)
    n = len(positions_df)
    
    from scipy.stats import skewnorm
    
    # For readability, pull out indiviudal variables to be passed to PyMC
    r = positions_df['r'].to_numpy('float64')
    t = positions_df['t'].to_numpy('float64')
    p = positions_df['p'].to_numpy('float64')
    
    r_lower = positions_df['r_lowerbound'].to_numpy('float64')
    r_upper = positions_df['r_upperbound'].to_numpy('float64')
    
    p_dyn_loc   = positions_df['p_dyn_loc'].to_numpy('float64')
    p_dyn_scale = positions_df['p_dyn_scale'].to_numpy('float64')
    p_dyn_alpha = positions_df['p_dyn_a'].to_numpy('float64')
    p_dyn_mu    = skewnorm.mean(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)
    p_dyn_sigma = skewnorm.std(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)

    test_pressure_dist = pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha)
    test_pressure_draws = pm.draw(test_pressure_dist, draws=n_pressures)
    # Hacky, but assume negative pressures correspond to small pressures
    test_pressure_draws[test_pressure_draws < 0] = np.min(test_pressure_draws[test_pressure_draws > 0])
    if len(np.shape(test_pressure_draws)) < 2:
        test_pressure_draws = [test_pressure_draws]
    
    # Model and MCMC params
    posterior_list = []    
    model_dict = BM.init(model_name)
    
    print("Testing {} pressure draws".format(len(test_pressure_draws)))
    for pressure_draw in test_pressure_draws:
        
        with pm.Model() as test_potential:
    
            p_dyn_obs = pressure_draw
            
            #   N.B.: 'param_dict' must keep the same name, as it is used in code string evaluation below
            param_dict = {}
            for param_name in model_dict['param_distributions'].keys():
                param_dist = model_dict['param_distributions'][param_name]
                
                param = param_dist(param_name, **model_dict['param_descriptions'][param_name])
                
                param_dict[param_name] = param
            
            # Assign variable names to a list of tracking in the sampler
            param_names = list(param_dict.keys())
            tracked_var_names = []
            for pname in param_names:
                tracked_var_names.append(pname)
            
            params = list(param_dict.values())
            
            # The predicted mean location of the boundary is a function of pressure
            mu0 = pm.Deterministic("mu0", model_dict['model'](params, [t, p, p_dyn_obs]))
            
            # This section adds a penalty for creating a closed boundary (i.e., no open field lines)
            if 'Shue' in model_name:
                flare_pred =  pm.Deterministic("flare_pred", model_dict['model'](params, [t, p, p_dyn_obs], return_a_f=True))
                closed_penalty = pm.Potential("closed_penalty", pm.math.switch(flare_pred < 0.5, -np.inf, 0))
            
            # The observed mean location will vary based on internal drivers
            # In a spherical coordinate system, the variation in r must increase with theta
            r_obs = pm.Uniform("r_obs", lower = r_lower, upper = r_upper)
            
            # Make sigma increase with t
            sigma_b = pm.HalfNormal("sigma_b", 1.0)
            sigma_m = pm.HalfNormal("sigma_m", 1.0)
            sigma_fn = pm.Deterministic("sigma_fn", sigma_b + sigma_m * mu0)
            tracked_var_names.extend(["sigma_b", "sigma_m"])

            # likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu = mu_pred, sigma = sigma_fn), value=r_obs))
            # sigma_dist = pm.Normal("sigma_dist", 0, 1)
            # likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu = mu_pred + (sigma_dist * sigma_fn)), value=r_obs))
            
            # Optionally, handle multiple modes           
            if n_modes == 1:
                likelihood = pm.Normal("likelihood", mu = mu0 - r_obs, sigma = sigma_fn, observed = np.zeros(n))
                
            elif n_modes == 2:
                buffer1 = pm.HalfNormal("buffer1", sigma=5)
                tracked_var_names.append("buffer1")
                mu1 = pm.Deterministic("mu1", mu0 - buffer1)
                
                # Weights for multi-modal fits
                w = pm.Dirichlet("w", np.ones(2))
                tracked_var_names.append("w")
                
                dists = [pm.Normal.dist(mu = mu0, sigma = sigma_fn),
                         pm.Normal.dist(mu = mu1, sigma = sigma_fn)]
                # likelihood = pm.Mixture("likelihood", w=w, comp_dists = dists,
                #                         observed = np.zeros(n))
                x = pm.Mixture.dist(w=w, comp_dists = dists)
                likelihood = pm.Potential("likelihood", pm.logp(x, value = r_obs))
            
        with test_potential:
            idata = pm.sample(tune=512, draws=n_draws, chains=n_chains, cores=3,
                              var_names = tracked_var_names,
                              target_accept=0.90)
                              # init = 'adapt_diag+jitter') # prevents 'jitter', which might move points init vals around too much here
            
        posterior = idata.posterior
        posterior_list.append(posterior)
    
    posterior = xarray.concat(posterior_list, dim = 'chain')
    
    # Separate out the Dirichlet weights if multi-modal
    if n_modes > 1:
        for mode_number in posterior['w_dim_0'].values:
            posterior['w{}'.format(mode_number)] = (("chain", "draw"), posterior['w'].values[:,:,mode_number])
        posterior = posterior.drop_vars('w')
        
    return posterior, model_dict


def _HybridMCMC_outdated(boundary, spacecraft_to_use, resolution,
               model_name,
               fraction, n_pressures, n_draws, n_chains):
    
    positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use)
    n = len(positions_df)
    
    positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')
    
    # FOR TESTING, TO MAKE THE MCMC SAMPLER RUN FASTER
    positions_df = positions_df.sample(frac=0.25, axis='rows', replace=False)
    
    from scipy.stats import skewnorm
    
    # For readability, pull out indiviudal variables to be passed to PyMC
    r = positions_df['r'].to_numpy('float64')
    t = positions_df['t'].to_numpy('float64')
    p = positions_df['p'].to_numpy('float64')
    
    r_lower = positions_df['r_lowerbound'].to_numpy('float64')
    r_upper = positions_df['r_upperbound'].to_numpy('float64')
    
    p_dyn_loc   = positions_df['p_dyn_loc'].to_numpy('float64')
    p_dyn_scale = positions_df['p_dyn_scale'].to_numpy('float64')
    p_dyn_alpha = positions_df['p_dyn_a'].to_numpy('float64')
    p_dyn_mu    = skewnorm.mean(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)
    p_dyn_sigma = skewnorm.std(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)

    test_pressure_dist = pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha)
    test_pressure_draws = pm.draw(test_pressure_dist, draws=n_pressures)
    # Hacky, but assume negative pressures correspond to small pressures
    test_pressure_draws[test_pressure_draws < 0] = np.min(test_pressure_draws[test_pressure_draws > 0])
    if len(np.shape(test_pressure_draws)) < 2:
        test_pressure_draws = [test_pressure_draws]
    
    # Model and MCMC params
    posterior_list = []    
    model_dict = BM.init(model_name)
    
    model_sigma_value = 5
    # mu_sigma_values = (10, 5)
    
    print("Testing {} pressure draws".format(len(test_pressure_draws)))
    for pressure_draw in test_pressure_draws:
        
        with pm.Model() as test_potential:
    
            p_dyn_obs = pressure_draw
            
            #   N.B.: 'param_dict' must keep the same name, as it is used in code string evaluation below
            param_dict = {}
            for param_name in model_dict['param_distributions'].keys():
                param_dist = model_dict['param_distributions'][param_name]
                
                param = param_dist(param_name, **model_dict['param_descriptions'][param_name])
                
                # param_mu = param_dist(param_name+'_mu', **model_dict['param_descriptions'][param_name])
                # param_sigma = pm.HalfNormal(param_name+'_sigma', 1)
                # param = pm.Normal(param_name, mu = param_mu, sigma = param_sigma)
                
                param_dict[param_name] = param
            
            # Assign variable names to a list of tracking in the sampler
            param_names = list(param_dict.keys())
            tracked_var_names = []
            for pname in param_names:
                tracked_var_names.append(pname)
            
            params = list(param_dict.values())
            #params = [r0, r1, a0, a1]
            
            # The predicted mean location of the boundary is a function of pressure
            mu_pred = pm.Deterministic("mu_pred", model_dict['model'](params, [t, p, p_dyn_obs]))
            
            # This section adds a penalty for creating a closed boundary (i.e., no open field lines)
            if 'Shue' in model_name:
                flare_pred =  pm.Deterministic("flare_pred", model_dict['model'](params, [t, p, p_dyn_obs], return_a_f=True))
                closed_penalty = pm.Potential("closed_penalty", pm.math.switch(flare_pred < 0.5, -np.inf, 0))
            
            # The observed mean location will vary based on internal drivers
            # In a spherical coordinate system, the variation in r must increase with theta
            
            # mu_obs_sigma_coeff = pm.HalfNormal("mu_obs_sigma_coeff", sigma = 5)
            # mu_obs_sigma_const = pm.HalfNormal("mu_obs_sigma_const", sigma = 10)
            # mu_obs_sigma = pm.Deterministic("mu_obs_sigma", mu_obs_sigma_const + np.sin(t/2)*mu_obs_sigma_coeff)
            # mu_obs = pm.Normal('mu_obs', mu=mu_pred, sigma=mu_obs_sigma)
            
            # r_b = pm.Deterministic("r_b", r0 * (p_dyn_obs) ** r1)
            # r_b = pm.Deterministic("r_b", (r0 + r2*np.cos(p)**2 + r3*np.sin(p)*np.sin(t))*((p_dyn_obs)**r1))
            # a_b = pm.Deterministic("a_b", a0 + a1 * (p_dyn_obs))
            
            r_obs = pm.Uniform("r_obs", lower = r_lower, upper = r_upper)
            
            # Make sigma increase with t
            # sigma_b = pm.HalfNormal("sigma_b", sigma=10)
            # sigma_m = pm.HalfNormal("sigma_m", sigma=5)
            # sigma = pm.HalfNormal("sigma", sigma = sigma_m * np.tan(t/2)**2 + sigma_b)
            
            # # Track sigma in the sampler as well
            # tracked_var_names.extend(['sigma_m', 'sigma_b'])
            
            # sigma = pm.HalfNormal("sigma", sigma=10)
            # sigma_t = pm.HalfNormal("sigma_t", sigma = sigma * (2/(1 + np.cos(t))))
            
            # fractional_sigma = 5
            
            # if 'r0' in param_dict.keys():
            #     sigma_param_dict = {}
            #     for key in param_dict.keys():
            #         if key in ['r0', 'r2', 'r3', 'r4', 'A0']:
            #             sigma_param_dict['sigma_'+key] = pm.HalfNormal('sigma_'+key, fractional_sigma)
            #             tracked_var_names.append('sigma_' + key)
            #         # elif key in ['A0', 'B0', 'C0']:
            #         #     sigma_param_dict['sigma_'+key] = pm.HalfNormal('sigma_'+key, fractional_sigma/120)
            #         #     tracked_var_names.append('sigma_' + key)
            #         else:
            #             sigma_param_dict['sigma_'+key] = param_dict[key]
        
            #     sigma_params = list(sigma_param_dict.values())
                
            #     # sigma_total = pm.HalfNormal("sigma_total", sigma = sigma_constant)
            #     sigma_fn = pm.Deterministic("sigma_fn", model_dict['model'](sigma_params, [t, p, p_dyn_obs]))
            # else:
            #     sigma_fn = pm.HalfNormal("sigma_fn", fractional_sigma)
            #     tracked_var_names.append("sigma_fn")
            
            sigma_b = pm.HalfNormal("sigma_b", 1.0)
            sigma_m = pm.HalfNormal("sigma_m", 1.0)
            sigma_fn = pm.Deterministic("sigma_fn", sigma_b + sigma_m * mu_pred)
            tracked_var_names.extend(["sigma_b", "sigma_m"])
            
            # sigma_total = pm.HalfNormal("sigma_total", sigma = 5)
            
            # tracked_var_names.extend(['sigma_dynamic'])
            
            # likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu=mu_pred, sigma=sigma_total), value=r_obs))
            
            # var = pm.Normal("var", 0, 1)
            
            # likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu = mu_pred, sigma = sigma_fn), value=r_obs))
            likelihood = pm.Normal("likelihood", mu = mu_pred - r_obs, sigma = sigma_fn, observed = np.zeros(len(r)))
            
            # sigma_dist = pm.Normal("sigma_dist", 0, 1)
            # likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu = mu_pred + (sigma_dist * sigma_fn)), value=r_obs))
    
        with test_potential:
            idata = pm.sample(tune=512, draws=n_draws, chains=n_chains, cores=3,
                              var_names = tracked_var_names,
                              target_accept=0.80)
                              # init = 'adapt_diag+jitter') # prevents 'jitter', which might move points init vals around too much here
        
        posterior = idata.posterior
        posterior_list.append(posterior)
    
    posterior = xarray.concat(posterior_list, dim = 'chain')
        
    
    # posterior = posterior.drop_vars(['mu_obs', 'mu_obs_sigma', 'mu_pred', 'r_obs'])
    #   Plot a trace to check for convergence
    az.plot_trace(posterior)
    #   Plot a corner plot to investigate covariance
    figure = corner.corner(posterior, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,)
    figure.set_size_inches(8,8)
    plt.show()
    
    # Save the posterior
    posterior_filename = "boundary-{}_model-{}_sc-{}_res-{}_pdyndraws-{}-draws-{}_chains-{}"
    posterior_filename = posterior_filename.format(boundary, model_name, 
                                                   '-'.join(spacecraft_to_use), 
                                                   resolution, 
                                                   n_pressures, 
                                                   n_draws, n_chains)
    posterior_filepath = '/Users/mrutala/projects/JupiterBoundaries/posteriors/' + posterior_filename + '.pkl'
    with open(posterior_filepath, 'wb') as f:
        pickle.dump(posterior, f)
        
    return posterior, model_dict
