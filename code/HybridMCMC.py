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
import pymc.sampling.jax as pmjax
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
               n_subsets, n_draws, n_chains, n_modes):
    "Shell to run HybridMCMC and save results"
    
    # Get the run start time for file-saving and make a corresponding dir
    now_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    savedir = '/Users/mrutala/projects/JupiterBoundaries/posteriors/{}/'.format(now_str)
    if not os.path.exists(savedir):
        mkdir_cmd = os.makedirs # Save command for later: no empty dirs
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
            'n_subsets': n_subsets,                                         # MCMC param
            'n_draws': n_draws,                                                 # MCMC param
            'n_chains': n_chains,                                               # MCMC param 
            'n_modes': n_modes}                                                 # MCMC param 
    
    # Run the MCMC sampler
    posterior, model_dict = _HybridMCMC(**args)
    
    # Make the directory and write to file
    mkdir_cmd(savedir)
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
                n_subsets, n_draws, n_chains, n_modes):
    from scipy.stats import skewnorm
    
    positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use,
                                                 delta_around_crossing = dt.timedelta(hours=5),
                                                 other_fraction = 0.04) # Roughly balanced
    
    # Limit this further? by x, y, z?
    # positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')
    positions_df = positions_df.query('(-250 < x < 250) & (rho > 30 | x > 0) & ((r_upperbound - r_lowerbound) < 1e3)')
        
    J02_original_params = JBC.get_JoyParameters()
    
    # Model and MCMC params
    posterior_list = []    
    model_dict = BM.init(model_name)
    
    # Resample the data and pressures n_subset times
    data_draws = [positions_df.sample(frac=fraction, axis='rows', replace=True) for i in range(n_subsets)]
    pressure_draws = []
    for i in range(n_subsets):
        
        data_draw = data_draws[i]
        data_draw = data_draw.sort_index()
        
        # Sample the pressure distribution
        p_dyn_loc   = data_draw['p_dyn_loc'].to_numpy('float64')
        p_dyn_scale = data_draw['p_dyn_scale'].to_numpy('float64')
        p_dyn_alpha = data_draw['p_dyn_a'].to_numpy('float64')
        p_dyn_mu    = skewnorm.mean(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)
        p_dyn_sigma = skewnorm.std(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)
        
        pressure_dist = pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha)
        pressure_draw = pm.draw(pressure_dist, draws=1)
        # Hacky, but assume negative pressures correspond to small pressures
        pressure_draw[pressure_draw < 0] = np.min(pressure_draw[pressure_draw > 0])
        data_draw['pressure_draw'] = pressure_draw
        
        data_draws[i] = data_draw
        
    # Print a size warning
    print("Testing {} randomly sampled subsets with replacement".format(n_subsets))

    for data_draw in data_draws:
        
        n = len(data_draw)
        print("n={} samples testing-- This should not exceed ~4000.".format(n))
        
        with pm.Model() as test_potential:
            tracked_var_names = []
               
            param_dict = {}
            #   N.B.: 'param_dict' must keep the same name, as it is used in code string evaluation below
            for param_name in model_dict['param_distributions'].keys():
                param_dist = model_dict['param_distributions'][param_name]
                
                param = param_dist(param_name, **model_dict['param_descriptions'][param_name])
                
                param_dict[param_name] = param
                
            # Assign variable names to a list of tracking in the sampler
            params = list(param_dict.values())
            tracked_var_names.extend(list(param_dict.keys()))
            
            # This section adds a penalty for creating a closed boundary
            if 'Shue' in model_name:
                flare_pred =  pm.Deterministic("flare_pred", model_dict['model'](list(param_dict.values()), data_draw[['t', 'p', 'pressure_draw']].to_numpy('float64').T, return_a_f=True))
                closed_penalty = pm.Potential("closed_penalty", pm.math.switch(flare_pred < 0.5, -np.inf, 0))
            
            r_J02_original = BM.Joylike_r1fixed(J02_original_params, data_draw[['t', 'p', 'pressure_draw']].to_numpy('float64').T)
            
            if 'log' not in model_name:
                mu0 = pm.Deterministic("mu0", model_dict['model'](list(param_dict.values()), data_draw[['t', 'p', 'pressure_draw']].to_numpy('float64').T))
                
                sigma_b = pm.HalfNormal("sigma_b", 4)
                sigma_m = pm.HalfNormal("sigma_m", 0.05)
                sigma = sigma_b + sigma_m * mu0
                tracked_var_names.extend(["sigma_b", "sigma_m"])
                
                # r_obs = pm.Uniform("r_obs", 
                #                    lower = data_draw['r_lowerbound'].to_numpy('float64'), 
                #                    upper = data_draw['r_upperbound'].to_numpy('float64'))
                r_obs = pm.TruncatedNormal("r_obs", 
                                           mu = r_J02_original, sigma=10,
                                           lower = data_draw['r_lowerbound'].to_numpy('float64'), 
                                           upper = data_draw['r_upperbound'].to_numpy('float64'))
                
                m, s, r = mu0, sigma, r_obs
                
            else:
                breakpoint()
                
                
            # Optionally, handle multiple modes           
            if n_modes == 1:
                dist = pm.Normal.dist(mu = m, sigma = s)
                likelihood = pm.Normal("likelihood", pm.logp(dist, value = r))
                
            elif n_modes == 2:
                #buffer_b = pm.HalfNormal("buffer_b", sigma=5)
                buffer_b = pm.TruncatedNormal("buffer_b", mu=20, sigma=5, lower=0)
                buffer_m = pm.TruncatedNormal("buffer_m", mu=0.1, sigma=0.1, lower=0)
                tracked_var_names.extend(["buffer_b", "buffer_m"])
                # log_buffer = pm.HalfNormal("log_buffer", 20/np.exp(log_mu0))
                
                buffer1 = pm.Deterministic("buffer1", buffer_b + buffer_m * mu0)
                mu1 = pm.Deterministic("mu1", mu0 + buffer1)
                # log_mu1 = pm.Deterministic("log_mu1", log_mu0 - log_buffer)
                
                # Weights for multi-modal fits
                w = pm.Dirichlet("w", [0.5, 0.5])
                tracked_var_names.append("w")
                
                dists = [pm.Normal.dist(mu = mu0, sigma = 2),
                         pm.Normal.dist(mu = mu1, sigma = 2)]
                # likelihood = pm.Mixture("likelihood", w=w, comp_dists = dists,
                #                         observed = np.zeros(n))
                x = pm.Mixture.dist(w=w, comp_dists = dists)
                likelihood = pm.Potential("likelihood", pm.logp(x, value = r))
        
        # Prior predictives-- check that the inputs are reasonable
        with test_potential:
            prior_checks = pm.sample_prior_predictive(samples=10000)
        prior = prior_checks.prior
        
        fig, axs = plt.subplots(nrows=3)
        # Prior predictive plot 1: Does mu0 lie between upperbound and lowerbound?
        axs[0].fill_between(np.arange(n), data_draw['r_lowerbound'].values, data_draw['r_upperbound'].values,
                            color = 'green', alpha = 0.2)
        mu0_levels = np.percentile(prior['mu0'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
        axs[0].scatter(np.arange(n), mu0_levels[4], s=2, color='black')
        
        if 'mu1' in list(prior.variables):
            mu1_levels = np.percentile(prior['mu1'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
            axs[0].scatter(np.arange(n), mu0_levels[4], s=2, color='black', marker='x')
        
        # Plot 2: overall histograms
        bins = np.arange(0, 1000, 10)
        h = axs[1].hist(prior['mu0'].values.flatten(), bins,
                        label = 'mu0 (Model Outputs)',
                        histtype = 'step', lw = 2, density = True)
        h = axs[1].hist(prior['r_obs'].values.flatten(), bins,
                        label = 'r_obs (Possible Boundary Locations)',
                        histtype = 'step', lw = 2, density = True)
        axs[1].legend()
        # Plot 3: sigma-normalized residuals
        residuals = prior['mu0'].values[0] - prior['r_obs'].values[0]
        norm_mean_residuals = np.mean(residuals, axis=0)/np.std(prior['r_obs'].values[0], axis=0)
        axs[2].plot(norm_mean_residuals,
                    label = 'Column-Mean Residuals, Normalized by the standard deviation of r_obs',
                    color = 'C2')
        axs[2].legend()
        plt.show()
        # breakpoint()
        
        # with test_potential:
        #     mean_fit = pm.fit()
        # approx_idata = mean_fit.sample(2000)
        # approx_posterior = approx_idata.posterior
        # vars_names_to_drop = list(set(list(approx_posterior.variables)) - set(tracked_var_names))
        # approx_posterior = approx_posterior.drop_vars(var_names_to_drop)
            
        # breakpoint()
        
        with test_potential:
            # 0.9 worked well for Joy+ models
            # 0.95 works well for Shuelike + asymmetries (BS, MP)
            step = pm.NUTS(max_treedepth=12, target_accept=0.90)
            idata = pm.sample(tune=2000, draws=n_draws, chains=n_chains, cores=3,
                              var_names = tracked_var_names,
                              # init="advi")
                              step = step)
                              # init = 'adapt_diag+jitter') # prevents 'jitter', which might move points init vals around too much here
            
            # idata = pmjax.sample_numpyro_nuts(tune=2048, draws=n_draws, chains=n_chains,
            #                                   var_names = tracked_var_names,
            #                                   target_accept = 0.95)
            
            # step = pm.NUTS(max_treedepth=12, target_accept=0.95)
            # idata = pm.sample(tune=1024, draws=n_draws, chains=n_chains, cores=3,
            #                   var_names = tracked_var_names,
            #                   # step = step,
            #                   nuts_sampler="blackjax")
            #                   # init = 'adapt_diag+jitter')
        posterior = idata.posterior
        posterior_list.append(posterior)
    
    posterior = xarray.concat(posterior_list, dim = 'chain')
    
    # Separate out the Dirichlet weights if multi-modal
    if n_modes > 1:
        for mode_number in posterior['w_dim_0'].values:
            posterior['w{}'.format(mode_number)] = (("chain", "draw"), posterior['w'].values[:,:,mode_number])
        posterior = posterior.drop_vars('w')
        
    return posterior, model_dict