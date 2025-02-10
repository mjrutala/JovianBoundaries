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

def quickrun():
    
    
    # res = HybridMCMC('MP', 'Shuelike', 
    #                   0.20, 1, 1000, 4, 1, (0.975*np.pi, 74))
    
    res = HybridMCMC('MP', 'ShuelikeAsymmetric_AsPerturbation_2', 
                      0.20, 1, 1000, 4, 1, (0.975*np.pi, 74))
    
    res = HybridMCMC('MP', 'Shuelike', 
                      0.20, 1, 1000, 4, 2, (0.975*np.pi, 74))
    
    res = HybridMCMC('MP', 'ShuelikeAsymmetric_AsPerturbation_2', 
                      0.20, 1, 1000, 4, 2, (0.975*np.pi, 74))
    
    
    # res = HybridMCMC('BS', 'Shuelike_r1fixed', 
    #                   0.15, 1, 1000, 4, 1, (0.975*np.pi, 130))
    
    # res = HybridMCMC('BS', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2', 
    #                   0.15, 1, 1000, 4, 1, (0.975*np.pi, 130))
    
    # res = HybridMCMC('BS', 'Shuelike_r1fixed', 
    #                   0.15, 1, 1000, 4, 2, (0.975*np.pi, 130))
    
    # res = HybridMCMC('BS', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2', 
    #                   0.15, 1, 1000, 4, 2, (0.975*np.pi, 130))
    
    
    
    
    # res = HybridMCMC('MP', 'Shuelike', 
    #                  0.20, 1, 1000, 4, 1, (0.975*np.pi, 74))
    
    # res = HybridMCMC('BS', 'Shuelike', 
    #                  0.20, 1, 1000, 4, 1, (0.975*np.pi, 130))
    
    # res = HybridMCMC('MP', 'ShuelikeAsymmetric_AsPerturbation_2', 
    #                  0.20, 1, 1000, 4, 1, (0.975*np.pi, 74))    
    
    # res = HybridMCMC('BS', 'ShuelikeAsymmetric_AsPerturbation_2', 
    #                  0.20, 1, 1000, 4, 1, (0.975*np.pi, 130))
    


    return
    
    

def HybridMCMC(boundary, model_name, 
               fraction,
               n_subsets, n_draws, n_chains, n_modes,
               rho_limit = False,
               advi_mode = False):
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
            'spacecraft_to_use': ['Ulysses', 'Galileo', 'Cassini', 'Juno'],     # Data param 'Cassini', 
            'resolution': '10Min',                                              # Data param
            'fraction': fraction,                                               # Data param
            'n_subsets': n_subsets,                                             # MCMC param
            'n_draws': n_draws,                                                 # MCMC param
            'n_chains': n_chains,                                               # MCMC param 
            'n_modes': n_modes,                                                 # MCMC param 
            'rho_limit': rho_limit,
            'advi_mode': advi_mode}                                                 
    
    # Run the MCMC sampler
    posterior_for_plot, model_dict, idata = _HybridMCMC(**args)
    
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
        pickle.dump(posterior_for_plot, f)
        
    # Inference Data
    id_filepath = savedir + stem + "_inferencedata.nc"
    az.to_netcdf(idata, id_filepath)
    
    #   Plot a trace to check for convergence
    fig, axs = plt.subplots(nrows=len(posterior_for_plot), ncols=2, figsize=(6.5, 2*len(posterior_for_plot)))
    plt.subplots_adjust(bottom=0.025, left=0.05, top=0.975, right=0.95,
                       hspace=0.5, wspace=0.2)
    az.plot_trace(posterior_for_plot, axes=axs, rug=True)
    plt.savefig(savedir + stem + "_traceplot.png", dpi=300)
    
    #   Plot a corner plot to investigate covariance
    figure = corner.corner(posterior_for_plot, 
                           quantiles=[0.1, 0.5, 0.9],
                           show_titles=True,)
    figure.set_size_inches(len(posterior_for_plot),len(posterior_for_plot))
    plt.savefig(savedir + stem + "_cornerplot.png", dpi=300)
    
    plt.show()

    return posterior_for_plot, model_dict, idata

def _HybridMCMC(boundary, model_name,
                spacecraft_to_use, resolution, fraction, 
                n_subsets, n_draws, n_chains, n_modes,
                rho_limit = False, 
                advi_mode = False):
    from scipy.stats import skewnorm
    
    # +/-10 hours and 8% is roughly balanced
    # +/-10 hours and 0.8% favors crossings
    # positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use,
    #                                              delta_around_crossing = dt.timedelta(hours=10),
    #                                              other_fraction = 0.008) 
    # positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use,
    #                                              delta_around_crossing = dt.timedelta(hours=15), # ~5 RJ covered in +/-10 hours, typically
    #                                              other_fraction = 0.07) # 15% is 4x bigger than +/-10 hours
    
    # Keep positions after applying this query
    exclusion_query = '(x > -600) & ((r_upperbound - r_lowerbound) < 1e4)'
    # Get positions
    positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use,
                                                 delta_around_crossing = dt.timedelta(hours=4), # ~5 RJ covered in +/-10 hours, typically
                                                 other_fraction = 0.03, # 15% is 4x bigger than +/-10 hours
                                                 exclusion_query=exclusion_query)
    
    # Limit this further? by x, y, z?
    # positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')
    # positions_df = positions_df.query('(x > -300) & (rho > 30 | x > 0) & ((r_upperbound - r_lowerbound) < 5e3)')
    
    # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # positions_df = positions_df.query('(x > -600) & ((r_upperbound - r_lowerbound) < 1e4)')

    # # Add deep tail phantom points that force the boundary to stay open?
    # phantom_ell, phantom_rho = -1000, 25
    # phantom_fraction = 0.20
    # phantom_number = int(phantom_fraction * len(positions_df))
    # phantom_rpl = [np.zeros(phantom_number) + phantom_rho,
    #                np.linspace(0, 2*np.pi, phantom_number),
    #                np.zeros(phantom_number) + phantom_ell]
    
    # phantom_df = pd.DataFrame(columns = positions_df.columns, index = np.arange(phantom_number))
    # phantom_df.loc[:, ['rho', 'phi', 'ell']] = np.array(phantom_rpl).T
    # phantom_df.loc[:, ['r', 't', 'p']] = BM.convert_CylindricalSolarToSphericalSolar(*phantom_rpl).T
    # phantom_df.loc[:, ['x', 'y', 'z']] = BM.convert_SphericalSolarToCartesian(*phantom_df.loc[:, ['r', 't', 'p']].to_numpy('float64').T).T
    # phantom_df.loc[:, 'region'] = 'MS'
    # phantom_df.loc[:, 'region_num'] = 2
    # phantom_df.loc[:, ['SW', 'SH', 'MS', 'UN']] = [0, 0, 1, 0]
    # phantom_df.loc[:, 'spacecraft'] = 'phantom'
    
    # bsl = postproc.find_BoundarySurfaceLimits(positions_df, boundary, spacecraft_to_use)
    # phantom_df = postproc.add_Bounds(phantom_df, boundary, bsl['upperbound'], bsl['lowerbound'])
    
    
    # # phantom_df.loc[:, 'r_lowerbound'] = phantom_df.loc[:, 'r']
    # breakpoint()
    
    J02_original_params = JBC.get_JoyParameters()
    
    # Model and MCMC params
    idata_list = []    
    model_dict = BM.init(model_name)
    
    # Resample the data and pressures n_subset times
    data_draws = [positions_df.sample(frac=fraction, axis='rows', replace=False) for i in range(n_subsets)] # replace=True?
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
        
        pressure_dist = pm.Truncated.dist(pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha),
                                          lower = 0)
        pressure_draw = pm.draw(pressure_dist, draws=1)
        data_draw['p_dyn_μ'] = p_dyn_mu
        data_draw['p_dyn_σ'] = p_dyn_sigma
        data_draw['p_dyn_α'] = p_dyn_alpha
        
        data_draw['pressure_draw'] = pressure_draw
        
        data_draws[i] = data_draw
    
    # Print a size warning
    print("Testing {} randomly sampled subsets with replacement".format(n_subsets))
    
    for data_draw in data_draws:
        
        n = len(data_draw)
        print("n={} samples testing-- This should not exceed ~4000.".format(n))
        
        with pm.Model() as test_potential:
            
            # USE THIS FOR SINGLE PRESSURE DISTRIBUTION
            
            # p_dyn_dist = pm.SkewNormal("p_dyn_dist", mu = data_draw['p_dyn_μ'].to_numpy('float64'), 
            #                                          sigma = data_draw['p_dyn_σ'].to_numpy('float64'), 
            #                                          alpha = data_draw['p_dyn_α'].to_numpy('float64'))
            
            p_dyn_μ_latent = pm.HalfNormal("p_dyn_μ_latent", np.mean(data_draw['p_dyn_μ']))
            p_dyn_σ_latent = pm.HalfNormal("p_dyn_σ_latent", np.mean(data_draw['p_dyn_σ']))
            p_dyn_α_latent = pm.Normal("p_dyn_α_latent", mu = 0, sigma = np.mean(np.abs(data_draw['p_dyn_α'])))
            p_dyn_draw = pm.SkewNormal("p_dyn_draw", 
                                       mu = p_dyn_μ_latent, 
                                       sigma = p_dyn_σ_latent,
                                       alpha = p_dyn_α_latent,
                                       observed = data_draw['pressure_draw'].to_numpy('float64'))
            
            data_draw_coords = [data_draw['t'].to_numpy('float64'), 
                                data_draw['p'].to_numpy('float64'),
                                p_dyn_draw]
            
            # This gives "NotImplementedError: LogCDF method not implemented for skewnormal_rv{"(),(),()->()"}"
            # p_dyn_skew_norm = pm.SkewNormal.dist(mu = data_draw['p_dyn_μ'].to_numpy('float64'), 
            #                                      sigma = data_draw['p_dyn_σ'].to_numpy('float64'), 
            #                                      alpha = data_draw['p_dyn_α'].to_numpy('float64'))
            # p_dyn_dist = pm.Truncated("p_dyn_dist", p_dyn_skew_norm, lower = 0)
            
            # # USE THIS FOR SEPARATE PRESSURE DRAWS
            # data_draw_coords = data_draw[['t', 'p', 'pressure_draw']].to_numpy('float64').T
            
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
            
            r_J02_original = BM.Joylike_r1fixed(J02_original_params, data_draw_coords)
            
            if 'log' not in model_name:
                mu0 = pm.Deterministic("mu0", model_dict['model'](list(param_dict.values()), data_draw_coords))
                
                sigma_b = pm.HalfNormal("sigma_b", 1.0)
                sigma_m = pm.HalfNormal("sigma_m", 0.01)
                sigma = sigma_b + sigma_m * mu0
                tracked_var_names.extend(["sigma_b", "sigma_m"])
                
                lb =  data_draw['r_lowerbound'].to_numpy('float64')
                ub =  data_draw['r_upperbound'].to_numpy('float64')
                r_obs = pm.Uniform("r_obs", 
                                    lower = data_draw['r_lowerbound'].to_numpy('float64'), 
                                    upper = data_draw['r_upperbound'].to_numpy('float64'))
                # r_obs = pm.TruncatedNormal("r_obs", 
                #                             mu = r_J02_original, sigma=16,
                #                             lower = lb, 
                #                             upper = ub)
                
                # m, s, r = mu0, sigma, r_obs
                
            else:
                breakpoint()
                
            # # This section adds a penalty for creating a closed boundary
            # if 'Shue' in model_name:
            #     flare_pred =  pm.Deterministic("flare_pred", model_dict['model'](list(param_dict.values()), data_draw_coords, return_a_f=True))
            #     closed_penalty = pm.Potential("closed_penalty", pm.math.switch(flare_pred > 0.5, 0, -((flare_pred - 0.5)/0.005)**2))
                
            #     # large_r2_penalty = pm.Potential("large_r2_penalty", pm.math.switch(-param_dict['r2'] < param_dict['r0'], 0, -((param_dict['r0'] + param_dict['r2'])/1)**4))
            #     # large_r2_penalty = pm.Potential("large_r2_penalty", pm.math.switch(np.array(False), 0, -((param_dict['r2']/0.1)**2)))
            #     # large_r3_penalty = pm.Potential("large_r3_penalty", pm.math.switch(np.array(False), 0, -((param_dict['r3']/2)**2)))
            #     # large_r4_penalty = pm.Potential("large_r4_penalty", pm.math.switch(np.array(False), 0, -((param_dict['r4']/2)**2)))
                
            # # #     # r_pred = pm.Deterministic("r_pred",  model_dict['model'](list(param_dict.values()), data_draw[['t', 'p', 'pressure_draw']].to_numpy('float64').T, return_r_ss=True))
            # # #     # neg_penalty = pm.Potential("neg_penalty", pm.math.switch(r_pred >= param_dict['r0'], 0, -((r_pred - param_dict['r0'])/3)**2))
            # #     # r2_penalty = pm.Potential("r2_penalty", pm.math.switch(-1 * param_dict['r2'] <= param_dict['r0']/2, 0, -((-1 * param_dict['r2'] - param_dict['r0'])/0.5)**2))
            
            # This section adds an explicit penalty for lying outside the bounds
            too_low_penalty = pm.Potential("too_low_penalty", pm.math.switch(mu0 > lb, 0, -((mu0 - lb)/0.1)**2))
            too_high_penalty = pm.Potential("too_high_penalty", pm.math.switch(mu0 < ub, 0, -((mu0 - ub)/0.1)**2))
            # negative_penalty = pm.Potential('neg', pm.math.switch(mu0 > 0, 0, -np.inf))
            
            # Force the boundary open
            # if boundary == 'MP':
            if rho_limit != False:
                om_theta_limit, om_rho_limit = rho_limit
                om_n = len(pressure_draw)
                om_rng = np.random.default_rng()
                om_coords = [om_theta_limit,  # r approx 300
                             om_rng.uniform(0, 2*np.pi, 1),
                             om_rng.choice(pressure_draw, 1)]
                om_r = model_dict['model'](list(param_dict.values()), om_coords)
                om_ell = om_r * np.sin(om_coords[0])
                om_condition = om_ell > om_rho_limit
                om_penalty = pm.Potential("open_magentosphere_penalty", pm.math.switch(om_condition, 0, -((rho_limit - om_ell)/0.5)**2))
            
            # if 'r2' in param_dict.keys():
            #     r2_condition = (np.abs(param_dict['r2']) < param_dict['r0'])
            #     r2_penalty = pm.Potential("r2_penalty", pm.math.switch(r2_condition, 0, -((np.abs(param_dict['r2']) - param_dict['r0'])/0.01)**2))
            # if 'r3' in param_dict.keys():
            #     r3_condition = (np.abs(param_dict['r3']) < param_dict['r0'])
            #     r3_penalty = pm.Potential("r3_penalty", pm.math.switch(r3_condition, 0, -((np.abs(param_dict['r3']) - param_dict['r0'])/0.01)**2))
            
            # a_f = model_dict['model'](list(param_dict.values()), data_draw_coords, return_a_f=True)
            # a_f_condition = a_f > 0.5
            # a_f_penalty = pm.Potential("a_f_penalty", pm.math.switch(a_f_condition, 0, -((0.5 - a_f)/0.0001)**2))
            
            # Optionally, handle multiple modes           
            if n_modes == 1:
                # dist = pm.Normal.dist(mu = mu0, sigma = sigma)
                # dist = pm.Gamma.dist(mu = mu0, sigma = sigma)
                # likelihood = pm.Potential("likelihood", pm.logp(dist, value = r_obs))
                likelihood = pm.Normal("likelihood", mu = mu0 - r_obs, sigma = sigma, observed=np.full(n, 0))
                
            elif n_modes == 2:
                #buffer_b = pm.HalfNormal("buffer_b", sigma=5)
                buffer_b = pm.TruncatedNormal("buffer_b", mu=5, sigma=1, lower=0)
                buffer_m = pm.TruncatedNormal("buffer_m", mu=0.05, sigma=0.05, lower=0)
                tracked_var_names.extend(["buffer_b", "buffer_m"])
                # log_buffer = pm.HalfNormal("log_buffer", 20/np.exp(log_mu0))
                
                buffer1 = pm.Deterministic("buffer1", buffer_b + buffer_m * mu0)
                mu1 = pm.Deterministic("mu1", mu0 + buffer1)
                # log_mu1 = pm.Deterministic("log_mu1", log_mu0 - log_buffer)
                
                # Weights for multi-modal fits
                w = pm.Dirichlet("w", [0.5, 0.5])
                tracked_var_names.append("w")
                
                dists = [pm.Gamma.dist(mu = mu0, sigma = sigma),
                         pm.Gamma.dist(mu = mu1, sigma = sigma)]
                # likelihood = pm.Mixture("likelihood", w=w, comp_dists = dists,
                #                         observed = np.zeros(n))
                x = pm.Mixture.dist(w=w, comp_dists = dists)
                likelihood = pm.Potential("likelihood", pm.logp(x, value = r_obs))
        
        # Prior predictives-- check that the inputs are reasonable
        with test_potential:
            prior_checks = pm.sample_prior_predictive(samples=1000)
        prior = prior_checks.prior
        
        fig, axs = plt.subplots(nrows=3)
        fig, ax = plt.subplots()
        ax.set(yscale='log')
        # Prior predictive plot 1: Does mu0 lie between upperbound and lowerbound?
        ax.fill_between(np.arange(n), data_draw['r_lowerbound'].values, data_draw['r_upperbound'].values,
                            color = 'green', alpha = 0.2)
        mu0_levels = np.percentile(prior['mu0'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
        ax.scatter(np.arange(n), mu0_levels[4], s=2, color='black')
        
        if 'mu1' in list(prior.variables):
            mu1_levels = np.percentile(prior['mu1'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
            ax.scatter(np.arange(n), mu0_levels[4], s=2, color='black', marker='x')
        
        # # Plot 2: overall histograms
        # bins = np.arange(0, 1000, 10)
        # h = axs[1].hist(prior['mu0'].values.flatten(), bins,
        #                 label = 'mu0 (Model Outputs)',
        #                 histtype = 'step', lw = 2, density = True)
        # h = axs[1].hist(prior['r_obs'].values.flatten(), bins,
        #                 label = 'r_obs (Possible Boundary Locations)',
        #                 histtype = 'step', lw = 2, density = True)
        # axs[1].legend()
        # # Plot 3: sigma-normalized residuals
        # residuals = prior['mu0'].values[0] - prior['r_obs'].values[0]
        # norm_mean_residuals = np.mean(residuals, axis=0)/np.std(prior['r_obs'].values[0], axis=0)
        # axs[2].plot(norm_mean_residuals,
        #             label = 'Column-Mean Residuals, Normalized by the standard deviation of r_obs',
        #             color = 'C2')
        # axs[2].legend()
        plt.show()
        # # breakpoint()
        
        if advi_mode == True:
            # with test_potential:
            #     # mean_fit = pm.fit(1000, method='svgd', inf_kwargs={'n_particles': 1000}, obj_optimizer=pm.sgd(learning_rate=0.01))
            #     mean_fit = pm.fit()
            # approx_idata = mean_fit.sample(2000)
            # approx_posterior = approx_idata.posterior
            # var_names_to_drop = list(set(list(approx_posterior.variables)) - set(tracked_var_names))
            # posterior_list.append(approx_posterior.drop_vars(var_names_to_drop))
            pass
        
        else:
            with test_potential:
                # 0.9 worked well for Joy+ models
                # 0.995 needed for new bimodal MP
                step = pm.NUTS(max_treedepth=15, target_accept=0.80) # max_treedepth=15, target_accept = 0.95
                idata = pm.sample(tune=1000, draws=n_draws, chains=n_chains, cores=2,
                                  # var_names = tracked_var_names,
                                  # init="advi")
                                  step = step,
                                  idata_kwargs={"log_likelihood": True})
                                  # init = 'adapt_diag+jitter') # prevents 'jitter', which might move points init vals around too much here

            idata_list.append(idata)
    
    # breakpoint()
    # if len(data_draws) > 1:
    #     # Need to uncomment posterior_list.append(posterior)
    #     # Need to uncomment posterior = xarray.concat(posterior_list, dim = 'chain')
    #     # breakpoint()
    # posterior = xarray.concat(posterior_list, dim = 'chain')
    
    # Concatenate all draws as added chains in one InferenceData
    idata_combined = az.concat(*idata_list, dim='chain')
    posterior_combined = idata_combined.posterior
    
    # Drop terms which we aren't tracking
    vars_to_drop = list(set(list(posterior_combined.keys())) - set(tracked_var_names))
    posterior_for_plot = posterior_combined.drop_vars(vars_to_drop)
    
    # Drop coords cooresponding to untracked vars
    posterior_coords = list(posterior_for_plot.coords)
    posterior_dim_coords = [c for c in posterior_coords if 'dim' in c]
    posterior_for_plot = posterior_for_plot.drop_vars(posterior_dim_coords)
    
    # Separate out the Dirichlet weights if multi-modal
    # if n_modes > 1:
    #     for mode_number in posterior['w_dim_0'].values:
    #         posterior['w{}'.format(mode_number)] = (("chain", "draw"), posterior['w'].values[:,:,mode_number])
    #     posterior = posterior.drop_vars('w')
    
    # # # BEFORE SAMPLING, ADD A POSTERIOR PREDICTIVE THAT COPIES THE PRIOR PREDICTIVE
    # fig, ax = plt.subplots()
    # # Prior predictive plot 1: Does mu0 lie between upperbound and lowerbound?
    # ax.set(yscale='log')
    # ax.fill_between(np.arange(n), data_draw['r_lowerbound'].values, data_draw['r_upperbound'].values,
    #                 color = 'green', alpha = 0.2)
    
    # import HybridMCMC_PosteriorPlots as HMCMCPP
    # f = HMCMCPP.make_BoundaryFunctionFromPosterior(model_name, posterior.to_dataframe())
    # r_models = f(data_draw.loc[:, ['t', 'p', 'p_dyn']].to_numpy().T, average=True)
    # for r_model in r_models:
    #     ax.scatter(np.arange(len(r_model)), r_model, marker='x', s=8)
    
    # mus = 
    # mu0_levels = np.percentile(prior['mu0'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
    # axs[0].scatter(np.arange(n), mu0_levels[4], s=2, color='black')
    
    # if 'mu1' in list(prior.variables):
    #     mu1_levels = np.percentile(prior['mu1'].values[0], [10, 20, 30, 40, 50, 60, 70, 80, 90], axis=0)
    #     axs[0].scatter(np.arange(n), mu0_levels[4], s=2, color='black', marker='x')
        
    return posterior_for_plot, model_dict, idata_combined