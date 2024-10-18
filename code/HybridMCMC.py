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

resolution = '10Min'

positions_df = postproc.PostprocessCrossings('BS', spacecraft_to_use = ['Ulysses', 'Cassini', 'Juno'])

# Galileo binary search-- what causes our problems?
n = len(positions_df)

positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')

# FOR TESTING, TO MAKE THE MCMC SAMPLER RUN FASTER
# positions_df = positions_df.sample(frac=0.5, axis='rows')

from scipy.stats import skewnorm

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

n_pressure_draws = 1
posterior_list = []

test_pressure_dist = pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha)
test_pressure_draws = pm.draw(test_pressure_dist, draws=n_pressure_draws)
# Hacky, but assume negative pressures correspond to small pressures
test_pressure_draws[test_pressure_draws < 0] = np.min(test_pressure_draws[test_pressure_draws > 0])

if len(np.shape(test_pressure_draws)) < 2:
    test_pressure_draws = [test_pressure_draws]

# Model and MCMC params
model_name = 'Shuelike_AsymmetryCase1'
model_dict = BM.init(model_name)
model_number = model_dict['model_number']

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
            # tracked_var_names.extend([pname+'_mu', pname+'_sigma', pname]) # Do track combined param
            # tracked_var_names.extend([pname+'_mu', pname+'_sigma']) # Don't track combined param
            tracked_var_names.append(pname)
        
        params = list(param_dict.values())
        #params = [r0, r1, a0, a1]
        
        # The predicted mean location of the boundary is a function of pressure
        mu_pred = pm.Deterministic("mu_pred", model_dict['model'](params, [t, p, p_dyn_obs]))
        
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
        
        sigma = pm.HalfNormal("sigma", sigma=10)
        sigma_t = pm.HalfNormal("sigma_t", sigma = sigma * (2/(1 + np.cos(t))))
        tracked_var_names.extend(['sigma'])
        
        likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu=mu_pred, sigma=sigma), value=r_obs))

    with test_potential:
        idata = pm.sample(tune=500, draws=500, chains=4, cores=3,
                          var_names = tracked_var_names,
                          target_accept=0.95)
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


# =============================================================================
#   Posterior predictive plots
#   (a.k.a, the ones that actually show us physical things)
#   - Show the 
# =============================================================================






# #   Sample the posterior
# samples = az.extract(idata, group='posterior', num_samples=50)
# r0_s, r1_s, a0_s, a1_s = samples['r0'].values, samples['r1'].values, samples['a0'].values, samples['a1'].values
# p_dyn_s = samples['p_dyn'].values
# #p_dyn_s = np.stack([p_dyn]*50).T
# sigma_s = samples['sigma_dist'].values

# obs_r_s = samples['observed_r'].values

# r_b_s = r0_s * (p_dyn_s**r1_s)
# f_b_s = a0_s + a1_s*p_dyn_s

# t_for_s = np.array([t_balanced]*50).T

# mu_s = r_b_s * (2/ (1 + np.cos(t_for_s)))**f_b_s

# #   Plotting the figure
# fig, axd = plt.subplot_mosaic("""
#                               aab
#                               ccc
#                               """, 
#                               width_ratios=[1,1,1], height_ratios=[1,1],
#                               figsize = (9,6))
# #   Plot a posterior predictive of the time series in the bottom panel
# c_limit = axd['c'].scatter(np.array([time_balanced]*50), obs_r_s.T,
#                            color='gray', alpha=0.05, marker='o', ec=None, s=1, zorder=-10,
#                            label = 'Refined Range of Possible \nBow Shock Surfaces')
# c_bound = axd['c'].scatter(np.array([time_balanced]*50), mu_s.T, 
#                            color='C0', alpha=1, marker='.', s=1, zorder=10,
#                            label = 'Modeled Bow Shock Locations')

# c_orbit = axd['c'].plot(coordinate_df.index, coordinate_df['r'], 
#                         color='C4', lw=1, zorder=2, ls='--',
#                         label = 'Spacecraft Position')

# axd['c'].set(xlabel='Date', 
#              ylim=(80, 200), ylabel=r'Radial Distance $[R_J]$', yscale='log')
    
# axd['c'].legend(loc='upper right')
# axd['c'].set_xlim(dt.datetime(2016,5,1), dt.datetime(2025,1,1))

# #   Plot the actual shape of this boundary
# p_dyn_s_10, p_dyn_s_50, p_dyn_s_90 = np.percentile(p_dyn_s.flatten(), [25, 50, 75])

# t_ = np.array([np.linspace(0, 0.75 * np.pi, 1000)]*50).T
# # p_ = np.zeros((1000, 50))
# p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_10
# # r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
# # rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)

# # a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
# #                         color='C2', lw = 1, alpha=1/5)
# # a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
# #                         color='C2', lw = 1, alpha=1)

# p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_50
# p_ = np.zeros((1000, 50)) + 1*np.pi/2.
# r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
# rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
# # a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
# #                         color='C4', lw = 1, alpha=1/5)
# a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
#                         color='C2', lw = 1, alpha=1,
#                         label = 'Mean Boundary Location \n@ Median Pressure')

# r_upper_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_)) + sigma_s
# rpl_upper_ = BM.convert_SphericalSolarToCylindricalSolar(r_upper_, t_, p_)
# r_lower_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_)) - sigma_s
# rpl_lower_ = BM.convert_SphericalSolarToCylindricalSolar(r_lower_, t_, p_)

# # axd['a'].fill(np.append(np.mean(rpl_upper_[2].T, 0), np.mean(rpl_lower_[2], 0)[::-1]), 
# #               np.append(np.mean(rpl_upper_[0].T, 0), np.mean(rpl_lower_[0], 0)[::-1]),
# #               color = 'C2', alpha = 0.4, edgecolor = None,
# #               label = r'Mean Boundary Location $\pm1\sigma$' + '\n@ Median Pressure')

# a_nor_m = axd['a'].plot(np.mean(rpl_upper_[2].T, 0), np.mean(rpl_upper_[0].T, 0),
#                         color='C2', lw = 1, alpha=1, ls=':',
#                         label = r'Mean Boundary Location $+1\sigma$')

# a_nor_m = axd['a'].plot(np.mean(rpl_lower_[2].T, 0), np.mean(rpl_lower_[0].T, 0),
#                         color='C2', lw = 1, alpha=1, ls=':', 
#                         label = r'Mean Boundary Location $-1\sigma$')
# axd['a'].legend()

# # p_ = np.zeros((1000, 50)) + p_dyn_s_90
# # r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
# # rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
# # a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
# #                         color='C5', lw = 1, alpha=1/5)
# # a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
# #                         color='C5', lw = 1, alpha=1)


# axd['a'].set(xlim=(150,-400),
#              # ylim=(000, 200),
#              aspect=1)


# #   Confusion matrix
# # cm_results = {'true_neg':[], 'false_pos':[], 'false_neg':[], 'true_pos':[]}
# cm = np.zeros((2,2))
# for mu_sample in mu_s.T:
    
#     #   If mu > spacecraft r, then we are modeled to be inside the bow shock
#     #   Inside gets a 1, outside gets a 0
#     predicted_location = [1 if entry else 0 for entry in (mu_sample > r_balanced)]
    
#     #   In actuality, we know how often we are outside the bow shock
#     measured_location = location_df['within_bs'].iloc[balanced_indices].to_numpy()
    
#     #   Compare these with a confusion matrix
#     cm += confusion_matrix(measured_location, predicted_location, labels = (0, 1))
#     # tn, fp, fn, tp = cm.ravel()
    
#     # cm_results['true_neg'].append(tn)
#     # cm_results['false_pos'].append(fp)
#     # cm_results['false_neg'].append(fn)
#     # cm_results['true_pos'].append(tp)
    
# axd['b'].imshow(cm / (n_balanced * 50) * 100, 
#                 cmap = 'plasma', extent=(0,1,0,1))

# import matplotlib.patheffects as pe
# axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,0]/(n_balanced*50)*100),
#                   (1/4, 3/4), ha='center', va='center', color='white',
#                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])
# axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,1]/(n_balanced*50)*100),
#                   (3/4, 3/4), ha='center', va='center', color='white',
#                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])
# axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,0]/(n_balanced*50)*100),
#                   (1/4, 1/4), ha='center', va='center', color='white',
#                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])
# axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,1]/(n_balanced*50)*100),
#                   (3/4, 1/4), ha='center', va='center', color='white',
#                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])

# #   This is verbose, but gives fine control over the labels for the confusion matrix
# axd['b'].set(xlabel = 'Model Prediction', xlim = (0,1), xticks=[0.5], xticklabels=[''],
#              ylabel = 'Real Location', ylim = (0,1), yticks = [0.5], yticklabels=[''])
# axd['b'].annotate('Inside', 
#                   (0,3/4), (-0.1,0), 'axes fraction', 'offset fontsize', 
#                   rotation='vertical', ha='right', va='center')
# axd['b'].annotate('Outside', 
#                   (0,1/4), (-0.1,0), 'axes fraction', 'offset fontsize', 
#                   rotation='vertical', ha='right', va='center')
# axd['b'].annotate('Inside', 
#                   (1/4,0), (0,-0.2), 'axes fraction', 'offset fontsize', 
#                   rotation='horizontal', ha='center', va='top')
# axd['b'].annotate('Outside', 
#                   (3/4,0), (0,-0.2), 'axes fraction', 'offset fontsize', 
#                   rotation='horizontal', ha='center', va='top')
# axd['b'].grid(visible=True)


# # axs[0].plot(time_balanced, np.mean(mu_s, 1), color='xkcd:magenta')
# # # axs[0].fill_between(time_balanced, np.mean(mu_s, 1)-np.mean(sigma_s), np.mean(mu_s, 1) + np.mean(sigma_s), 
# # #                     color='xkcd:magenta', alpha=0.25)

# # axs[0].plot(time_balanced, r_balanced, color='xkcd:blue', linestyle='--')
# # axs[0].set(ylim = (80, 150))

# # axs0_twinx = axs[0].twinx()
# # axs0_twinx.plot(time_balanced, inorout_arr[balanced_indices], color='red', linewidth=3)
# # axs0_twinx.set(ylim = (0.1, -1))

# # # axs[1].plot(np.vstack([location_df.index]*50), p_dyn_s.T, color='black', alpha=0.005)
# # axs[1].plot(time_balanced, p_dyn_s.T[0], color='black', alpha=1)

# # plt.show()


# # cm_results = {'true_neg':[], 'false_pos':[], 'false_neg':[], 'true_pos':[]}
# # for mu_sample in mu_s.T:
    
# #     #   If mu > spacecraft r, then we are modeled to be inside the bow shock
# #     #   Inside gets a 1, outside gets a 0
# #     predicted_location = [1 if entry else 0 for entry in (mu_sample > r_balanced)]
    
# #     #   In actuality, we know how often we are outside the bow shock
# #     measured_location = location_df['within_bs'].iloc[balanced_indices].to_numpy()
    
# #     #   Compare these with a confusion matrix
# #     cm = confusion_matrix(measured_location, predicted_location, labels = (0, 1))
# #     tn, fp, fn, tp = cm.ravel()
    
# #     cm_results['true_neg'].append(tn)
# #     cm_results['false_pos'].append(fp)
# #     cm_results['false_neg'].append(fn)
# #     cm_results['true_pos'].append(tp)

