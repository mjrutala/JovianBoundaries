#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:26:05 2024

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

from sklearn.metrics import confusion_matrix

import BoundaryForwardModeling as BFM
import BoundaryModels as BM
import JoyBoundaryCoords as JBC

plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')

#   Data
location_df, coordinate_df = BFM._ready_DataFrames(dt.datetime(2016, 6, 1), dt.datetime(2024, 1, 1), resolution=10)

model_name = 'Shuelike' # '_Asymmetric'
model_dict = BM.init(model_name)
model_number = model_dict['model_number']

# breakpoint()

def find_BoundsForBowShock(location_df, coordinate_df):
    
    #   Set up some dummy coordinates for making nice plots
    # n = 10000
    # t = np.linspace(0, 0.9*np.pi, n)
    # p = np.zeros(n)
    # p_dyn = np.zeros(n)
    
    #   Times when the spacecraft is in the solar wind
    in_sw_indx = location_df.query('within_bs == 0').index
    not_in_sw_indx = location_df.query('within_bs == 1').index
    
    #   Create an array of potential standoff distances to try
    rs_totry = np.arange(50, 150, 0.5)
    
    #   For the lower bound: Shue+ model, no pressure dependence, flare exponent fixed at 0.5
    #   Cycle through increasing standoff distances until a solar wind measurement is detected
    #   The previous step sets a lower limit on the bow shock location
    fixed_fb = 0.5
    for i, r_totry in enumerate(rs_totry):
        
        t_test = coordinate_df.loc[in_sw_indx, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test))
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((r_totry, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = coordinate_df.loc[in_sw_indx, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_true > r_test).all() == False:
            lower_bound_params = (rs_totry[i-1], 0, fixed_fb, 0)
            break
        else:
            lower_bound_params = None
    
    #   For the upper bound: Shue+ model, no pressure dependence, flare exponent fixed at 0.8
    #   !!!! Set this flare based on a fit of the Shue model to Joy model
    #   Cycle through decreasing standoff distances until a non-solar-wind-measurement is detected
    #   The previous step sets an upper limit on the bow shock location
    fixed_fb = 0.8
    for i, r_totry in enumerate(np.flip(rs_totry)):
        
        t_test = coordinate_df.loc[not_in_sw_indx, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test))
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((r_totry, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = coordinate_df.loc[not_in_sw_indx, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_test > r_true).all() == False:
            upper_bound_params = (np.flip(rs_totry)[i-1], 0, fixed_fb, 0)
            break    
        else:
            upper_bound_params = None
    
    if (lower_bound_params is None) or (upper_bound_params is None):
        print('Failed finding one set of bounds!')
        print('This is likely due to too restrictive a range of standoff distances to test.')
        
    return lower_bound_params, upper_bound_params
    
lower_bound_params, upper_bound_params = find_BoundsForBowShock(location_df, coordinate_df)


coords = coordinate_df.loc[:, ['r', 't', 'p', 'p_dyn']].to_numpy()
r, t, p, p_dyn_loc = coords.T

#   Think about what this is!
sigma = 1

#
inorout_arr = location_df['within_bs'].to_numpy()

r_surface_mu, r_surface_sigma, r_surface_lower, r_surface_upper = [], [], [], []
for inorout, r_sc, t_sc, p_sc in zip(inorout_arr, r, t, p):
    if inorout == 0:
        # r_surface_mu.append(60)
        # r_surface_sigma.append(30)
        r_surface_lower.append(model_dict['model'](lower_bound_params, (t_sc, p_sc, 0.03)))
        r_surface_upper.append(r_sc)
    else:
        # r_surface_mu.append(60)
        # r_surface_sigma.append(30)
        r_surface_lower.append(r_sc)
        r_surface_upper.append(model_dict['model'](upper_bound_params, (t_sc, p_sc, 0.03)))
        
r_surface_mu = np.array(r_surface_mu)
r_surface_sigma = np.array(r_surface_sigma)
r_surface_lower = np.array(r_surface_lower)
r_surface_upper = np.array(r_surface_upper)

diff_test = [u - l for u, l in zip(r_surface_upper, r_surface_lower)]
if (np.array(diff_test) < 0).any():
    print('Something went wrong!')
    breakpoint()
    
#   Show the range of possible boundary surface locations
fig, ax = plt.subplots()
in_sw_indx = location_df.query('in_sw == 1').index
ax.scatter(coordinate_df.loc[in_sw_indx, 'ell'], 
           coordinate_df.loc[in_sw_indx, 'rho'],
           label = 'Solar Wind',
           s = 2, color='C0', marker='o')

# in_msh_indx = location_df.query('in_msh == 1').index
# ax.scatter(coordinate_df.loc[in_msh_indx, 'ell'],
#            coordinate_df.loc[in_msh_indx, 'rho'],
#            label = 'Magnetosheath',
#            s = 2, color='C4', marker='o')

t_fc = np.linspace(0, 0.9*np.pi, 1000)
p_fc = np.zeros(1000) + np.pi/2
p_dyn_fc = np.zeros(1000)
r_fc_upper = model_dict['model'](upper_bound_params, (t_fc, p_fc, p_dyn_fc))
r_fc_lower = model_dict['model'](lower_bound_params, (t_fc, p_fc, p_dyn_fc))
rpl_upper = BM.convert_SphericalSolarToCylindricalSolar(r_fc_upper, t_fc, p_fc)
rpl_lower = BM.convert_SphericalSolarToCylindricalSolar(r_fc_lower, t_fc, p_fc)

ax.plot(rpl_upper[2], rpl_upper[0],
        color = 'xkcd:gray', linestyle = '--',
        label = '$r_b$ = {}, $f_b$ = {}'.format(upper_bound_params[0], upper_bound_params[2]))
ax.plot(rpl_lower[2], rpl_lower[0],
        color = 'xkcd:gray', linestyle = '--',
        label = '$r_b$ = {}, $f_b$ = {}'.format(lower_bound_params[0], lower_bound_params[2]))
ax.fill(np.append(rpl_upper[2], rpl_lower[2][::-1]), np.append(rpl_upper[0], rpl_lower[0][::-1]),
        color = 'black', alpha = 0.25, edgecolor = None,
        label = 'Range of Possible \nBow Shock Surfaces')

x_joy = np.linspace(-150, 150, 10000)
ps_dyn_joy = [0.075, 0.375]
colors_joy = ['C2', 'C3']
for p_dyn_joy, color_joy in zip(ps_dyn_joy, colors_joy):
    z_joy = JBC.find_JoyBoundaries(p_dyn_joy, 'BS', x = x_joy, y = 0)
    ax.plot(x_joy, np.abs(z_joy[0]), color = color_joy, linestyle='--',
            label = r'Joy, $p_{{dyn}} = {} nPa, \phi = {}^\circ$'.format(p_dyn_joy, 0))
    
    y_joy = JBC.find_JoyBoundaries(p_dyn_joy, 'BS', x = x_joy, z = 0)
    ax.plot(x_joy, np.abs(y_joy[0]), color = color_joy, linestyle='-.',
            label = r'Joy, $p_{{dyn}} = {} nPa, \phi = {}^\circ$'.format(p_dyn_joy, 270))
    ax.plot(x_joy, y_joy[1], color = color_joy, linestyle=':',
            label = r'Joy, $p_{{dyn}} = {} nPa, \phi = {}^\circ$'.format(p_dyn_joy, 90))
    

ax.set(xlim = (150, -150), xlabel = r'$x_{JSS}$ [$R_J$] (+ toward Sun)',
       ylim = (0, 200), ylabel = r'$\rho_{JSS} = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]',
       aspect = 1)

ax.legend()
plt.show()

# breakpoint()

#   Trim data to account for class imbalance
#   We have way more times inside than outside (1s vs 0s)
#   We're only going to look at times when we're outside, 
#   plus the times inside before and after of equal lengths
    
#   Isolate chunks where we are outside the boundary
outside = np.argwhere(inorout_arr == 0).flatten()

list_of_outsides = np.split(outside, np.argwhere(np.diff(outside) != 1).flatten()+1)

balanced_indices = []
for outside_chunk in list_of_outsides:
    
    duration = len(outside_chunk)
    
    inside_chunk_left_indx = np.arange(outside_chunk[0]-duration-1,
                                          outside_chunk[0]-1)
    
    inside_chunk_right_indx = np.arange(outside_chunk[-1]+1,
                                        outside_chunk[-1]+duration+1)
    
    balanced_indices.extend(inside_chunk_left_indx)
    balanced_indices.extend(outside_chunk)
    balanced_indices.extend(inside_chunk_right_indx)

# #   Quasi-balanced, but with more sampling further down tail:
# #   Sample only near apojove
# balanced_indices = np.argwhere(r > 70).flatten()
# balanced_df = coordinate_df.query('r > 90 & ell > -100')
# balanced_indices.extend([coordinate_df.index.get_loc(i) for i in balanced_df.index])

apojove_indx = np.argwhere((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:])).flatten() + 1
balanced_indices.extend([i + step for i in apojove_indx for step in np.arange(-72, 72)])    #   +/- 12 hours (in ten minute intervals)

#   Make sure theres no negative or repeated indices
balanced_indices = np.array(list(set(balanced_indices)))
balanced_indices = balanced_indices[balanced_indices >= 0]
balanced_indices = np.sort(balanced_indices)


n_balanced = len(balanced_indices)
time_balanced = location_df.index[balanced_indices]

r_balanced = r[balanced_indices]
t_balanced = t[balanced_indices]
p_balanced = p[balanced_indices]  

p_dyn_loc_balanced = p_dyn_loc[balanced_indices]

r_surface_lower_balanced = r_surface_lower[balanced_indices]
r_surface_upper_balanced = r_surface_upper[balanced_indices]
# breakpoint()

with pm.Model() as potential_model:
    
    #   !!!! To be replaced with well-formed uncertainties
    p_dyn = pm.InverseGamma("p_dyn", mu=p_dyn_loc_balanced*2, sigma=p_dyn_loc_balanced)
    #p_dyn = p_dyn_loc_balanced*2
    
    #   N.B.: 'param_dict' must keep the same name, as it is used in code string evaluation below
    param_dict = {}
    for param_name in model_dict['param_distributions'].keys():
        param_dist = model_dict['param_distributions'][param_name]
        
        #   'EVAL_NEEDED' is for instances where parameter descriptions refer
        #   to other parameters (e.g., the minimum of one parameter is the 
        #   value of another). The code below executes these
        if 'EVAL_NEEDED' in model_dict['param_descriptions'][param_name].keys():
            
            this_param_descriptions = {}
            for key in model_dict['param_descriptions'][param_name].keys():
                if key != 'EVAL_NEEDED':
                    this_param_descriptions[key] = eval(model_dict['param_descriptions'][param_name][key])
                
            param = param_dist(param_name, **this_param_descriptions)
            del this_param_descriptions
            
        else:
            param = param_dist(param_name, **model_dict['param_descriptions'][param_name])
        param_dict[param_name] = param
       
    params = list(param_dict.values())
        
    #params = [r0, r1, a0, a1]
    
    sigma_dist = pm.HalfNormal("sigma_dist", sigma=sigma)
    
    # #   !!!! NOT WORKING YET
    # coords_std = np.zeros(np.shape(coords))
    # coords_std[:,0:3] = 1e-4
    # coords_std[:,3] = 1e-4 # artificial pressure errors, none on xyz
    # coords_dist = pm.TruncatedNormal('coords_dist', mu=coords.T, sigma=coords_std.T, lower = 1e-4, shape=np.shape(coords.T))
    # # # coords_dist = pm.Uniform('coords_dist', lower = coords.T - coords_std.T, upper = coords.T + coords_std.T)
    
    
    
    #   Both these are the same, but with pm.Deterministic it is tracked
    # r_b = pm.Deterministic("r_b", param_dict['r0'] * ((p_dyn)**param_dict['r1']))
    # f_b = pm.Deterministic("f_b", param_dict['a0'] + param_dict['a1']*p_dyn)
    # mu = pm.Deterministic("mu", r_b * (2/(1 + pm.math.cos(t)))**f_b)
   
    r_b = param_dict['r0'] * ((p_dyn)**param_dict['r1'])
    f_b = param_dict['a0'] + param_dict['a1']*p_dyn
    mu = r_b * (2/(1 + pm.math.cos(t_balanced)))**f_b
    
    #   !!!! Move this into the Boundary Models section?
    #   Now that we've rewritten this code, does this even make sense?
    #   Should it just be the normal distribution sigma, since that is about y?
    # internal_sigma = pm.Normal("stochastic_sigma", mu=0, sigma=10)
    
    # observed_r = pm.TruncatedNormal("observed_r",
    #                                 mu = r_surface_mu,
    #                                 sigma = r_surface_sigma,
    #                                 lower = r_surface_lower,
    #                                 upper = r_surface_upper)
    observed_r = pm.Uniform("observed_r",
                            lower = r_surface_lower_balanced,
                            upper = r_surface_upper_balanced)
    
    # Define likelihood
    likelihood = pm.Normal("obs", mu = mu - observed_r, 
                            sigma=sigma_dist, 
                            observed = np.zeros(n_balanced))
    
    # likelihood = pm.Potential("obs", pm.logp(pm.Normal.dist(mu=mu, sigma=sigma_dist), value=observed_r))


    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    # idata = sample(3000)
    
    idata = pm.sample(tune=1000, draws=1000, chains=4, cores=4)
    
    # posterior = pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    
    # coords_dist = coords.T
    
    # # use a Potential instead of a CustomDist
    # pm.Potential("likelihood", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    
    # # pred = pm.Potential("pred", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    # # #pred = pm.Deterministic("pred", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    
    # # likelihood = pm.Normal("likelihood", pred, sigma_dist, observed=coords.T[0])

#   Get the posterior from the inference data
posterior = idata.posterior

#   Select only the variables which don't change with time/spacecraft position
mod_posterior = posterior.drop_vars(['observed_r', 'p_dyn'])

#   Plot a trace to check for convergence
az.plot_trace(mod_posterior)

#   Plot a corner plot to investigate covariance
figure = corner.corner(mod_posterior, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,)
figure.set_size_inches(8,8)

plt.show()

breakpoint()
# =============================================================================
#   Posterior predictive plots
#   (a.k.a, the ones that actually show us physical things)
#   - Show the 
# =============================================================================

#   Sample the posterior
samples = az.extract(idata, group='posterior', num_samples=50)
r0_s, r1_s, a0_s, a1_s = samples['r0'].values, samples['r1'].values, samples['a0'].values, samples['a1'].values
p_dyn_s = samples['p_dyn'].values
#p_dyn_s = np.stack([p_dyn]*50).T
sigma_s = samples['sigma_dist'].values

obs_r_s = samples['observed_r'].values

r_b_s = r0_s * (p_dyn_s**r1_s)
f_b_s = a0_s + a1_s*p_dyn_s

t_for_s = np.array([t_balanced]*50).T

mu_s = r_b_s * (2/ (1 + np.cos(t_for_s)))**f_b_s

#   Plotting the figure
fig, axd = plt.subplot_mosaic("""
                              aab
                              ccc
                              """, 
                              width_ratios=[1,1,1], height_ratios=[1,1],
                              figsize = (6,4))
#   Plot a posterior predictive of the time series in the bottom panel
c_limit = axd['c'].scatter(np.array([time_balanced]*50), obs_r_s.T,
                           color='xkcd:light gray', alpha=1, marker='o', ec=None, s=1, zorder=-10,
                           label = 'Refined Observations')
c_bound = axd['c'].scatter(np.array([time_balanced]*50), mu_s.T, 
                           color='C0', alpha=1/256, marker='.', s=1, zorder=10,
                           label = 'Modeled Bow Shock Locations')

c_orbit = axd['c'].plot(coordinate_df.index, coordinate_df['r'], 
                        color='xkcd:Blue', lw=1, zorder=2,
                        label = 'Spacecraft Position')

axd['c'].set(xlabel='Date', 
             ylim=(80, 200), ylabel=r'Radial Distance $[R_J]$', yscale='log')
    

#   Plot the actual shape of this boundary
p_dyn_s_10, p_dyn_s_50, p_dyn_s_90 = np.percentile(p_dyn_s.flatten(), [25, 50, 75])

t_ = np.array([np.linspace(0, 0.75 * np.pi, 1000)]*50).T
p_ = np.zeros((1000, 50))
p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_10
r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)

a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
                        color='C2', lw = 1, alpha=1/5)
a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
                        color='C2', lw = 1, alpha=1)

p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_50
p_ = np.zeros((1000, 50)) + 1*np.pi/2.
r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
                        color='C4', lw = 1, alpha=1/5)
a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
                        color='C4', lw = 1, alpha=1)

p_ = np.zeros((1000, 50)) + p_dyn_s_90
r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
                        color='C5', lw = 1, alpha=1/5)
a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
                        color='C5', lw = 1, alpha=1)


axd['a'].set(xlim=(125,-250),
             ylim=(000, 200),
             aspect=1)


#   Confusion matrix
# cm_results = {'true_neg':[], 'false_pos':[], 'false_neg':[], 'true_pos':[]}
cm = np.zeros((2,2))
for mu_sample in mu_s.T:
    
    #   If mu > spacecraft r, then we are modeled to be inside the bow shock
    #   Inside gets a 1, outside gets a 0
    predicted_location = [1 if entry else 0 for entry in (mu_sample > r_balanced)]
    
    #   In actuality, we know how often we are outside the bow shock
    measured_location = location_df['within_bs'].iloc[balanced_indices].to_numpy()
    
    #   Compare these with a confusion matrix
    cm += confusion_matrix(measured_location, predicted_location, labels = (0, 1))
    # tn, fp, fn, tp = cm.ravel()
    
    # cm_results['true_neg'].append(tn)
    # cm_results['false_pos'].append(fp)
    # cm_results['false_neg'].append(fn)
    # cm_results['true_pos'].append(tp)
    
axd['b'].imshow(cm / (n_balanced * 50) * 100, 
                cmap = 'plasma', extent=(0,1,0,1))

axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,0]/(n_balanced*50)*100),
                  (1/4, 3/4), ha='center', va='center')
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,1]/(n_balanced*50)*100),
                  (3/4, 3/4), ha='center', va='center')
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,0]/(n_balanced*50)*100),
                  (1/4, 1/4), ha='center', va='center')
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,1]/(n_balanced*50)*100),
                  (3/4, 1/4), ha='center', va='center')

#   This is verbose, but gives fine control over the labels for the confusion matrix
axd['b'].set(xlabel = 'Model Prediction', xlim = (0,1), xticks=[0.5], xticklabels=[''],
             ylabel = 'Real Location', ylim = (0,1), yticks = [0.5], yticklabels=[''])
axd['b'].annotate('Inside', 
                  (0,3/4), (-0.1,0), 'axes fraction', 'offset fontsize', 
                  rotation='vertical', ha='right', va='center')
axd['b'].annotate('Outside', 
                  (0,1/4), (-0.1,0), 'axes fraction', 'offset fontsize', 
                  rotation='vertical', ha='right', va='center')
axd['b'].annotate('Inside', 
                  (1/4,0), (0,-0.2), 'axes fraction', 'offset fontsize', 
                  rotation='horizontal', ha='center', va='top')
axd['b'].annotate('Outside', 
                  (3/4,0), (0,-0.2), 'axes fraction', 'offset fontsize', 
                  rotation='horizontal', ha='center', va='top')
axd['b'].grid(visible=True)


# axs[0].plot(time_balanced, np.mean(mu_s, 1), color='xkcd:magenta')
# # axs[0].fill_between(time_balanced, np.mean(mu_s, 1)-np.mean(sigma_s), np.mean(mu_s, 1) + np.mean(sigma_s), 
# #                     color='xkcd:magenta', alpha=0.25)

# axs[0].plot(time_balanced, r_balanced, color='xkcd:blue', linestyle='--')
# axs[0].set(ylim = (80, 150))

# axs0_twinx = axs[0].twinx()
# axs0_twinx.plot(time_balanced, inorout_arr[balanced_indices], color='red', linewidth=3)
# axs0_twinx.set(ylim = (0.1, -1))

# # axs[1].plot(np.vstack([location_df.index]*50), p_dyn_s.T, color='black', alpha=0.005)
# axs[1].plot(time_balanced, p_dyn_s.T[0], color='black', alpha=1)

# plt.show()


# cm_results = {'true_neg':[], 'false_pos':[], 'false_neg':[], 'true_pos':[]}
# for mu_sample in mu_s.T:
    
#     #   If mu > spacecraft r, then we are modeled to be inside the bow shock
#     #   Inside gets a 1, outside gets a 0
#     predicted_location = [1 if entry else 0 for entry in (mu_sample > r_balanced)]
    
#     #   In actuality, we know how often we are outside the bow shock
#     measured_location = location_df['within_bs'].iloc[balanced_indices].to_numpy()
    
#     #   Compare these with a confusion matrix
#     cm = confusion_matrix(measured_location, predicted_location, labels = (0, 1))
#     tn, fp, fn, tp = cm.ravel()
    
#     cm_results['true_neg'].append(tn)
#     cm_results['false_pos'].append(fp)
#     cm_results['false_neg'].append(fn)
#     cm_results['true_pos'].append(tp)

