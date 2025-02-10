#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:34:34 2024

@author: mrutala
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import datetime as dt
import pandas as pd
import tqdm

import sklearn.metrics as metrics
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.stats import gamma

import JoyBoundaryCoords as JBC
import BoundaryModels as BM
import CrossingPreprocessingRoutines as preproc
import CrossingPostprocessingRoutines as postproc

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

def PosteriorAnalysis(boundary):
    
    basedir = '/Users/mrutala/projects/JupiterBoundaries/posteriors/'
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    
    # Setup a dataframe to hold the different models
    df = pd.DataFrame(columns = ['label', 'model_name', 'posterior_path', 'posterior_fn', 'inference_path'])
    
    # Add the basic J02 model
    J02_params = JBC.get_JoyParameters(boundary)
    J02_keylist = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1', 'F0', 'F1']
    J02_comparison_df = pd.DataFrame([J02_params], columns=J02_keylist, index=[0])
    J02_comparison_df['sigma_m'] = 0.001 #  0.095 # From Joy+ 2002
    J02_comparison_df['sigma_b'] = 0.001 #  8 # From Joy+ 2002
    J02_fn = make_BoundaryFunctionFromPosterior('Joylike_r1fixed', J02_comparison_df)
    df.loc[len(df)] = ['J02 (original)', 'Joylike_r1fixed', 'NULL', J02_fn, 'NULL']
    
    if boundary == 'MP':
        # Post-Cassini MP inclusion
        df.loc[len(df)] = [r'S97*', 'Shuelike_r1fixed',
                           basedir + 'MP_S97/Shuelike_r1fixed-MP_20250202181534_posterior.pkl', '',
                           basedir + 'MP_S97/Shuelike_r1fixed-MP_20250202181534_inferencedata.nc']
        df.loc[len(df)] = [r'New', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
                           basedir + 'MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_posterior.pkl', '',
                           basedir + 'MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_inferencedata.nc']
        demoplot_kwargs = {'xlim': [200, -200], 'aspect': 1}
        
        
    elif boundary == 'BS':
        df.loc[len(df)] = [r'S97*', 'Shuelike_r1fixed',
                           basedir + 'BS_S97/Shuelike_r1fixed-BS_20250202195829_posterior.pkl', '',
                           basedir + 'BS_S97/Shuelike_r1fixed-BS_20250202195829_inferencedata.nc']
        df.loc[len(df)] = [r'New', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
                           basedir + 'BS_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20250202214617_posterior.pkl', '',
                           basedir + 'BS_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20250202214617_inferencedata.nc']
        demoplot_kwargs = {'xlim': [200, -200], 'aspect': 1}
        
        
    # Iterate over each model (other than J02), adding a function for calculating the boundaries
    for index, row in df.loc[1:].iterrows():
        posterior_df = read_PosteriorAsDataFrame(row['posterior_path'])
        fn = make_BoundaryFunctionFromPosterior(row['model_name'], posterior_df)
        df.loc[index, 'posterior_fn'] = fn
    
    positions_df = get_PositionsForAnalysis()
    positions_df = positions_df.query("UN != 1")
    
    # Add the bounds to each entry in the df
    bound_params = postproc.find_BoundarySurfaceLimits(positions_df, boundary, spacecraft_to_use)
    positions_df = postproc.add_Bounds(positions_df, boundary, bound_params['upperbound'], bound_params['lowerbound'])
    
    # # Print the confusion matrix statistics
    # for _, row in df.iterrows():
    #     print(row['label'])
    #     WithinBounds(boundary, row['model_name'], row['posterior_fn'], positions_df)
    #     # ConfusionMatrix(boundary, row['model_name'], row['posterior_fn'], positions_df)
    #     # GuessCrossings(boundary, row['model_name'], row['posterior_fn'], positions_df)
    #     # ConfusionMatrix(boundary, row['model_name'], row['posterior_path'])
    
    # COMPARE BAYESIAN MODELS
    bayesian_comparison_dict = {}
    for _, row in df.iterrows():
        if row['inference_path'] != 'NULL':
            
            # We have to remove the p_dyn_draw log likelihood-- 
            # WAIC can only be computed for one log likelihood (at a time)
            temp_idata = az.from_netcdf(row['inference_path'])
            temp_idata.log_likelihood = temp_idata.log_likelihood.drop_vars(['p_dyn_draw', 'p_dyn_draw_dim_0'])
            bayesian_comparison_dict[row['label']] = temp_idata
            
    loo_comparison = az.compare(bayesian_comparison_dict, 'loo')
    waic_comparison = az.compare(bayesian_comparison_dict, 'waic')
    print("LOO Results")
    print(loo_comparison['weight'])
    print("WAIC Results")
    print(waic_comparison['weight'])

    # breakpoint()    
        
    p_dyn_df = pd.DataFrame(columns = ['percentile', 'value', 'color'])
    p_dyn_df['percentile'] = [10, 50, 90]
    p_dyn_df['value'] = np.percentile(positions_df['p_dyn'], p_dyn_df['percentile'])
    p_dyn_df['color'] = ['C0', 'C1', 'C3']
    
    p_df = pd.DataFrame(columns = ['location', 'angle', 'ls'])
    p_df['location'] = ['Polar', 'Dawn', 'Dusk']
    p_df['angle'] = [0, np.pi/2, -np.pi/2]
    p_df['ls'] = ['-', '-.', ':']
    
    # Print the polar flattening and dawn/dusk ratios
    coords_north = [np.pi/2, 0, p_dyn_df.query('percentile == 50')['value']]
    coords_dawn = [np.pi/2, np.pi/2., p_dyn_df.query('percentile == 50')['value']]
    coords_dusk = [np.pi/2, -np.pi/2., p_dyn_df.query('percentile == 50')['value']]
    for _, row in df.iterrows():
        # r_north = np.array(get_BoundariesFromPosterior(row['model_name'], coords_north, row['posterior_path'], n=100)).flatten()
        # r_dawn = np.array(get_BoundariesFromPosterior(row['model_name'], coords_dawn, row['posterior_path'], n=100)).flatten()
        # r_dusk = np.array(get_BoundariesFromPosterior(row['model_name'], coords_dusk, row['posterior_path'], n=100)).flatten()
        
        r_north = np.array(row['posterior_fn'](coords_north, n=1000)).flatten()
        r_dawn = np.array(row['posterior_fn'](coords_dawn, n=1000)).flatten()
        r_dusk = np.array(row['posterior_fn'](coords_dusk, n=100)).flatten()
        
        print(row['label'])
        mu_flattening = (np.mean(r_dawn) - np.mean(r_north)) / np.mean(r_dawn)
        sig_flattening = mu_flattening * np.sqrt((np.sqrt(np.std(r_dawn)**2 + np.std(r_north)**2)/(np.mean(r_dawn) - np.mean(r_north)))**2 + (np.std(r_dawn)/np.mean(r_dawn))**2)
        print("Dawn-to-Polar Flattening: {:.3f} +/- {:.3f}".format(mu_flattening, sig_flattening))
        # mu_flattening = (np.mean(r_dusk) - np.mean(r_north)) / np.mean(r_dusk)
        # sig_flattening = mu_flattening * np.sqrt((np.sqrt(np.std(r_dusk)**2 + np.std(r_north)**2)/(np.mean(r_dusk) - np.mean(r_north)))**2 + (np.std(r_dusk)/np.mean(r_dusk))**2)
        # print("Dusk-to-Polar Flattening: {:.3f} +/- {:.3f}".format(mu_flattening, sig_flattening))
        # print(" ")
        
    
    # =============================================================================
    # Posterior plot + 'frequentized' comparisons w/ Joy+ (2002)
    # =============================================================================
    # fig, axd = plt.subplot_mosaic("""
    #                              ab
    #                              cd
    #                              ef
    #                              ..
    #                              gg
    #                              hh
    #                              """, height_ratios=[3, 3, 3, 1, 5, 1], figsize = (6, 8))
    # plt.subplots_adjust(bottom = 0.50/8, left = 0.75/6, top = 1 - 0.75/8, right = 1 - 0.125/6,
    #                     wspace = 0.50/6, hspace = 0.75/8)
    fig, axd = plt.subplot_mosaic("""
                                 ab
                                 cd
                                 ef
                                 """, height_ratios=[3, 3, 3], width_ratios=[1, 1], figsize = (6, 6))
    plt.subplots_adjust(bottom = 0.75/6, left = 0.75/6, top = 1 - 0.75/6, right = 1 - 0.125/6,
                        wspace = 0.50/6, hspace = 0.50/6)
    
    # Calculate the ylimits needed for the biggest equal-aspect subplots
    bbox = axd['a'].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_ratio = bbox.height/bbox.width
    demoplot_kwargs['ylim'] = [0, abs(demoplot_kwargs['xlim'][1] - demoplot_kwargs['xlim'][0]) * bbox_ratio]
    
    # Set axes titles
    pos = [(np.array(axd['a'].get_position())[0][0] + np.array(axd['b'].get_position())[1][0])/2,
           (np.array(axd['e'].get_position())[0][1] + 0)/3]
    fig.text(*pos, r'$x_{JSS}$ $[R_J]$ (+ Sunward)',
             fontsize='large', ha='center', va='center')
    
    pos = [(0 + np.array(axd['a'].get_position())[0][0])/5,
           (np.array(axd['a'].get_position())[0][1] + np.array(axd['e'].get_position())[1][1])/2]
    fig.text(*pos, r'$\rho = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ $[R_J]$',
             fontsize='large', rotation=90, ha='center', va='center')
    
    pos = [(np.array(axd['a'].get_position())[1][0] + np.array(axd['b'].get_position())[0][0])/2,
           (np.array(axd['b'].get_position())[0][1] + np.array(axd['f'].get_position())[1][1])/2]
    fig.text(*pos, r'$z_{JSS} [R_J]$ (+ Northward)',
             fontsize='large', rotation=90, ha='center', va='center')

    # Label each subplot
    for letter, ax in axd.items():
        ax.annotate('({})'.format(letter), (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    ha = 'center', va = 'center')
        

    for key, (_, row) in zip(['a', 'c', 'e'], df.iterrows()):
        
        # Adjust plots
        axd[key].set(ylabel = row['label'], **demoplot_kwargs)
        
        for _, p_dyn_row in p_dyn_df.iterrows():
            for _, p_row in p_df.iterrows():
                
                # Get coordinates for plotting
                t = np.linspace(0, 0.99*np.pi, 1000)
                p = np.full(1000, p_row['angle'])
                p_dyn = np.full(1000, p_dyn_row['value'])
                
                # rs may have multiple modes, so loop over each
                rs = row['posterior_fn']([t, p, p_dyn], average=True)
                for r in rs:
                    rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                    axd[key].plot(rpl[2], rpl[0], 
                                  color = p_dyn_row['color'], 
                                  ls = p_row['ls'], 
                                  lw = 2)
            
    # In these plots, show uncertainties + data, for reference        
    for key, (_, row) in zip(['b', 'd', 'f',], df.iterrows()):
        
        # Adjust plots
        axd[key].set(**demoplot_kwargs)
        
        for _, p_dyn_row in p_dyn_df.iterrows():
            
            # Nominal coordinate values
            t = np.linspace(0, 0.99 * np.pi, 1000)
            p = np.full(1000, 0)
            p_dyn = np.full(1000, p_dyn_row['value'])
            
            # Sample the boundary fit *n* times
            rs = row['posterior_fn']([t, p, p_dyn], n=100)
            for r in rs:
                rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                axd[key].plot(rpl[2], rpl[0], 
                                  color=p_dyn_row['color'], lw = 1.5, alpha=0.05, zorder=-2)
            
            # Compare to the average boundary
            rs = row['posterior_fn']([t, p, p_dyn], average=True)
            for r in rs:
                rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                # Plotting black behind these lines help them stick out against samples
                axd[key].plot(rpl[2], rpl[0], 
                              color='black', lw = 2.4, alpha=1, zorder=16)
                axd[key].plot(rpl[2], rpl[0], 
                              color=p_dyn_row['color'], lw=2, alpha=1, zorder=16)
        
        # =============================================================================
        #         # Rotate data to North
        # =============================================================================
        # 1) Where is the nominal boundary compared to the point?
        # r_boundary_at_point = get_BoundariesFromPosterior(row['model_name'], 
        #                                                   positions_df[['t', 'p', 'p_dyn']].to_numpy().T, 
        #                                                   row['posterior_path'],
        #                                                   mean=True)
        r_boundary_at_point = row['posterior_fn'](positions_df[['t', 'p', 'p_dyn']].to_numpy().T, average=True)
        # delta_r = [positions_df['r'].to_numpy() - r_at_point for r_at_point in r_boundary_at_point]
        fraction_r = [positions_df['r'].to_numpy() / r_at_point for r_at_point in r_boundary_at_point]
        
        # 2) Preserve delta_r 
        t_rotated = positions_df['t'].to_numpy()
        p_rotated = np.full(np.shape(t_rotated), 0) # Forced to North
        p_dyn_rotated = positions_df['p_dyn'].to_numpy()
        
        # r_rotated = get_BoundariesFromPosterior(row['model_name'], 
        #                                         [t_rotated, p_rotated, p_dyn_rotated], 
        #                                         row['posterior_path'],
        #                                         mean=True)
        r_rotated = row['posterior_fn']([t_rotated, p_rotated, p_dyn_rotated], average=True)
        # r_rotated = [r + dr for r, dr in zip(r_rotated, delta_r)]
        r_rotated = [r * fr for r, fr in zip(r_rotated, fraction_r)]
        
        # 3) Reassign to df
        rotated_df = positions_df.copy(deep=True)
        rotated_df['p'] = p_rotated
        rotated_df['r'] = r_rotated[0] # 0 is the 'default' mode
        
        xyz = BM.convert_SphericalSolarToCartesian(*rotated_df[['r', 't', 'p']].to_numpy().T)
        rotated_df.loc[:, ['x', 'y', 'z']] = xyz.T
        rpl = BM.convert_SphericalSolarToCylindricalSolar(*rotated_df[['r', 't', 'p']].to_numpy().T)
        rotated_df.loc[:, ['rho', 'phi', 'ell']] = rpl.T
        
        # 4) Plot the actual crossings
        for spacecraft in spacecraft_to_use:
            subset_df = rotated_df.query('spacecraft == @spacecraft')
            # subset_df = positions_df.query('spacecraft == @spacecraft')
            
            if boundary == 'BS':
                boundary_entries_index = (subset_df['SW'].shift(1) == 1) & (subset_df['SW'] == 0)
                boundary_exits_index = (subset_df['SW'].shift(1) == 0) & (subset_df['SW'] == 1)
                outside_mask = subset_df['SW'] == 1
                inside_mask = (subset_df['SH'] == 1) | (subset_df['MS'] == 1)
            elif boundary == 'MP':
                boundary_entries_index = (subset_df['MS'].shift(1) == 0) & (subset_df['MS'] == 1)
                boundary_exits_index = (subset_df['MS'].shift(1) == 1) & (subset_df['MS'] == 0)
                outside_mask = (subset_df['SW'] == 1) | (subset_df['SH'] == 1)
                inside_mask = subset_df['MS'] == 1
                
            axd[key].scatter(subset_df.loc[boundary_entries_index, 'ell'],
                             subset_df.loc[boundary_entries_index, 'rho'],
                             s = 16, color='#0001a7', marker='x', lw = 1, zorder=32)
            axd[key].scatter(subset_df.loc[boundary_exits_index, 'ell'],
                             subset_df.loc[boundary_exits_index, 'rho'],
                             s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 1, zorder=32)
    
    for key in ['a', 'b', 'c', 'd']:
        axd[key].set_xticks(axd[key].get_xticks(), labels=['',]*len(axd[key].get_xticks()))
    for key in ['b', 'd', 'f']:
        axd[key].set_yticks(axd[key].get_yticks(), labels=['',]*len(axd[key].get_yticks()))
        
    # Plot dummy lines for the legend
    for (_, p_dyn_row), (_, p_row) in zip(p_dyn_df.iterrows(), p_df.iterrows()):
        axd['a'].plot([0,0], [-1,-1], color=p_dyn_row['color'],
                      lw = 3, ls = '-',
                      label = r'${:.0f}^{{th}}$ %ile $p_{{SW}}$'.format(p_dyn_row['percentile']))
        axd['a'].plot([0,0], [-1,-1], color='black',
                      ls = p_row['ls'], lw=3, 
                      label = p_row['location'])
    axd['a'].legend(bbox_to_anchor=(0, 1.04, 2.04, 0.15), 
                    ncol=3, handlelength=4,
                    loc="lower left", borderaxespad=0, mode='expand',
                    fontsize='large')
    
    plt.savefig('/Users/mrutala/projects/JupiterBoundaries/paper/figures/{}Demonstration.png'.format(boundary))
    plt.show()
    # =============================================================================
    #     Residual plot for frequentized data
    # =============================================================================
    fig, axd = plt.subplot_mosaic("""
                                 gg
                                 hh
                                 """, height_ratios=[5, 1], figsize = (6, 3))
    plt.subplots_adjust(bottom = 0.50/3, left = 0.50/6, top = 1 - 0.25/3, right = 1 - 0.125/6,
                        wspace = 0.50/6, hspace = 0.75/8)
    
    for letter, (_, ax) in zip(['a', 'b'], axd.items()):
        ax.annotate('({})'.format(letter), (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    ha = 'center', va = 'center')
        
    # filter for crossings or near-crossings
    from scipy.stats import gaussian_kde
    
    crossing_positions_df = postproc.balance_Classes(positions_df, boundary, dt.timedelta(minutes=1), other_fraction=0.0)
    
    def kde_mean_std(x, y):
        from scipy import integrate
        mean = integrate.cumulative_trapezoid(x * y, x)[-1]
        std = integrate.cumulative_trapezoid(x**2 * y, x)[-1] - mean**2
        
        return mean, std
    
    def kde_percentiles(x, y):
        from scipy import integrate
        running_values = integrate.cumulative_trapezoid(y, x)
        total = running_values[-1]
        
        ptile_25 = np.interp(25, 100*running_values/total, (x[1:] + x[:-1])/2)
        ptile_50 = np.interp(50, 100*running_values/total, (x[1:] + x[:-1])/2)
        ptile_75 = np.interp(75, 100*running_values/total, (x[1:] + x[:-1])/2)
        
        return ptile_25, ptile_50, ptile_75
    
    stats_dict = {'log_mu': [],
                  'log_sigma': []}
    
    # fig, axs = plt.subplots(figsize=(6.5, 4), nrows=2, height_ratios=[7,1], sharex=True)
    # plt.subplots_adjust(left=0.06, right=0.985, bottom=0.1, top=0.975, hspace=0.04)
    # bins = np.arange(-0.5, 0.5, 0.005)
    bins = np.logspace(-1, 1, 1000)
    log_bins = np.linspace(-1, 1, 1000)
    
    rugplot_y = 1
    colors = ['C0', 'C1', 'C2', 'C3', 'C5']
    for (_, row), this_color in zip(df.iterrows(), colors):
        coords = [crossing_positions_df['t'].to_numpy(),
                  crossing_positions_df['p'].to_numpy(),
                  crossing_positions_df['p_dyn'].to_numpy()]
        
        r_nominal_boundary, weights = row['posterior_fn'](coords, average=True, return_weights=True)
        
        # residual = [r - crossing_positions_df['r'].to_numpy() for r in r_nominal_boundary]
        fraction = [crossing_positions_df['r'].to_numpy() / r for r in r_nominal_boundary]
        
        # Calculate overall weighted kde for each model
        weights_arr = np.array([[w]*len(f) for f, w in zip(fraction, weights)]).flatten()
        kde = gaussian_kde(np.array(fraction).flatten(), 0.2, weights_arr)
        log_kde = gaussian_kde(np.log10(fraction).flatten(), 0.2, weights_arr)
        
        axd['g'].plot(bins, kde(bins), label = row['label'], lw=2, color=this_color)
        # this_color = ax.get_lines()[-1].get_c() # last line's color
        print('+'*42)
        print(row['label'] + ' overall stats:')
        mu, sig = kde_mean_std(bins, kde(bins))
        log_mu, log_sig = kde_mean_std(log_bins, log_kde(log_bins))
        axd['g'].scatter(10**log_mu, kde(10**log_mu), 
                         marker='X', color=this_color, facecolor='white', s=32, 
                         zorder=10)
        print('R_mean: {:.3f} + {:.3f} - {:.3f}'.format(10**log_mu, 10**(log_mu + log_sig) - 10**log_mu, 10**log_mu - 10**(log_mu - log_sig)))
        p25, p50, p75 = kde_percentiles(bins, kde(bins))
        axd['g'].scatter(p50, kde(p50), 
                       marker='o', color=this_color, facecolor='white', s=32, 
                       zorder=10)
        print('R(50 %ile): {:.3f}; R(25 %ile): {:.3f}; R(75 %ile): {:.3f}'.format(p50, p25, p75))
        most_probable = bins[np.argmax(kde(bins))]
        print('Most Probable R: {:.3f}'.format(most_probable))
        axd['g'].scatter(most_probable, kde(most_probable),
                       marker = '^', color=this_color, facecolor='white', s=32, 
                       zorder = 10)
        print('='*42)
        
        print(' ')
        stats_dict['log_mu'].append(mu)
        stats_dict['log_sigma'].append(sig)
        
        # Plot the rugplot
        axd['h'].scatter(fraction, np.full(np.array(fraction).shape, rugplot_y), marker='|', alpha=0.2, s=16, color=this_color)
        rugplot_y -= 1/(len(df))
        
        # Calculate kde for each mode
        # if len(fraction) > 1:
        # for mode, frac_by_mode in enumerate(fraction):
        #     # h = ax.hist(np.log10(frac_by_mode), bins = bins,
        #     #             label = row['label'], histtype='step', density=True, lw=2)
            
        #     dashes = [4, 2]
        #     dashes.extend((mode) * [1, 2])
            
        #     kde = gaussian_kde(np.log10(frac_by_mode), 0.2)
        #     axd['g'].plot(bins, kde(bins), lw=2, color=this_color, alpha = 1, dashes=dashes)
            
        #     axd['h'].scatter(np.log10(frac_by_mode), np.full(frac_by_mode.shape, rugplot_y), marker='|', alpha=0.2, s=16, color=this_color)
        
        #     rugplot_y -= 1/(len(df)+2)
            
        #     print('mode ' + str(mode) + ' stats:')
        #     mu, sig = kde_mean_std(bins, kde(bins))
        #     axd['g'].scatter(mu, kde(mu), 
        #                marker='X', color=this_color, facecolor='white', s=32, 
        #                zorder=10)
        #     print('R_mean: {:.3f} + {:.3f} - {:.3f}'.format(10**mu, 10**(mu + sig) - 10**mu, 10**mu - 10**(mu - sig)))
        #     p25, p50, p75 = kde_percentiles(bins, kde(bins))
        #     axd['g'].scatter(p50, kde(p50), 
        #                marker='o', color=this_color, facecolor='white', s=32, 
        #                zorder=10)
        #     print('R(50 %ile): {:.3f}; R(25 %ile): {:.3f}; R(75 %ile): {:.3f}'.format(10**p50, 10**p25, 10**p75))
        #     most_probable = bins[np.argmax(kde(bins))]
        #     print('Most Probable R: {:.3f}'.format(10**most_probable))
        #     axd['g'].scatter(most_probable, kde(most_probable),
        #                    marker = '^', color=this_color, facecolor='white', s=32, 
        #                    zorder = 10)
        #     print(' ')
        # print('-'*42)
    
    axd['g'].scatter([0], [-2], marker='X', color='black', facecolor='white', s=32,
               label = 'Mean R')
    axd['g'].scatter([0], [-2], marker='o', color='black', facecolor='white', s=32,
               label = 'Median R')
    axd['g'].scatter([0], [-2], marker='^', color='black', facecolor='white', s=32,
               label = 'Most Probable R')
    axd['g'].legend()
    
    
    for key in ['g', 'h']:
        axd[key].axvline(1, linestyle='--', color='black', linewidth=1, zorder=-2)
        axd[key].set(xscale = 'log', xlim = [0.1, 10])
    axd['h'].set(xlabel = r"$R = r_{b}^{obs}/r_{b}^{model}$",
                 ylabel = "", ylim=[0,1+1/(len(df)+2)])
    axd['h'].set_yticks([])
    
    axd['g'].set(ylabel = r"Probability Density", ylim = [-0.2, 2])
    axd['g'].set_xticks([0.1, 1, 10], labels=['', '', '']) # Doing it more elegantly didn't work
        
    # for key in ['g', 'h']:
    #     bbox = axd[key].get_position()
    #     bbox.x1 = axd['f'].get_position().x1
    #     axd[key].set_position(bbox)    
    #     axd[key].set_xlabel(axd[key].get_xlabel(), size='large')
    #     axd[key].set_ylabel(axd[key].get_ylabel(), size='large')
    
    plt.savefig('/Users/mrutala/projects/JupiterBoundaries/paper/figures/{}Comparison.png'.format(boundary))
    plt.show()
    
    breakpoint() 
    
    print("+"*42)
    for indx, row in df.iterrows():
        print('Overall Score for {}:'.format(row['label']))
        overall_score = np.abs(stats_dict['log_mu'][indx]/np.std(stats_dict['log_mu'])) + (stats_dict['log_sigma'][indx]/np.mean(stats_dict['log_sigma'])) 
        print('{:.3f}'.format(overall_score))
        print('='*42)
    print("-"*42)
    
    return

def PosteriorAnalysis_For_SI(boundary):
    
    basedir = '/Users/mrutala/projects/JupiterBoundaries/posteriors/'
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    
    # Setup a dataframe to hold the different models
    df = pd.DataFrame(columns = ['label', 'model_name', 'posterior_path', 'posterior_fn', 'inference_path'])
    
    # Add the basic J02 model
    J02_params = JBC.get_JoyParameters(boundary)
    J02_keylist = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1', 'F0', 'F1']
    J02_comparison_df = pd.DataFrame([J02_params], columns=J02_keylist, index=[0])
    J02_comparison_df['sigma_m'] = 0.001 #  0.095 # From Joy+ 2002
    J02_comparison_df['sigma_b'] = 0.001 #  8 # From Joy+ 2002
    J02_fn = make_BoundaryFunctionFromPosterior('Joylike_r1fixed', J02_comparison_df)
    df.loc[len(df)] = ['J02 (original)', 'Joylike_r1fixed', 'NULL', J02_fn, 'NULL']
    
    if boundary == 'MP':
        
        # Pre-Cassini MP inclusion
        # df.loc[len(df)] = [r'S97 (unimodal)', 'Shuelike_r1fixed',
        #                    basedir + '20241209145627/Shuelike_r1fixed-MP_20241209145627_posterior.pkl', '']
        # df.loc[len(df)] = [r'New (unimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + '20241209150506/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20241209150506_posterior.pkl', '']
        # df.loc[len(df)] = [r'S97 (bimodal)', 'Shuelike_r1fixed',
        #                    basedir + '20241209160253/Shuelike_r1fixed-MP_20241209160253_posterior.pkl', '']
        # df.loc[len(df)] = [r'New (bimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + '20241209200347/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20241209200347_posterior.pkl', '']

        # Post-Cassini MP inclusion
        df.loc[len(df)] = [r'S97 (unimodal)', 'Shuelike_r1fixed',
                           basedir + 'MP_S97/Shuelike_r1fixed-MP_20250202181534_posterior.pkl', '',
                           basedir + 'MP_S97/Shuelike_r1fixed-MP_20250202181534_inferencedata.nc']
                    
        # df.loc[len(df)] = [r'New (unimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + 'MP2/ShuelikeAsymmetric_AsPerturbation_r1fixed-MP_20250127155253_posterior.pkl', '']
        # df.loc[len(df)] = [r'New (unimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + '20250130191123/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250130191123_posterior.pkl', '']
        df.loc[len(df)] = [r'New (unimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
                           basedir + 'MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_posterior.pkl', '',
                           basedir + 'MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_inferencedata.nc']
        # df.loc[len(df)] = [r'S97 (bimodal)', 'Shuelike_r1fixed',
        #                    basedir + 'MP3/Shuelike_r1fixed-MP_20250123143444_posterior.pkl', '']
        # df.loc[len(df)] = [r'New (bimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + 'MP4/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250122092819_posterior.pkl', '']

    elif boundary == 'BS':
        df.loc[len(df)] = [r'S97 (unimodal)', 'Shuelike_r1fixed',
                           basedir + 'BS_S97/Shuelike_r1fixed-BS_20250131181629_posterior.pkl', '']
        df.loc[len(df)] = [r'New (unimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
                           basedir + 'BS_new/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20250131195853_posterior.pkl', '']
        # df.loc[len(df)] = [r'S97 (bimodal)', 'Shuelike_r1fixed',
        #                    basedir + 'BS3/Shuelike_r1fixed-BS_20241210101958_posterior.pkl', '']
        # df.loc[len(df)] = [r'New (bimodal)', 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2',
        #                    basedir + 'BS4/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20241210103010_posterior.pkl', '']
        
    # Iterate over each model (other than J02), adding a function for calculating the boundaries
    for index, row in df.loc[1:].iterrows():
        posterior_df = read_PosteriorAsDataFrame(row['posterior_path'])
        fn = make_BoundaryFunctionFromPosterior(row['model_name'], posterior_df)
        df.loc[index, 'posterior_fn'] = fn
    
    positions_df = get_PositionsForAnalysis()
    positions_df = positions_df.query("UN != 1")
    
    # Add the bounds to each entry in the df
    bound_params = postproc.find_BoundarySurfaceLimits(positions_df, boundary, spacecraft_to_use)
    positions_df = postproc.add_Bounds(positions_df, boundary, bound_params['upperbound'], bound_params['lowerbound'])
    
    # Print the confusion matrix statistics
    for _, row in df.iterrows():
        print(row['label'])
        WithinBounds(boundary, row['model_name'], row['posterior_fn'], positions_df)
        
        # ConfusionMatrix(boundary, row['model_name'], row['posterior_fn'], positions_df)
        
        # GuessCrossings(boundary, row['model_name'], row['posterior_fn'], positions_df)
        # ConfusionMatrix(boundary, row['model_name'], row['posterior_path'])
    
    # COMPARE BAYESIAN MODELS
    bayesian_comparison_dict = {}
    for _, row in df.iterrows():
        if row['inference_path'] != 'NULL':
            
            # We have to remove the p_dyn_draw log likelihood-- 
            # WAIC can only be computed for one log likelihood (at a time)
            temp_idata = az.from_netcdf(row['inference_path'])
            temp_idata.log_likelihood = temp_idata.log_likelihood.drop_vars(['p_dyn_draw', 'p_dyn_draw_dim_0'])
            bayesian_comparison_dict[row['label']] = temp_idata
            
    test = az.compare(bayesian_comparison_dict, 'waic')

    breakpoint()    
        
    p_dyn_df = pd.DataFrame(columns = ['percentile', 'value', 'color'])
    p_dyn_df['percentile'] = [90, 50, 10]
    p_dyn_df['value'] = np.percentile(positions_df['p_dyn'], p_dyn_df['percentile'])
    p_dyn_df['color'] = ['C3', 'C1', 'C0']
    
    p_df = pd.DataFrame(columns = ['location', 'angle', 'ls'])
    p_df['location'] = ['Polar', 'Dawn', 'Dusk']
    p_df['angle'] = [0, np.pi/2, -np.pi/2]
    p_df['ls'] = ['-', '-.', ':']
    
    # Print the polar flattening and dawn/dusk ratios
    coords_north = [np.pi/2, 0, p_dyn_df.query('percentile == 50')['value']]
    coords_dawn = [np.pi/2, np.pi/2., p_dyn_df.query('percentile == 50')['value']]
    coords_dusk = [np.pi/2, -np.pi/2., p_dyn_df.query('percentile == 50')['value']]
    for _, row in df.iterrows():
        # r_north = np.array(get_BoundariesFromPosterior(row['model_name'], coords_north, row['posterior_path'], n=100)).flatten()
        # r_dawn = np.array(get_BoundariesFromPosterior(row['model_name'], coords_dawn, row['posterior_path'], n=100)).flatten()
        # r_dusk = np.array(get_BoundariesFromPosterior(row['model_name'], coords_dusk, row['posterior_path'], n=100)).flatten()
        
        r_north = np.array(row['posterior_fn'](coords_north, n=100)).flatten()
        r_dawn = np.array(row['posterior_fn'](coords_dawn, n=100)).flatten()
        r_dusk = np.array(row['posterior_fn'](coords_dusk, n=100)).flatten()
        
        print(row['label'])
        mu_flattening = (np.mean(r_dawn) - np.mean(r_north)) / np.mean(r_dawn)
        sig_flattening = mu_flattening * np.sqrt((np.sqrt(np.std(r_dawn)**2 + np.std(r_north)**2)/(np.mean(r_dawn) - np.mean(r_north)))**2 + (np.std(r_dawn)/np.mean(r_dawn))**2)
        print("Dawn-to-Polar Ratio: {:.3f} +/- {:.3f}".format(mu_flattening, sig_flattening))
        mu_flattening = (np.mean(r_dusk) - np.mean(r_north)) / np.mean(r_dusk)
        sig_flattening = mu_flattening * np.sqrt((np.sqrt(np.std(r_dusk)**2 + np.std(r_north)**2)/(np.mean(r_dusk) - np.mean(r_north)))**2 + (np.std(r_dusk)/np.mean(r_dusk))**2)
        print("Dusk-to-Polar Ratio: {:.3f} +/- {:.3f}".format(mu_flattening, sig_flattening))
        print(" ")
    
    # Plot the boundaries:
    fig, axs = plt.subplots(nrows=len(df), ncols=2, figsize=(6.5, 8.5),
                            sharex=True, sharey=True)
    plt.subplots_adjust(bottom = 0.06, left = 0.12, top = 0.925, right = 0.975,
                        wspace = 0.15, hspace = 0.05)
    
    axs_tl1 = np.array(axs[0,0].get_position())[0, 0], np.array(axs[0,0].get_position())[1, 1]
    axs_tl2 = np.array(axs[0,-1].get_position())[0, 0], np.array(axs[0,-1].get_position())[1,1]
    axs_br = np.array(axs[-1,-1].get_position())[1, 0], np.array(axs[-1,-1].get_position())[0,1]
    
    x_label_pos = (axs_tl1[0] + axs_br[0])/2, axs_br[1]/3
    y_label1_pos = axs_tl1[0]/5, (axs_tl1[1] + axs_br[1])/2
    y_label2_pos = axs_tl2[0] - axs_tl1[0]/5 , (axs_tl2[1] + axs_br[1])/2
    
    fig.text(*x_label_pos, r'$x_{JSS}$ $[R_J]$ (+ Sunward)', 
             fontsize='large',
             ha = 'center', va='center')
    fig.text(*y_label1_pos, r'$\rho = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ $[R_J]$', 
             fontsize='large', rotation=90,
             ha = 'center', va='center')
    fig.text(*y_label2_pos, r'$z_{JSS} [R_J]$ (+ Northward)', 
             fontsize='large', rotation=90,
             ha = 'center', va='center')
    
    # fig.supxlabel(r'$x_{JSS}$ (+ Sunward)')
    # fig.supylabel(r'$z_{JSS}$ (+ Northward)')
    
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', ' i', ' j', 'k', 'l']
    for i, ax in enumerate(axs.flatten()):
        ax.set(aspect = 1,
               xlim = [200, -200], # [200, -400],
               ylim = [0, 220])    #, [0, 350])
        ax.annotate('({})'.format(letters[i]), (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    ha = 'center', va = 'center')
    
    for (_, row), ax_row0 in zip(df.iterrows(), axs[:,0]):
        for _, p_dyn_row in p_dyn_df.iterrows():
            for _, p_row in p_df.iterrows():
                # Get coordinates
                t = np.linspace(0, 0.975 * np.pi, 1000)
                p = np.zeros(1000) + p_row['angle']
                p_dyn = np.zeros(1000) + p_dyn_row['value']
                
                # Plot J02 original for comparison
                # J02 Original boundary
                # r_J02 = get_BoundariesFromPosterior('Joylike_r1fixed', [t, p, p_dyn], J02_comparison)
                # rpl_J02 = BM.convert_SphericalSolarToCylindricalSolar(r_J02[0], t, p)
                # ax_row0.plot(rpl_J02[2], rpl_J02[0], color='black', alpha=0.5)
                
                # rs may have multiple modes, so loop over each
                rs = row['posterior_fn']([t, p, p_dyn], average=True)
                for r in rs:
                    rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                    ax_row0.plot(rpl[2], rpl[0], 
                                 color=p_dyn_row['color'], ls=p_row['ls'], lw=2)
        ax_row0.set(ylabel = row['label'])
            
    # In these plots, show uncertainties + data, for reference        
    for (_, row), ax_row1 in zip(df.iterrows(), axs[:,1]):
        # uncertainties_lines = []
        for _, p_dyn_row in p_dyn_df.iterrows():
            # Nominal coordinate values
            t = np.linspace(0, 0.975 * np.pi, 1000)
            p = np.zeros(1000) + 0
            p_dyn = np.zeros(1000) + p_dyn_row['value']
            
            rs = row['posterior_fn']([t, p, p_dyn], n=500)
            for r in rs:
                rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                ax_row1.plot(rpl[2], rpl[0], 
                              color=p_dyn_row['color'], lw = 1.5, alpha=0.02, zorder=-2)
                # uncertainties_lines.append([rpl[2], rpl[0]])
            
            rs = row['posterior_fn']([t, p, p_dyn], average=True)
            for r in rs:
                rpl = BM.convert_SphericalSolarToCylindricalSolar(r, t, p)
                ax_row1.plot(rpl[2], rpl[0], 
                             color='black', lw = 2.4, alpha=1, zorder=16)
                ax_row1.plot(rpl[2], rpl[0], 
                             color=p_dyn_row['color'], lw=2, alpha=1, zorder=16)
                
        # # !!! THIS DIDN'T WORK- would need to have color info too
        # # Shuffle the list of uncertainty lines
        # # This prevents the last set of lines from overwriting others
        # rng = np.random.default_rng()
        # uncertainties_lines_shuffled = rng.choice(uncertainties_lines, len(uncertainties_lines), replace=False)
        # for uls in uncertainties_lines_shuffled:
        #     ax_row1.plot(uls[0], uls[1], 
        #                  color=p_dyn_row['color'], lw = 1.5, alpha=0.02, zorder=-2)
            
        # =============================================================================
        #         # Rotate data to North
        # =============================================================================
        # 1) Where is the nominal boundary compared to the point?
        # r_boundary_at_point = get_BoundariesFromPosterior(row['model_name'], 
        #                                                   positions_df[['t', 'p', 'p_dyn']].to_numpy().T, 
        #                                                   row['posterior_path'],
        #                                                   mean=True)
        r_boundary_at_point = row['posterior_fn'](positions_df[['t', 'p', 'p_dyn']].to_numpy().T, average=True)
        # delta_r = [positions_df['r'].to_numpy() - r_at_point for r_at_point in r_boundary_at_point]
        fraction_r = [positions_df['r'].to_numpy() / r_at_point for r_at_point in r_boundary_at_point]
        
        # 2) Preserve delta_r compared to the Dawn crossing
        t_rotated = positions_df['t'].to_numpy()
        p_rotated = np.full(np.shape(t_rotated), 0) # Forced to North
        p_dyn_rotated = positions_df['p_dyn'].to_numpy()
        
        # r_rotated = get_BoundariesFromPosterior(row['model_name'], 
        #                                         [t_rotated, p_rotated, p_dyn_rotated], 
        #                                         row['posterior_path'],
        #                                         mean=True)
        r_rotated = row['posterior_fn']([t_rotated, p_rotated, p_dyn_rotated], average=True)
        # r_rotated = [r + dr for r, dr in zip(r_rotated, delta_r)]
        r_rotated = [r * fr for r, fr in zip(r_rotated, fraction_r)]
        
        # 3) Reassign to df
        rotated_df = positions_df.copy(deep=True)
        rotated_df['p'] = p_rotated
        rotated_df['r'] = r_rotated[0] # 0 is the 'default' mode
        
        xyz = BM.convert_SphericalSolarToCartesian(*rotated_df[['r', 't', 'p']].to_numpy().T)
        rotated_df.loc[:, ['x', 'y', 'z']] = xyz.T
        rpl = BM.convert_SphericalSolarToCylindricalSolar(*rotated_df[['r', 't', 'p']].to_numpy().T)
        rotated_df.loc[:, ['rho', 'phi', 'ell']] = rpl.T
        
        # 4) Plot the actual crossings
        for spacecraft in spacecraft_to_use:
            subset_df = rotated_df.query('spacecraft == @spacecraft')
            # subset_df = positions_df.query('spacecraft == @spacecraft')
            
            if boundary == 'BS':
                boundary_entries_index = (subset_df['SW'].shift(1) == 1) & (subset_df['SW'] == 0)
                boundary_exits_index = (subset_df['SW'].shift(1) == 0) & (subset_df['SW'] == 1)
                outside_mask = subset_df['SW'] == 1
                inside_mask = (subset_df['SH'] == 1) | (subset_df['MS'] == 1)
            elif boundary == 'MP':
                boundary_entries_index = (subset_df['MS'].shift(1) == 0) & (subset_df['MS'] == 1)
                boundary_exits_index = (subset_df['MS'].shift(1) == 1) & (subset_df['MS'] == 0)
                outside_mask = (subset_df['SW'] == 1) | (subset_df['SH'] == 1)
                inside_mask = subset_df['MS'] == 1
                
            ax_row1.scatter(subset_df.loc[boundary_entries_index, 'ell'],
                            subset_df.loc[boundary_entries_index, 'rho'],
                            s = 16, color='#0001a7', marker='x', lw = 1, zorder=32)
            ax_row1.scatter(subset_df.loc[boundary_exits_index, 'ell'],
                            subset_df.loc[boundary_exits_index, 'rho'],
                            s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 1, zorder=32)
            
        
        # !!!!!
        # r_SS (from model) as a function of p_dyn, as a 2D colored array
        # Overplot, as points, the calculated r_SS and measured p_dyn for each spacecraft crossing?

        # breakpoint() # What does plotting delta_r look like?
    
    # Plot dummy lines for the legend
    # for _, p_dyn_row in p_dyn_df.iterrows():
    #     axs[0,0].plot([0,0], [-1,-1], color=p_dyn_row['color'],
    #                   lw = 2, ls = '-',
    #                   label = r'${:.0f}^{{th}}$ %ile $p_{{dyn}}$'.format(p_dyn_row['percentile']))
    # for _, p_row in p_df.iterrows(): 
    #     axs[0,0].plot([0,0], [-1,-1], color='black',
    #                   ls = p_row['ls'], lw=2, 
    #                   label = p_row['location'])
    for (_, p_dyn_row), (_, p_row) in zip(p_dyn_df.iterrows(), p_df.iterrows()):
        axs[0,0].plot([0,0], [-1,-1], color=p_dyn_row['color'],
                      lw = 3, ls = '-',
                      label = r'${:.0f}^{{th}}$ %ile $p_{{SW}}$'.format(p_dyn_row['percentile']))
        axs[0,0].plot([0,0], [-1,-1], color='black',
                      ls = p_row['ls'], lw=3, 
                      label = p_row['location'])
    axs[0,0].legend(bbox_to_anchor=(0, 1.05, 2.15, 0.2), 
                    ncol=3, handlelength=5,
                    loc="lower left", borderaxespad=0, mode='expand',
                    fontsize='large')

    
    plt.show()
    
    # =============================================================================
    #     Residual plot, to be spun off probably
    # =============================================================================
    # filter for crossings or near-crossings
    from scipy.stats import gaussian_kde
    
    crossing_positions_df = postproc.balance_Classes(positions_df, boundary, dt.timedelta(minutes=1), other_fraction=0.0)
    
    def kde_mean_std(x, y):
        from scipy import integrate
        mean = integrate.cumulative_trapezoid(x * y, x)[-1]
        std = integrate.cumulative_trapezoid(x**2 * y, x)[-1] - mean**2
        
        return mean, std
    
    def kde_percentiles(x, y):
        from scipy import integrate
        running_values = integrate.cumulative_trapezoid(y, x)
        total = running_values[-1]
        
        ptile_25 = np.interp(25, 100*running_values/total, (x[1:] + x[:-1])/2)
        ptile_50 = np.interp(50, 100*running_values/total, (x[1:] + x[:-1])/2)
        ptile_75 = np.interp(75, 100*running_values/total, (x[1:] + x[:-1])/2)
        
        return ptile_25, ptile_50, ptile_75
    
    stats_dict = {'log_mu': [],
                  'log_sigma': []}
    
    fig, axs = plt.subplots(figsize=(6.5, 4), nrows=2, height_ratios=[7,1], sharex=True)
    plt.subplots_adjust(left=0.06, right=0.985, bottom=0.1, top=0.975, hspace=0.04)
    bins = np.arange(-0.5, 0.5, 0.005)
    axs[1].set(xlabel = r"$log_{10}(R) = log_{10}(r_{b, data}/r_{b, model})$", xlim = [-0.5, 0.5],
               ylabel = "", ylim=[0,1+1/(len(df)+2)])
    axs[1].set_yticks([])
    axs[0].set(ylabel = r"Probability Density", ylim = [-0.6, 4])
    
    axs[0].annotate('(a)', (0,1), (1,-1), 
                    xycoords='axes fraction', textcoords='offset fontsize',
                    va='center', ha='center')
    axs[1].annotate('(b)', (0,1), (1,-1), 
                    xycoords='axes fraction', textcoords='offset fontsize',
                    va='center', ha='center')
    
    for ax in axs:
        ax.axvline(0, linestyle='--', color='black', linewidth=1)
    # ax.set(xscale = 'log', xlim = [0.01,100])
    
    rugplot_y = 1
    colors = ['C0', 'C1', 'C2', 'C3', 'C5']
    for (_, row), this_color in zip(df.iterrows(), colors):
        coords = [crossing_positions_df['t'].to_numpy(),
                  crossing_positions_df['p'].to_numpy(),
                  crossing_positions_df['p_dyn'].to_numpy()]
        
        r_nominal_boundary, weights = row['posterior_fn'](coords, average=True, return_weights=True)
        
        # residual = [r - crossing_positions_df['r'].to_numpy() for r in r_nominal_boundary]
        fraction = [crossing_positions_df['r'].to_numpy() / r for r in r_nominal_boundary]
        
        # Calculate overall weighted kde for each model
        weights_arr = np.array([[w]*len(f) for f, w in zip(fraction, weights)]).flatten()
        kde = gaussian_kde(np.log10(fraction).flatten(), 0.2, weights_arr)
        axs[0].plot(bins, kde(bins), label = row['label'], lw=2, color=this_color)
        # this_color = ax.get_lines()[-1].get_c() # last line's color
        print('+'*42)
        print(row['label'] + ' overall stats:')
        mu, sig = kde_mean_std(bins, kde(bins))
        axs[0].scatter(mu, kde(mu), 
                       marker='X', color=this_color, facecolor='white', s=32, 
                       zorder=10)
        print('R_mean: {:.3f} + {:.3f} - {:.3f}'.format(10**mu, 10**(mu + sig) - 10**mu, 10**mu - 10**(mu - sig)))
        p25, p50, p75 = kde_percentiles(bins, kde(bins))
        axs[0].scatter(p50, kde(p50), 
                       marker='o', color=this_color, facecolor='white', s=32, 
                       zorder=10)
        print('R(50 %ile): {:.3f}; R(25 %ile): {:.3f}; R(75 %ile): {:.3f}'.format(10**p50, 10**p25, 10**p75))
        most_probable = bins[np.argmax(kde(bins))]
        print('Most Probable R: {:.3f}'.format(10**most_probable))
        axs[0].scatter(most_probable, kde(most_probable),
                       marker = '^', color=this_color, facecolor='white', s=32, 
                       zorder = 10)
        print('='*42)
        
        print(' ')
        stats_dict['log_mu'].append(mu)
        stats_dict['log_sigma'].append(sig)
        
        # Calculate kde for each mode
        # if len(fraction) > 1:
        for mode, frac_by_mode in enumerate(fraction):
            # h = ax.hist(np.log10(frac_by_mode), bins = bins,
            #             label = row['label'], histtype='step', density=True, lw=2)
            
            dashes = [4, 2]
            dashes.extend((mode) * [1, 2])
            
            kde = gaussian_kde(np.log10(frac_by_mode), 0.2)
            axs[0].plot(bins, kde(bins), lw=2, color=this_color, alpha = 1, dashes=dashes)
            
            axs[1].scatter(np.log10(frac_by_mode), np.full(frac_by_mode.shape, rugplot_y), marker='|', alpha=0.2, s=16, color=this_color)
        
            rugplot_y -= 1/(len(df)+2)
            
            print('mode ' + str(mode) + ' stats:')
            mu, sig = kde_mean_std(bins, kde(bins))
            axs[0].scatter(mu, kde(mu), 
                       marker='X', color=this_color, facecolor='white', s=32, 
                       zorder=10)
            print('R_mean: {:.3f} + {:.3f} - {:.3f}'.format(10**mu, 10**(mu + sig) - 10**mu, 10**mu - 10**(mu - sig)))
            p25, p50, p75 = kde_percentiles(bins, kde(bins))
            axs[0].scatter(p50, kde(p50), 
                       marker='o', color=this_color, facecolor='white', s=32, 
                       zorder=10)
            print('R(50 %ile): {:.3f}; R(25 %ile): {:.3f}; R(75 %ile): {:.3f}'.format(10**p50, 10**p25, 10**p75))
            most_probable = bins[np.argmax(kde(bins))]
            print('Most Probable R: {:.3f}'.format(10**most_probable))
            axs[0].scatter(most_probable, kde(most_probable),
                           marker = '^', color=this_color, facecolor='white', s=32, 
                           zorder = 10)
            print(' ')
        print('-'*42)
    
    axs[0].scatter([0], [-2], marker='X', color='black', facecolor='white', s=32,
               label = 'Mean Ratio (per model)')
    axs[0].scatter([0], [-2], marker='o', color='black', facecolor='white', s=32,
               label = 'Median Ratio (per model)')
    axs[0].scatter([0], [-2], marker='^', color='black', facecolor='white', s=32,
               label = 'Most Probable Ratio (per model)')
    axs[0].legend()
    plt.show()
    
    print("+"*42)
    for indx, row in df.iterrows():
        print('Overall Score for {}:'.format(row['label']))
        overall_score = np.abs(stats_dict['log_mu'][indx]/np.std(stats_dict['log_mu'])) + (stats_dict['log_sigma'][indx]/np.mean(stats_dict['log_sigma'])) 
        print('{:.3f}'.format(overall_score))
        print('='*42)
    print("-"*42)
    
    breakpoint()
    # =============================================================================
    # Plot r between upper and lower bounds
    # =============================================================================
    
    for _, row in df.iterrows():
        fig, ax = plt.subplots()
        ax.set(yscale='log')
        
        ax.fill_between(np.arange(len(positions_df)), positions_df['r_lowerbound'], positions_df['r_upperbound'], 
                        color = 'C2', alpha=0.5)
        
        coords = positions_df.loc[:, ['t', 'p', 'p_dyn']].to_numpy().T
        rs = row['posterior_fn'](coords, average=True)
        
        for mode, r in enumerate(rs):
            ax.scatter(np.arange(len(r)), r, marker='.', 
                       label = "{}: mode {}".format(row['label'], mode))
        
        ax.legend()
        
        plt.show()
    
    breakpoint()
    return



def CheckOverlap(boundary, posterior_path, model_name):
    
    # m = BM.init(model_name)
    p = read_PosteriorFromFile(posterior_path)
    
    positions_df = get_PositionsForAnalysis()
    positions_df = positions_df.query("UN != 1")
    
    # Add the bounds to each entry in the df
    bound_params = postproc.find_BoundarySurfaceLimits(positions_df, boundary, ['Ulysses', 'Galilo', 'Cassini', 'Juno'])
    positions_df = postproc.add_Bounds(positions_df, boundary, bound_params['upperbound'], bound_params['lowerbound'])
        
    result = _CheckOverlap(p, model_name, positions_df)
    
    return

def _CheckOverlap(posterior, model_name, positions_df):
    
    #
    f = make_BoundaryFunctionFromPosterior(model_name, posterior, n=1000)
    rs = f(positions_df.loc[:, ['t', 'p', 'p_dyn']].to_numpy('float64').T)
    
    number_within_bounds = []
    for r in rs:
        # indx = (positions_df['r_lowerbound'] <= r) & (positions_df['r_upperbound'] >= r)
        test = positions_df.query("r_lowerbound <= @r <= r_upperbound")
        number_within_bounds.append(len(test))
    
    breakpoint()
    
    return
    

def plot_BestFits3D(fixed_p_dyn=0.07):
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    
    BS_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/BS_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20250202214617_posterior.pkl'
    BS_model_name = 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2'
    BS_posterior = read_PosteriorAsDataFrame(BS_posterior_path)
    BS_function = make_BoundaryFunctionFromPosterior(BS_model_name, BS_posterior)
    
    # MP_model_name = 'Shuelike_rasymmetric_simple'    
    MP_model_name = 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2'    
    # MP_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/MP2/ShuelikeAsymmetric_AsPerturbation_r1fixed-MP_20250127155253_posterior.pkl'
    # MP_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/20250130214528/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250130214528_posterior.pkl'
    MP_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_posterior.pkl'
    MP_posterior = read_PosteriorAsDataFrame(MP_posterior_path)
    MP_function = make_BoundaryFunctionFromPosterior(MP_model_name, MP_posterior)
    
    l = 4000
    # sfixed_p_dyn = 0.071
    epsilon = 0.25
    
    # # FOR MP
    # xlim, ylim, zlim = [+150, -150], [+150, -150], [-150, +150]
    # cuts = [-60, -30, 0, 30, 60]
    # norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    
    # FOR BS
    xlim, ylim, zlim = [+250, -250], [+250, -250], [-250, +250]
    cuts = [-120, -60, 0, 60, 120]
    norm = mpl.colors.Normalize(vmin=-200, vmax=200)
    
    cmap = mpl.colormaps['plasma']

    fig, axd = plt.subplot_mosaic([['xy', '3d'],
                                   ['xz', 'yz']],
                                  figsize = (6.5, 5.75))
    fig.delaxes(axd['3d'])
    axd['3d'] = plt.subplot(2, 2, 2, projection='3d')
    plt.subplots_adjust(left = 0.1, bottom = 0.1, top = 0.9625, right = 0.85)
    for _, ax in axd.items():
        ax.set(aspect='equal')
    axd['3d'].set(xlim = [max(xlim), min(xlim)], xlabel = r'$x_{JSS}$',
                  ylim = [min(ylim), max(ylim)], ylabel = r'$y_{JSS}$',
                  zlim = [min(zlim), max(zlim)], zlabel = r'$z_{JSS}$')
    axd['3d'].view_init(elev=30, azim=-135, roll=0)
    axd['3d'].grid(False)
    axd['xy'].set(xlim = [max(xlim), min(xlim)],
                  ylim = [max(ylim), min(ylim)], ylabel = r'$y_{JSS} [R_J]$ (+Duskward)')
    axd['xz'].set(xlim = [max(xlim), min(xlim)], xlabel = r'$x_{JSS} [R_J]$ (+Sunward)',
                  ylim = [min(zlim), max(zlim)], ylabel = r'$z_{JSS} [R_J]$ (+Northward)')
    axd['yz'].set(xlim = [max(ylim), min(ylim)], xlabel = r'$y_{JSS} [R_J]$ (+Duskward)',
                  ylim = [min(zlim), max(zlim)])
    ML = 10
    axd['xy'].xaxis.set_minor_locator(MultipleLocator(ML))
    axd['xz'].xaxis.set_minor_locator(MultipleLocator(ML))
    axd['yz'].xaxis.set_minor_locator(MultipleLocator(ML))
    axd['xy'].yaxis.set_minor_locator(MultipleLocator(ML))
    axd['xz'].yaxis.set_minor_locator(MultipleLocator(ML))
    axd['yz'].yaxis.set_minor_locator(MultipleLocator(ML))
    
    
    # Plot the 3D surface
    t_1d = np.linspace(0, np.radians(175), l)
    p_1d = np.linspace(0, np.radians(360), l)
    t_2d, p_2d = np.meshgrid(t_1d, p_1d)
    p_dyn_2d = np.full(t_2d.shape, fixed_p_dyn)
    
    MP_rs_2d = MP_function([t_2d, p_2d, p_dyn_2d], average=True)
    MP_rs_2d = [r_2d.reshape(l, -1) for r_2d in MP_rs_2d]
    
    BS_rs_2d = BS_function([t_2d, p_2d, p_dyn_2d], average=True)
    BS_rs_2d = [r_2d.reshape(l, -1) for r_2d in BS_rs_2d]
    
    for rs_2d in [BS_rs_2d]:
        xyz_2d = BM.convert_SphericalSolarToCartesian(rs_2d[0], t_2d, p_2d)
    
        # Plot the surface
        X = xyz_2d[0]
        # X_min, X_max = np.min(xl), np.max(xlim)
        # colors = plt.cm.plasma((X-X_min)/(X_max-X_min))
        colors = cmap(norm(X))
        # colors[:, :, 3][X < X_min] = 0
        colors[:, :, 3] = 0.5
        
        xyz_2d_plot = xyz_2d
        xyz_2d_plot[0][(xyz_2d_plot[0] < min(xlim)) | (xyz_2d_plot[0] > max(xlim))] = np.nan
        xyz_2d_plot[1][(xyz_2d_plot[1] < min(ylim)) | (xyz_2d_plot[1] > max(ylim))] = np.nan
        xyz_2d_plot[2][(xyz_2d_plot[2] < min(zlim)) | (xyz_2d_plot[2] > max(zlim))] = np.nan
        surf = axd['3d'].plot_surface(xyz_2d_plot[0], xyz_2d_plot[1], xyz_2d_plot[2],
                                      linewidth=0, antialiased=False, 
                                      vmin = -200, vmax= 300, 
                                      facecolors = colors, shade=False)
        # XY plane, viewed from North
        # t_xy = np.abs(np.linspace(-0.9*np.pi, 0.9*np.pi, 1000))
        # p_xy = np.concatenate([np.full(500, -np.pi/2), np.full(500, np.pi/2)])
        # p_dyn = np.full(1000, fixed_p_dyn)
        # r_xy = MP_function([t_xy, p_xy, p_dyn], average=True)
        # xyz_xy = BM.convert_SphericalSolarToCartesian(r_xy[0], t_xy, p_xy)
        
        for z_slice in cuts:
            color = cmap(norm(z_slice))
            z_indx = np.argwhere((xyz_2d[2] > z_slice - epsilon) &
                                 (xyz_2d[2] < z_slice + epsilon))
            X = xyz_2d[0][z_indx.T[0], z_indx.T[1]]
            Y = xyz_2d[1][z_indx.T[0], z_indx.T[1]]
            order = np.argsort(np.arctan2(-Y, -(X+60)) % (2*np.pi))
            # axd['xy'].scatter(X, Y, marker='.', s=1, color=color)
            axd['xy'].plot(X[order], Y[order], lw=1.5, color=color)
        
        # p_xy = np.concatenate([np.full(500, -np.radians(45)), np.full(500, np.radians(45))])
        # r_xy = MP_function([t_xy, p_xy, p_dyn], average=True)
        # xyz_xy = BM.convert_SphericalSolarToCartesian(r_xy[0], t_xy, p_xy)
        # axd['xy'].plot(xyz_xy[0], xyz_xy[1], lw=2, color='orange')
        
        # XZ plane, viewed from Dusk
        # t_xz = np.abs(np.linspace(-0.9*np.pi, 0.9*np.pi, 1000))
        # p_xz = np.concatenate([np.full(500, 0), np.full(500, np.pi)])
        # r_xz = MP_function([t_xz, p_xz, p_dyn], average=True)
        
        # xyz_xz = BM.convert_SphericalSolarToCartesian(r_xz[0], t_xz, p_xz)
        # axd['xz'].plot(xyz_xz[0], xyz_xz[2], color='black', lw=2)
        
        for y_slice in cuts:
            color = cmap(norm(y_slice))
            y_indx = np.argwhere((xyz_2d[1] > y_slice - epsilon) &
                                 (xyz_2d[1] < y_slice + epsilon))
            X = xyz_2d[0][y_indx.T[0], y_indx.T[1]]
            Z = xyz_2d[2][y_indx.T[0], y_indx.T[1]]
            order = np.argsort(np.arctan2(-Z, -(X+60)) % (2*np.pi))
            axd['xz'].plot(X[order], Z[order], lw=1.5, color=color)
        
        # # YZ plane, viewed from the Sun
        # t_yz = np.full(1000, np.pi/2)
        # p_yz = np.linspace(0, 2*np.pi, 1000)
        # r_yz = MP_function([t_yz, p_yz, p_dyn], average=True)
        
        # xyz_yz = BM.convert_SphericalSolarToCartesian(r_yz[0], t_yz, p_yz)
        # axd['yz'].plot(xyz_yz[1], xyz_yz[2], color='black', lw=2)
        
        for x_slice in cuts:
            color = cmap(norm(x_slice))
            x_indx = np.argwhere((xyz_2d[0] > x_slice - epsilon) &
                                 (xyz_2d[0] < x_slice + epsilon))
            Y = xyz_2d[1][x_indx.T[0], x_indx.T[1]]
            Z = xyz_2d[2][x_indx.T[0], x_indx.T[1]]
            axd['yz'].scatter(Y, Z, marker='.', s=2, color=color)
   
    cax = fig.add_axes([0.87, 0.1, 0.045, 0.8625])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation='vertical', label='Distance out of plane [$R_J$]')
    
    for cut in cuts:
        if len(str(cut)) == 1:
            cut_str = '    {}'.format(cut)
        elif len(str(cut)) ==2:
            cut_str = '  {}'.format(cut)
        else:
            cut_str = '{}'.format(cut)
        cax.annotate(cut_str, (0.5, cut), (0,0), 
                     'data', 'offset pixels', va='center', ha='center',
                     bbox = dict(facecolor=cmap(norm(cut)), edgecolor='black', linewidth=0.5, boxstyle='round'))
    
    letters = ['a', 'b', 'c', 'd']
    for i, ax in enumerate(axd.values()):
        ax.annotate('({})'.format(letters[i]), (0,1), (1,-1), 
                    'axes fraction', 'offset fontsize',
                    ha = 'center', va = 'center')
    
    plt.savefig('/Users/mrutala/projects/JupiterBoundaries/paper/figures/Best.png')
    plt.show()
    
    
   
    breakpoint()
    
    return

def plot_JunoIntersection(fixed_p_dyn=0.07):
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    
    BS_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/BS_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-BS_20250202214617_posterior.pkl'
    BS_model_name = 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2'
    BS_posterior = read_PosteriorAsDataFrame(BS_posterior_path)
    BS_function = make_BoundaryFunctionFromPosterior(BS_model_name, BS_posterior)
    
    # MP_model_name = 'Shuelike_rasymmetric_simple'    
    # MP_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/20241127110105/Shuelike_rasymmetric_simple-MP_20241127110105_posterior.pkl'
    MP_model_name = 'ShuelikeAsymmetric_AsPerturbation_r1fixed_2'    
    MP_posterior_path = '/Users/mrutala/projects/JupiterBoundaries/posteriors/MP_NEW/ShuelikeAsymmetric_AsPerturbation_r1fixed_2-MP_20250202172608_posterior.pkl' 
    MP_posterior = read_PosteriorAsDataFrame(MP_posterior_path)
    MP_function = make_BoundaryFunctionFromPosterior(MP_model_name, MP_posterior)
    
    # Same for Joy+ 2002
    J02_params = JBC.get_JoyParameters('BS')
    J02_keylist = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1', 'F0', 'F1']
    J02_comparison_df = pd.DataFrame([J02_params], columns=J02_keylist, index=[0])
    J02_comparison_df['sigma_m'] = 0.001 #  0.095 # From Joy+ 2002
    J02_comparison_df['sigma_b'] = 0.001 #  8 # From Joy+ 2002
    BS_function_J02 = make_BoundaryFunctionFromPosterior('Joylike_r1fixed', J02_comparison_df)
    
    J02_params = JBC.get_JoyParameters('MP')
    J02_keylist = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1', 'F0', 'F1']
    J02_comparison_df = pd.DataFrame([J02_params], columns=J02_keylist, index=[0])
    J02_comparison_df['sigma_m'] = 0.001 #  0.095 # From Joy+ 2002
    J02_comparison_df['sigma_b'] = 0.001 #  8 # From Joy+ 2002
    MP_function_J02 = make_BoundaryFunctionFromPosterior('Joylike_r1fixed', J02_comparison_df)
    
    # Juno orbit dates
    orb_path = '/Users/mrutala/SPICE/juno/kernels/spk/juno_pred_orbit.orb'
    df = pd.read_table(orb_path, sep = '\s+', 
                       header=None, skiprows=2, 
                       names = ['Orbit', 'AJ_year', 'AJ_month', 'AJ_day', 'AJ_time', 
                                'SCLK', 'PJ_year', 'PJ_month', 'PJ_day', 'PJ_time', 
                                'SolLon', 'SolLat', 'SCLon', 'SCLat', 'Alt', 'SolR'])
    df['AJ_datetime'] = [dt.datetime.strptime('{}-{}-{} {}'.format(row['AJ_year'], row['AJ_month'], row['AJ_day'], row['AJ_time']), 
                                              '%Y-%b-%d %H:%M:%S') for _, row in df.iterrows()]
    df['PJ_datetime'] = [dt.datetime.strptime('{}-{}-{} {}'.format(row['PJ_year'], row['PJ_month'], row['PJ_day'], row['PJ_time']), 
                                              '%Y-%b-%d %H:%M:%S') for _, row in df.iterrows()]
    
    # Get pressure distribution
    sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
    sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1).dropna()
    p_dyn_dist = sw_mme['p_dyn'].to_numpy('float64')
    
    n_pressures = 1000
    rng = np.random.default_rng()
    pressures = rng.choice(p_dyn_dist, n_pressures)
    
    # Loop over PJs
    juno_df_list = []
    for indx, row in df.query('Orbit >= 35').iloc[:-1,:].iterrows(): # No ephemeris for last PJ
        print(indx)
        starttime = row['PJ_datetime']
        if indx < len(df)-1:
            stoptime = df.loc[indx+1, 'PJ_datetime']
        else:
            stoptime = starttime + (starttime - df.loc[indx-1, 'PJ_datetime'])
            
        time_range = np.arange(starttime, stoptime, dt.timedelta(hours=1)).astype(dt.datetime)
        
        juno_df = pd.DataFrame(index = pd.DatetimeIndex(time_range), columns = ['spacecraft'])
        juno_df['spacecraft'] = 'Juno'
        
        # If we have sufficient coverage, get the ephemerides
        juno_df = preproc.get_SpacecraftPositions(juno_df, 'Juno')
        
        # How often is Juno in the magnetosphere, magnetopause, and solar wind?
        r_MP, r_BS, r_MPJ02, r_BSJ02 = [], [], [], []
        for pressure in pressures:
            r_MP.extend(MP_function([juno_df['t'], juno_df['p'], np.full(len(juno_df), pressure)], average=True))
            r_BS.extend(BS_function([juno_df['t'], juno_df['p'], np.full(len(juno_df), pressure)], average=True))
            
            r_MPJ02.extend(MP_function_J02([juno_df['t'], juno_df['p'], np.full(len(juno_df), pressure)], average=True))
            r_BSJ02.extend(BS_function_J02([juno_df['t'], juno_df['p'], np.full(len(juno_df), pressure)], average=True))
            
        r_MP = np.array(r_MP)
        r_BS = np.array(r_BS)
        r_MPJ02 = np.array(r_MPJ02)
        r_BSJ02 = np.array(r_BSJ02)
        
        in_MS_fraction = [juno_df['r'].to_numpy() < r0 for r0 in r_MP]
        in_SH_fraction = [(juno_df['r'].to_numpy() > r0) & (juno_df['r'].to_numpy() < r1) for r0, r1 in zip(r_MP, r_BS)]
        in_SW_fraction = [juno_df['r'].to_numpy() > r1 for r1 in r_BS]
        
        in_MS_fractionJ02 = [juno_df['r'].to_numpy() < r0 for r0 in r_MPJ02]
        in_SH_fractionJ02 = [(juno_df['r'].to_numpy() > r0) & (juno_df['r'].to_numpy() < r1) for r0, r1 in zip(r_MPJ02, r_BSJ02)]
        in_SW_fractionJ02 = [juno_df['r'].to_numpy() > r1 for r1 in r_BSJ02]
        
        for key, frac in zip(['in_MS', 'in_SH', 'in_SW'], [in_MS_fraction, in_SH_fraction, in_SW_fraction]):
        #     p25, p50, p75 = np.percentile(np.mean(frac, 1), [25, 50, 75])
        #     df.loc[indx, key+'25'] = p25
        #     df.loc[indx, key+'50'] = p50
        #     df.loc[indx, key+'75'] = p75
        
            df.loc[indx, key] = np.mean(frac)
            
        for key, frac in zip(['in_MS', 'in_SH', 'in_SW'], [in_MS_fractionJ02, in_SH_fractionJ02, in_SW_fractionJ02]):        
            df.loc[indx, key+'J02'] = np.mean(frac)

        juno_df_list.append(juno_df)

    juno_df = pd.concat(juno_df_list)
    
    fig, axs = plt.subplot_mosaic("""
                                  ab
                                  cc
                                  """,
                                  height_ratios = [1, 1], figsize=(6.5, 4))
    plt.subplots_adjust(left = 0.1, right = 0.75, bottom = 0.1, top = 0.98,
                        wspace = 0.14, hspace = 0.24)
    
    axs['a'].set(xlim = [120, -80], xlabel = r'$x_{JSS} [R_J]$', 
                 ylim = [0, 150], ylabel = r'$\rho$ $[R_J]$',
                 aspect=1)
    axs['a'].set_title('Magnetopause Models', size='medium', pad=3)
    axs['b'].set(xlim = [120, -80], xlabel = r'$x_{JSS}$ $[R_J]$',
                 ylim = [0, 150], yticklabels = [], 
                 aspect=1)
    axs['b'].set_title('Bow Shock Models', size='medium', pad=3)
    axs['c'].set(xlabel = 'Juno Orbit Number (Extended Mission)',
                 ylabel = 'Average Fraction' + '\n' + 'of Orbit in Region', ylim=[1e-3, 1],
                 yscale='log'
                 )
    for key in axs.keys():
        axs[key].annotate('({})'.format(key), (0, 1), (1, -1), 
                          'axes fraction', 'offset fontsize', ha='center', va='center')
    
    axs['a'].plot(juno_df['ell'], juno_df['rho'], color='black', alpha = 0.5, lw=0.5)
    axs['b'].plot(juno_df['ell'], juno_df['rho'], color='black', alpha = 0.5, lw=0.5)
    
    t_plot = np.linspace(-0.975*np.pi, 0.975*np.pi, 1000)
    p_plot = np.zeros(1000)
    p_dyn_plot = np.zeros(1000)
    constant_pressures = [0.02, 0.071, 0.22]
    
    for const_pressure, color, label in zip(constant_pressures, ['C0', 'C1', 'C3'], ['$10^{th}\%ile$', '$50^{th}\%ile$', '$90^{th}\%ile$']):
        r_plot = MP_function([t_plot, p_plot, p_dyn_plot + const_pressure], average=True)
        rpl = BM.convert_SphericalSolarToCylindricalSolar(r_plot[0], t_plot, p_plot)
        axs['a'].plot(rpl[2], rpl[0], color=color, lw=1.5,
                      label = r'$r_{{new, MP}}(p_{{dyn}}$ = {})'.format(label))
        
        r_plot = BS_function([t_plot, p_plot, p_dyn_plot + const_pressure], average=True)
        rpl = BM.convert_SphericalSolarToCylindricalSolar(r_plot[0], t_plot, p_plot)
        axs['b'].plot(rpl[2], rpl[0], color=color, lw = 1.5,
                      label = r'$r_{{new}}$({} $p_{{dyn}}$)'.format(label))
        
    for const_pressure, color, label in zip(constant_pressures, ['C0', 'C1', 'C3'], ['$10^{th}\%ile$', '$50^{th}\%ile$', '$90^{th}\%ile$']):
        
        r_plot = MP_function_J02([t_plot, p_plot, p_dyn_plot + const_pressure], average=True)
        rpl = BM.convert_SphericalSolarToCylindricalSolar(r_plot[0], t_plot, p_plot)
        axs['a'].plot(rpl[2], rpl[0], color=color, lw=1.5, ls='--',
                      label = r'$r_{{J02, MP}}(p_{{dyn}}$ = {})'.format(label))
        
        r_plot = BS_function_J02([t_plot, p_plot, p_dyn_plot + const_pressure], average=True)
        rpl = BM.convert_SphericalSolarToCylindricalSolar(r_plot[0], t_plot, p_plot)
        axs['b'].plot(rpl[2], rpl[0], color=color, lw = 1.5, ls='--',
                      label = r'$r_{{J02}}$({} $p_{{dyn}}$)'.format(label))
    
    # axs['a'].legend(loc='lower right')
    axs['b'].legend(loc='upper left', bbox_to_anchor=(1, 1.03, 0.6, 0), mode='extend')
    
    # Plot the fractions of time spent in different magnetospheric regions
    # MP
    # ms0 = axs['c'].plot(df['Orbit'], df['in_MS'], markersize=10, lw=0.75, edgecolor='C3', facecolor='white', zorder=0)
    axs['c'].plot(df['Orbit'], df['in_MS'], color='C2', lw=1, linestyle='-', zorder=0,
                  markerfacecolor = 'white', markeredgecolor = 'C2', markersize=2, marker = 'o',
                  label = 'Magnetosphere (new)') 
    axs['c'].plot(df['Orbit'], df['in_SH'], color='C4', lw=1, linestyle='-', zorder=0,
                  markerfacecolor = 'white', markeredgecolor = 'C4', markersize=2, marker = '^',
                  label = 'Magnetosheath (new)') 
    axs['c'].plot(df['Orbit'], df['in_SW'], color='C5', lw=1, linestyle='-', zorder=0,
                  markerfacecolor = 'white', markeredgecolor = 'C5', markersize=2, marker = 'X',
                  label = 'Solar Wind (new)') 
    
    
    
    axs['c'].plot(df['Orbit'], df['in_MSJ02'], color='#bbbbbb', lw=1, linestyle='--', zorder=-2,
                  markerfacecolor = 'white', markeredgecolor = '#bbbbbb', markersize=2, marker = 'o',
                  label = 'Magnetosphere (J02)') 
    axs['c'].plot(df['Orbit'], df['in_SHJ02'], color='#bbbbbb', lw=1, linestyle='--', zorder=-2,
                  markerfacecolor = 'white', markeredgecolor = '#bbbbbb', markersize=2, marker = '^',
                  label = 'Magnetosheath (J02)') 
    axs['c'].plot(df['Orbit'], df['in_SWJ02'], color='#bbbbbb', lw=1, linestyle='--', zorder=-2,
                  markerfacecolor = 'white', markeredgecolor = '#bbbbbb', markersize=2, marker = 'X',
                  label = 'Solar Wind (J02)') 

    axs['c'].legend(loc='upper left', bbox_to_anchor=(1, 1.03, 0.6, 0), mode='extend')
    
    plt.show()
    
    breakpoint()
    
    breakpoint()
    
    return

def plot_Interpretation(model_dict, posterior, positions_df):
    
    # Draw samples to give a sense of the model spread:
    posterior_params_samples = az.extract(posterior, num_samples=100)
    
    posterior_params_mean = []
    posterior_params_vals = []
    for param_name in model_dict['param_distributions'].keys():
        0.8*0
        # Get mean values for each parameter
        posterior_params_mean.append(np.mean(posterior[param_name].values))
        
        # And record the sample values
        posterior_params_vals.append(posterior_params_samples[param_name].values)
    
    # Transpose so we get a list of params in proper order
    posterior_params_vals = np.array(posterior_params_vals).T
    
    # Get sigmas
    # posterior_sigma_mean = np.mean(posterior['sigma'].values)
    posterior_sigma_m_vals = posterior_params_samples['sigma_m'].values
    posterior_sigma_b_vals = posterior_params_samples['sigma_b'].values
        
    # Get rng for adding sigmas
    rng = np.random.default_rng()
        
    # Plotting coords
    n_coords = int(1e4)
    mean_p_dyn = np.mean(positions_df['p_dyn'])
    
    t_coord = np.linspace(0, 0.99*np.pi, n_coords)
    t_coord = np.concatenate([np.flip(t_coord), t_coord])
    
    # p_coords = {'North': np.full(n_coords, 0),
    #             'South': np.full(n_coords, +np.pi),
    #             'Dawn': np.full(n_coords, +np.pi/2.),
    #             'Dusk': np.full(n_coords, -np.pi/2.)
    #             }
    p_coords = {'NS': np.concatenate([np.full(n_coords, 0), np.full(n_coords, +np.pi)]),
                'DD': np.concatenate([np.full(n_coords, +np.pi/2.), np.full(n_coords, -np.pi/2.)]),
                }
    
    p_dyn_coords = {'16': np.full(2*n_coords, np.percentile(positions_df['p_dyn'], 16)),
                    '50': np.full(2*n_coords, np.percentile(positions_df['p_dyn'], 50)),
                    '84': np.full(2*n_coords, np.percentile(positions_df['p_dyn'], 84))
                    }
    
    fig, axs = plt.subplots(nrows = 3, sharex = True,
                            figsize = (6.5, 5))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.7, top=0.98,
                        hspace=0.08)
    
    # Set up each set of axes
    # x_label_centered_x = (axs[0].get_position()._points[0,0] + axs[0].get_position()._points[1,0])/2.
    # x_label_centered_y = (0 + axs[1].get_position()._points[0,1])/2.
    # fig.supxlabel(r'$x_{JSS}$ [$R_J$] (+ toward Sun)', 
    #               position = (x_label_centered_x, x_label_centered_y),
    #               ha = 'center', va = 'top')
    # y_label_centered_x = (0 + axs[0].get_position()._points[0,0])/2.
    # y_label_centered_y = (axs[0].get_position()._points[1,1] + axs[1].get_position()._points[0,1])/2.
    # fig.supylabel(r'$\rho_{JSS} = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]', 
    #               position = (y_label_centered_x, y_label_centered_y),
    #               ha = 'right', va = 'center')
    
    axs[0].set(xlim = (200, -600),
               ylim = (-300, 300),
               aspect = 1)
    axs[1].set(ylim = (-300, 300),
               aspect = 1)
    axs[2].set(ylim = (0, 300),
               aspect = 1)
    
    axs[0].annotate('(a)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    axs[1].annotate('(b)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    axs[2].annotate('(c)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    
    # direction_colors = {'North': 'C0',
    #                     'South': 'C1',
    #                     'Dawn': 'C3',
    #                     'Dusk': 'C5'}
        
    direction_colors = {'NS': 'C0',
                        'DD': 'C5'}
    p_dyn_linestyles = {'16': ':',
                        '50': '-',
                        '84': '--'}

    # Top axes: Side-view, dusk on bottom
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['DD'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['DD'])
    axs[0].plot(xyz[0], xyz[1], color='black')
    
    # r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
    # xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
    # axs[0].plot(xyz[0], xyz[1], color='black')
    
    for params, sigma_m, sigma_b in zip(posterior_params_vals, posterior_sigma_m_vals, posterior_sigma_b_vals):
        r_coord = model_dict['model'](params, [t_coord, p_coords['DD'], p_dyn_coords['50']])
        r_coord = r_coord + (rng.normal(loc = 0, scale = sigma_m) * np.sin(t_coord/2)**2 + rng.normal(loc = 0, scale = sigma_b))
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['DD'])
        axs[0].plot(xyz[0], xyz[1], color='black', alpha=0.05, zorder=-10)
        
        # r_coord = model_dict['model'](params, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
        # r_coord = r_coord + rng.normal(loc = 0, scale = sigma)
        # xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
        # axs[0].plot(xyz[0], xyz[1], color='black', alpha=0.05, zorder=-10)
    
    # Middle axes: Top-down view
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['NS'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['NS'])
    axs[1].plot(xyz[0], xyz[2], color='black')
    
    # r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['South'], p_dyn_coords['50']])
    # xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['South'])
    # axs[1].plot(xyz[0], xyz[2], color='black')
    
    for params, sigma_m, sigma_b in zip(posterior_params_vals, posterior_sigma_m_vals, posterior_sigma_b_vals):
        r_coord = model_dict['model'](params, [t_coord, p_coords['NS'], p_dyn_coords['50']])
        r_coord = r_coord + (rng.normal(loc = 0, scale = sigma_m) * np.sin(t_coord/2)**2 + rng.normal(loc = 0, scale = sigma_b))
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['NS'])
        axs[1].plot(xyz[0], xyz[2], color='black', alpha=0.05, zorder=-10)
        
        # r_coord = model_dict['model'](params, [t_coord, p_coords['South'], p_dyn_coords['50']])
        # r_coord = r_coord + rng.normal(loc = 0, scale = sigma)
        # xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['South'])
        # axs[1].plot(xyz[0], xyz[2], color='black', alpha=0.05, zorder=-10)
    
    # Bottom axes: plot for different pressures, superimposed
    for p_dyn_value in p_dyn_coords.keys():
        for direction in ['NS', 'DD']:
            
            r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords[direction], p_dyn_coords[p_dyn_value]])
            rpl = BM.convert_SphericalSolarToCylindricalSolar(r_coord, t_coord, p_coords[direction])
            axs[2].plot(rpl[2], rpl[0],
                        color = direction_colors[direction], ls = p_dyn_linestyles[p_dyn_value],
                        label = r'{}, $p_{{dyn}} = {}^{{th}} \%ile$'.format(direction, p_dyn_value))
            
            axs[2].legend()
            
    plt.show()
            
def plot_AgainstTrajectories(boundary = 'BS', model_dict = None, posterior = None):
    
    # Parse the posterior object
    if posterior is not None:
        # Draw samples to give a sense of the model spread:
        posterior_params_samples = az.extract(posterior, num_samples=400)
        
        posterior_params_mean = []
        posterior_params_vals = []
        
        posterior_sigmas_mean = []
        posterior_sigmas_vals = []
        for param_name in model_dict['param_distributions'].keys():
            
            # Get mean values for each parameter
            posterior_params_mean.append(np.mean(posterior[param_name].values))
            
            # And record the sample values
            posterior_params_vals.append(posterior_params_samples[param_name].values)
            
            # Get sigmas
            sigma_param_name = 'sigma_' + param_name
            if sigma_param_name in list(posterior.variables):
                posterior_sigmas_mean.append(np.median(posterior[sigma_param_name].values))
                
                posterior_sigmas_vals.append(posterior_params_samples[sigma_param_name].values)
            else:
                posterior_sigmas_mean.append(np.median(posterior[param_name].values))
                posterior_sigmas_vals.append(posterior_params_samples[param_name].values)
        
        # Transpose so we get a list of params in proper order
        posterior_params_vals = np.array(posterior_params_vals).T
        posterior_sigmas_vals = np.array(posterior_sigmas_vals).T
        
        # Get rng for adding sigmas
        rng = np.random.default_rng()
        
    resolution = '10Min'
    sc_colors = {'Ulysses': '#910951', 
                 'Galileo': '#b6544a', 
                 'Cassini': '#d98b3a', 
                 'Juno': '#fac205'}
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    
    # Read Crossings (no class balancing)
    positions_df = preproc.read_AllCrossings(resolution = resolution, padding = dt.timedelta(hours=3000))
    positions_df = positions_df.query("spacecraft in @spacecraft_to_use")
    # Replace datetime index with integers, deal with duplicated rows later
    positions_df = positions_df.reset_index(names='datetime')
    
    # Add pressure information
    sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
    sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1)
    p_dyn_mme = sw_mme.filter(like = 'p_dyn').resample(resolution).interpolate()
   
    # Drop NaNs
    p_dyn_mme = p_dyn_mme.dropna(axis = 'index', how = 'any')
    
    # Make sure positions_df overlaps pressures and concatenate
    positions_df = positions_df.query("@p_dyn_mme.index[0] <= datetime <= @p_dyn_mme.index[-1]")   
    positions_df.loc[:, 'p_dyn'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_a'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_a'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_loc'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_loc'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_scale'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_scale'].to_numpy('float64')
    
    breakpoint()
    
    # Rotate all crossings to North & scale to constant mean pressure
    # Difference between actual position and boundary position
    boundary_at_point = model_dict['model'](posterior_params_mean, [positions_df['t'], positions_df['p'], positions_df['p_dyn']])
    delta_r = positions_df['r'] - boundary_at_point
    
    # If the point were on the boundary, where would it be rotated to North and w/ constant mean pressure?
    n_df = len(positions_df)
    p_dyn_mean = np.mean(positions_df['p_dyn'])
    p_to_rotate_to = np.pi/2. # Dawn
    
    coords_rotated_scaled = [positions_df['t'], np.full(n_df, p_to_rotate_to), np.full(n_df, p_dyn_mean)]
    r_rotated_scaled = model_dict['model'](posterior_params_mean, coords_rotated_scaled)
    
    new_r = r_rotated_scaled + (delta_r * (r_rotated_scaled/boundary_at_point))
    
    # Set r = new_r, p = 0 in positions_df, recalculate
    translated_positions_df = positions_df.copy(deep = True)
    translated_positions_df['r'] = new_r
    translated_positions_df['p'] = p_to_rotate_to = +np.pi/2.
    
    xyz = BM.convert_SphericalSolarToCartesian(*translated_positions_df[['r', 't', 'p']].to_numpy().T)
    translated_positions_df = translated_positions_df.assign(x=xyz[0], y=xyz[1], z=xyz[2])
    rpl = BM.convert_SphericalSolarToCylindricalSolar(*translated_positions_df[['r', 't', 'p']].to_numpy().T)
    translated_positions_df = translated_positions_df.assign(rho=rpl[0], phi=rpl[1], ell=rpl[2])

    # Show the range of possible boundary surface locations
    fig, axs = plt.subplots(nrows = 2, sharex = True,
                            figsize = (4.5, 5))
    plt.subplots_adjust(left=0.16, bottom=0.08, right=0.94, top=0.98,
                        hspace=0.04)
    
    # Set up each set of axes
    x_label_centered_x = (axs[0].get_position()._points[0,0] + axs[0].get_position()._points[1,0])/2.
    x_label_centered_y = (0 + axs[1].get_position()._points[0,1])/2.
    fig.supxlabel(r'$x_{JSS}$ [$R_J$] (+ toward Sun)', 
                  position = (x_label_centered_x, x_label_centered_y),
                  ha = 'center', va = 'top')
    
    y_label_centered_x = (0 + axs[0].get_position()._points[0,0])/2.
    y_label_centered_y = (axs[0].get_position()._points[1,1] + axs[1].get_position()._points[0,1])/2.
    fig.supylabel(r'$\rho_{JSS} = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]', 
                  position = (y_label_centered_x, y_label_centered_y),
                  ha = 'right', va = 'center')
    
    if boundary == 'BS':
        axs[0].set(xlim = (200, -700),
                   ylim = (0, 500),
                   aspect = 1)
        axs[1].set(xlim = (200, -700),
                    ylim = (0, -500),
                    aspect = 1)
    elif boundary == 'MP':
        axs[0].set(xlim = (140, -400),
                   ylim = (0, 300),
                   aspect = 1)
        axs[1].set(xlim = (140, -400),
                    ylim = (0, -300),
                    aspect = 1)
    else:
        breakpoint()
        
    axs[0].annotate('(a)', 
                    (0,1), (0.5,-1), 
                    'axes fraction', 'offset fontsize', ha='left', va='top')
    
    axs[1].annotate('(b)', 
                    (0,1), (0.5,-1), 
                    'axes fraction', 'offset fontsize', ha = 'left', va='top')
    axs[1].annotate('Rotated to the same plane & \nScaled to ' + r'$\overline{{p}}_{{dyn}} = {:.3f} nPa$'.format(p_dyn_mean), 
                    (0,1), (2.5,-1), 
                    'axes fraction', 'offset fontsize', ha = 'left', va='top')

    #   Dummy coords for plotting
    t_plot = np.linspace(0, 0.75*np.pi, 1000)
    p_plot = np.zeros(1000) + np.pi/2
    p_dyn_plot = np.zeros(1000)
    coords_plot = (t_plot, p_plot, p_dyn_plot)
    
    # Plot translated crossings
    for spacecraft in spacecraft_to_use:
        
        subset_df = translated_positions_df.query('spacecraft == @spacecraft')
        
        if boundary == 'BS':
            boundary_entries_index = (subset_df['SW'].shift(1) == 1) & (subset_df['SW'] == 0)
            boundary_exits_index = (subset_df['SW'].shift(1) == 0) & (subset_df['SW'] == 1)
            outside_mask = subset_df['SW'] == 1
            inside_mask = (subset_df['SH'] == 1) | (subset_df['MS'] == 1)
        elif boundary == 'MP':
            boundary_entries_index = (subset_df['MS'].shift(1) == 0) & (subset_df['MS'] == 1)
            boundary_exits_index = (subset_df['MS'].shift(1) == 1) & (subset_df['MS'] == 0)
            outside_mask = (subset_df['SW'] == 1) | (subset_df['SH'] == 1)
            inside_mask = subset_df['MS'] == 1
        
        # Trajectories commented out as they don't make sense post-translation
        # # Make DataFrames with NaNs where not in region, to make plotting easy
        # inside_df = subset_df[['spacecraft', 'rho', 'phi', 'ell']]
        # inside_df.loc[~inside_mask, ['rho', 'phi', 'ell']] = np.nan
        
        # outside_df = subset_df[['spacecraft', 'rho', 'phi', 'ell']]
        # outside_df.loc[~outside_mask, ['rho', 'phi', 'ell']] = np.nan

        # # Ensure that we don't flip back and forth between coincident spacecraft
        # ax.plot(inside_df.query("spacecraft == @spacecraft")['ell'],
        #         inside_df.query("spacecraft == @spacecraft")['rho'],
        #         # label = '',
        #         color = sc_colors[spacecraft], lw = 1, ls = '--',
        #         zorder = 9)
        # ax.plot(outside_df.query("spacecraft == @spacecraft")['ell'],
        #         outside_df.query("spacecraft == @spacecraft")['rho'],
        #         label = '{} Trajectory'.format(spacecraft),
        #         color = sc_colors[spacecraft], lw = 1, ls = '-',
        #         zorder = 9)
        
        # Crossings
        crossing_regions = ['SW', 'SH'] if boundary == 'BS' else ['SH', 'MS']
        label_str = r'{0} $\rightarrow$ {1}'.format(*crossing_regions) 
        axs[1].scatter(subset_df.loc[boundary_entries_index, 'x'], 
                   subset_df.loc[boundary_entries_index, 'y'],
                   label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                   s = 16, color='#0001a7', marker='x', lw = 1,
                   zorder = 10)
        label_str = r'{1} $\rightarrow$ {0}'.format(*crossing_regions)
        axs[1].scatter(subset_df.loc[boundary_exits_index, 'x'], 
                   subset_df.loc[boundary_exits_index, 'y'],
                   label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                   s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 1,
                   zorder = 10)
        
    # Plotting coords
    n_coords = int(1e4)
    mean_p_dyn = np.mean(positions_df['p_dyn'])
    
    t_coord = np.linspace(0, 0.99*np.pi, n_coords)
    # t_coord = np.concatenate([np.flip(t_coord), t_coord])
    
    p_coords = {'North': np.full(n_coords, 0),
                'South': np.full(n_coords, +np.pi),
                'Dawn': np.full(n_coords, +np.pi/2.),
                'Dusk': np.full(n_coords, -np.pi/2.)
                }
    # p_coords = {'NS': np.concatenate([np.full(n_coords, 0), np.full(n_coords, +np.pi)]),
    #             'DD': np.concatenate([np.full(n_coords, +np.pi/2.), np.full(n_coords, -np.pi/2.)]),
    #             }
    
    p_dyn_coords = {'16': np.full(n_coords, np.percentile(positions_df['p_dyn'], 16)),
                    '50': np.full(n_coords, np.percentile(positions_df['p_dyn'], 50)),
                    '84': np.full(n_coords, np.percentile(positions_df['p_dyn'], 84))
                    }
    
    direction_colors = {'North': 'C0',
                        'Dawn': 'C1',
                        'South': 'C0',
                        'Dusk': 'C5'}
        
    p_dyn_linestyles = {'16': '-',
                        '50': '--',
                        '84': ':'}

    # =============================================================================
    #     
    # =============================================================================
    # Include a sense of sigma
    filtered_params, other_params = [], []
    all_r = []
    for params, sigma_params in zip(posterior_params_vals, posterior_sigmas_vals):
        r_coord = model_dict['model'](params, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
        # r_coord_sigma = (rng.normal(loc = 0, scale = sigma/2) * (2/(1 + np.cos(t_coord))))
        # r_coord_sigma = r_coord * (rng.normal(loc = 0, scale = sigma/2) / params[0]) # !!!! Move this parametrization to sampler
        
        r_sigma = model_dict['model'](sigma_params, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
        
        r_coord += rng.normal(loc = 0, scale = 1, size = 1) * r_sigma
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
        
        # Check for physical descriptions 
        if (xyz[0] < 500).all():
            axs[1].plot(xyz[0], xyz[1], color='black', alpha=0.01, zorder=-10)
            filtered_params.append(params)
            
            all_r.append(r_coord)
        
        # breakpoint()
            
    # # All said and done:    
    # mean_all_r = np.mean(all_r, 0)
    # mean_xyz = BM.convert_SphericalSolarToCartesian(mean_all_r, t_coord, p_coords['Dawn'])
            
    # Bottom axes: Dawn, to compare with translated crossings
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
    axs[1].plot(xyz[0], xyz[1], label = 'New Fit', 
                color='black', zorder=2)

    # Compare to Joy+ 2002
    joy2002_params = JBC.get_JoyParameters(boundary = boundary)
    r_coord = BM.Joylike(joy2002_params, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
    axs[1].plot(xyz[0], xyz[1], label = 'Joy+ (2002)'.format(p_dyn_mean),
                color='C2', ls = '--', zorder=0)
    
    joy_coords = JBC.find_JoyBoundaries(p_dyn_coords['50'][0], boundary=boundary, 
                                        x=xyz[0], y=False, z=0)
    
    
    leg = axs[1].legend(scatterpoints=3, handlelength=3,
                    loc='lower right')
    for line in leg.get_lines():
            line.set_linewidth(2.0)

    # Top axes: plot for different pressures, superimposed
    for p_dyn_value in p_dyn_coords.keys():
        for direction in ['North', 'Dawn', 'Dusk']:
            
            r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords[direction], p_dyn_coords[p_dyn_value]])
            rpl = BM.convert_SphericalSolarToCylindricalSolar(r_coord, t_coord, p_coords[direction])
            axs[0].plot(rpl[2][0:n_coords], rpl[0][0:n_coords],
                        color = direction_colors[direction], ls = p_dyn_linestyles[p_dyn_value],
                        label = '_')
            
    axs[0].plot([0,0], [-100,-100], color = direction_colors['North'], 
                lw=2, label = 'North-South')
    axs[0].plot([0,0], [-100,-100], color = direction_colors['Dawn'], 
                lw=2, label = 'Dawn')
    axs[0].plot([0,0], [-100,-100], color = direction_colors['Dusk'], 
                lw=2, label = 'Dusk')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['16'],
                lw=2, label = r'$p_{dyn}: 16^{th} \%ile$')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['50'],
                lw=2, label = r'$p_{dyn}: 50^{th} \%ile$')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['84'],
                lw=2, label = r'$p_{dyn}: 84^{th} \%ile$')
    
    axs[0].legend(loc='lower right')
            
    breakpoint()
            
    plt.show()

def read_PosteriorAsDataFrame(posterior_path):
    import pickle

    # Read the params from posterior
    with open(posterior_path, 'rb') as f:
        posterior = pickle.load(f)
    
    return posterior.to_dataframe()
    
def make_BoundaryFunctionFromPosterior(model_name, posterior_df):
    
    def BoundaryFunction(coords, n = 10, average = False, return_weights=False):
        
        # Names of all parameteres in the posterior file
        posterior_param_names = list(posterior_df.columns)
        # Check if bimodal or not
        if 'w0' in posterior_param_names:
            n_modes = 2
        else:
            n_modes = 1
        
        # Sample the posterior
        if len(posterior_df) < n:
            samples_df = posterior_df.sample(n, replace=True)
        else:
            samples_df = posterior_df.sample(n, replace=False)
            
        # Initialize dataframes to hold all the parameters
        params_df = pd.DataFrame(index = np.arange(n), columns = posterior_param_names)
        averages_df = pd.DataFrame(index = [0], columns = posterior_param_names)
        
        # Populate the dataframes
        for name in posterior_param_names:
            params_df[name] = samples_df[name].values
            averages_df[name] = np.median(posterior_df[name].values)
    
        # If we want the mean of a bimodal, add a second entry with different odds
        if 'w0' in posterior_param_names:
            averages_df.loc[1] = averages_df.loc[0]
            averages_df['w0'] = [1,0]
            averages_df['w1'] = [0,1]
            
        # Get the model description and a list of expected param names
        model_dict = BM.init(model_name)
        model_param_names = list(model_dict['param_descriptions'].keys())
        
        # Choose which df to use
        if average == True:
            df = averages_df
        else:
            df = params_df
        
        r = []
        for index, row in df.iterrows():
            
            mu0 = model_dict['model'](row[model_param_names], coords)
            
            if n_modes == 2:
                mu1 = mu0 + row['buffer_m'] * mu0 + row['buffer_b']
                dirichlet_odds = [row['w0'], row['w1']]
            elif n_modes == 1:
                mu1 = mu0 * 0
                dirichlet_odds = [1, 0] # Definitely mode 0
            
            # Choose a mode
            rng = np.random.default_rng()
            
            # 
            mu_choice = rng.choice([mu0, mu1], p = dirichlet_odds)
            
            # 
            sigma = row['sigma_m'] * mu0 + row['sigma_b'] # sigma based on mu0, not mu_choice
            
            # Get gamma distribution shape parameters
            # if (sigma == 0).any():
            #     breakpoint()
            # shape_param_k = (mu_choice / sigma)**2
            # shape_param_theta = sigma**2 / mu_choice
            
            gamma_alpha = (mu_choice / sigma)**2
            gamma_beta = mu_choice / sigma**2
            
            # The boundary mean is specified by a gamma distribution
            # sigma should vary surface-to-surface, not within one surface
            # So we can't simply do rng.gamma() for input coord
            # Instead, get a complete gamma distribution
            # gamma_dist = pm.Gamma.dist(mu = mu_choice, sigma = sigma)
            
            # Determine how many sigma away each sample should be
            normal_draw = rng.normal(loc = 0, scale = 1, size = 1)
            
            # Convert these to percentiles
            percentile_draw = norm.cdf(normal_draw, loc = 0, scale = 1)
            
            
            
            # Sample the gamma distribution at that percentile
            if average == False:

                mu_final = gamma.ppf(percentile_draw, gamma_alpha, scale = 1/gamma_beta)
            else:
                
                mu_final = gamma.ppf(0.50, gamma_alpha, scale = 1/gamma_beta)

            r.append(mu_final.flatten())
            
        if return_weights == True:
            if n_modes == 1:
                weights = [1] * len(r)
            elif n_modes == 2:
                if average == True:
                    weights = np.mean(posterior_df['w0']), np.mean(posterior_df['w1'])
                else:
                    weights = np.mean(samples_df['w0']), np.mean(samples_df['w1'])
                    
            result = r, weights
        else:
            result = r
        
        return result

    return BoundaryFunction
  

def WithinBounds(boundary, model_name, posterior_fn, positions_df):
    
    rs_mu = posterior_fn(positions_df[['t', 'p', 'p_dyn']].to_numpy().T, n = 100)

    bool_list = []
    for r_mu in rs_mu:
        condition = (positions_df['r_lowerbound'] <= r_mu) & (r_mu <= positions_df['r_upperbound'])
        bool_list.append(condition.to_numpy())
        
    withinbounds_per_sample = np.mean(bool_list, axis=1)
    
    print("Within bounds: {:.3f} +/- {:.3f}".format(np.mean(withinbounds_per_sample), np.std(withinbounds_per_sample)))
    
    return

def get_BoundariesFromPosterior(model_name, coordinates,
                                posterior_path=None,
                                n=10,
                                mean = False):
    """
    Given a model name and posterior, return N possible surfaces

    Parameters
    ----------
    boundary : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    posterior_path : TYPE, optional
        DESCRIPTION. The default is None.
    params : TYPE, optional
        DESCRIPTION. The default is None.
    n : Integer, optional
    
    mean : Boolean, optional
        If True, return the mean surface rather than a sample

    Returns
    -------
    None.

    """
    import pickle
    import pandas as pd

    # Read the params from posterior or a dict, and get the param names
    if type(posterior_path) is str:
        ptype = 'pathlike'
        with open(posterior_path, 'rb') as f:
            posterior = pickle.load(f)
        param_names = list(posterior.keys())
        if mean == False:
            samples = az.extract(posterior, num_samples=n)
        else:
            n = 1
            mean_param_vals = [np.mean(posterior[param_name].values) for param_name in param_names]
            samples = {k: v for k, v in zip(param_names, mean_param_vals)}
    else:
        ptype = 'dictlike'
        param_names = list(posterior_path.keys())
        posterior = samples = posterior_path
        
        if mean == True:
            n = 1
        
    # Check if bimodal or not
    if 'w0' in param_names:
        n_modes = 2
    else:
        n_modes = 1
        
    # Initialize a dataframe to hold all the parameters
    params_df = pd.DataFrame(index = np.arange(n), columns = param_names)
    # Populate the dataframe
    for param_name in param_names:
        # Clunky catch for xarray vs arrays/lists/floats
        try:
            params_df[param_name] = samples[param_name].values
        except:
            params_df[param_name] = samples[param_name]
    # If we want the mean of a bimodal, add a second entry with different odds
    if (mean == True) & ('w0' in param_names):
        params_df.loc[1] = params_df.loc[0]
        params_df['w0'] = [1, 0]
        params_df['w1'] = [0, 1]
        
    # Get the model description and a list of expected param names
    model_dict = BM.init(model_name)
    
    # Get possible surfaces
    possible_surface_list = []
    
    # Constant shape params, only sigma and w change each row
    if ptype == 'pathlike':
        shape_params = [np.median(posterior[vals].values) for vals in list(model_dict['param_descriptions'].keys())]
    else:
        shape_params = [np.median(posterior[vals]) for vals in list(model_dict['param_descriptions'].keys())]
    
    # Constant buffer params, only sigma and w change each row
    if n_modes == 2:
        buffer_params = [np.median(posterior[vals].values) for vals in ['buffer_m', 'buffer_b']]
    else:
        buffer_params = [0, 0]
        
    for index, row in params_df.iterrows():
        # # Shape params change in each row
        # shape_params = [row[val] for val in list(model_dict['param_descriptions'].keys())]
        
        mu0 = model_dict['model'](shape_params, coordinates)
        
        if n_modes == 2:
            mu1 = mu0 + (buffer_params[0] * mu0 + buffer_params[1])
            dirichlet_odds = [row['w0'], row['w1']]
        elif n_modes == 1:
            mu1 = mu0 * 0
            dirichlet_odds = [1, 0] # Definitely mode 0
        
        
        # Choose a mode
        rng = np.random.default_rng()
        mu_choice = rng.choice([mu0, mu1], p = dirichlet_odds)
        
        # Get and sample uncertainties
        if mean == False:
            sigma = row['sigma_m'] * mu0 + row['sigma_b']
            sigma_choice = rng.normal(loc=0, scale=1, size=1) * sigma
        else:
            sigma_choice = 0
        
        mu_final = mu_choice + sigma_choice
        possible_surface_list.append(mu_final)
    
    # breakpoint()
    return possible_surface_list

def get_PositionsForAnalysis():
        
    # # "Constants"
    resolution = '10Min'
    # sc_colors = {'Ulysses': '#910951', 
    #              'Galileo': '#b6544a', 
    #              'Cassini': '#d98b3a', 
    #              'Juno': '#fac205'}
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    
    # Read Crossings (no class balancing)
    positions_df = preproc.read_AllCrossings(resolution = resolution, padding = dt.timedelta(hours=3000))
    positions_df = positions_df.query("spacecraft in @spacecraft_to_use")
    # Replace datetime index with integers, deal with duplicated rows later
    positions_df = positions_df.reset_index(names='datetime')
    
    # Add pressure information
    sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
    sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1)
    p_dyn_mme = sw_mme.filter(like = 'p_dyn').resample(resolution).interpolate()
   
    # Drop NaNs
    p_dyn_mme = p_dyn_mme.dropna(axis = 'index', how = 'any')
    
    # Make sure positions_df overlaps pressures and concatenate
    positions_df = positions_df.query("@p_dyn_mme.index[0] <= datetime <= @p_dyn_mme.index[-1]")   
    positions_df.loc[:, 'p_dyn'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_a'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_a'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_loc'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_loc'].to_numpy('float64')
    positions_df.loc[:, 'p_dyn_scale'] = p_dyn_mme.loc[positions_df['datetime'], 'p_dyn_scale'].to_numpy('float64')
    
    p_dyn_mu    = skewnorm.mean(loc = positions_df['p_dyn_loc'].to_numpy(), 
                                scale = positions_df['p_dyn_scale'].to_numpy(), 
                                a = positions_df['p_dyn_a'].to_numpy())
    p_dyn_sigma = skewnorm.std(loc = positions_df['p_dyn_loc'].to_numpy(), 
                                scale = positions_df['p_dyn_scale'].to_numpy(), 
                                a = positions_df['p_dyn_a'].to_numpy())
    
    pressure_dist = pm.Truncated.dist(pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = positions_df['p_dyn_a'].to_numpy()),
                                      lower = 0)
    pressure_draws = pm.draw(pressure_dist, draws=20)
    positions_df.loc[:, 'p_dyn'] = np.median(pressure_draws, axis=0)
    
    # Get rid of times when the region is unknown, as these can't be used for testing
    # positions_df = positions_df.query("region != 'UN'")
    
    # breakpoint()
    # positions_df = positions_df.reset_index()
    return positions_df

def GuessCrossings(boundary, model_name, posterior_fn, positions_df):
    
    crossings = postproc.balance_Classes(positions_df, boundary, dt.timedelta(minutes=1), other_fraction=0.0)
    
    breakpoint()
    
    return

def ConfusionMatrix(boundary, model_name, posterior_fn, positions_df):
    
    # original_positions_df = get_PositionsForAnalysis()
    # original_positions_df = original_positions_df.query("region != 'UN'")
    
    # filter for crossings or near-crossings
    case1 = postproc.balance_Classes(positions_df, boundary, dt.timedelta(hours=1), other_fraction=0.0)
    case2 = postproc.balance_Classes(positions_df, boundary, dt.timedelta(hours=10), other_fraction=0.00)
    case3 = postproc.balance_Classes(positions_df, boundary, dt.timedelta(hours=10), other_fraction=0.20)
    
    for df_label, df in zip(['1h', '10h', '10h+20%'], [case1, case2, case3]):
        # True Positive: Both are outside the boundary (i.e., both True)
        # True Negative: Both are inside the boundary (i.e., both False)
        # False Positive: Actually inside, model says outside
        # False Negative: Actually outside, model says inside
        total = len(df)
        
        # Get the true locations
        if boundary == 'BS':
            outside_boundary_data = (df['region'] == 'SW').to_numpy()
        elif boundary == 'MP':
            outside_boundary_data = (df['region'] != 'MS').to_numpy()
        else:
            print("WRONG BOUNDARY NAME")
            breakpoint()
            
        # For each point in positions_df, get the expected boundary location
        # Do this n times to account for uncertainty, and m times for different pressure samples
        n, m = 50, 50
        
        # Setup pressure for random draws
        p_dyn_mu    = skewnorm.mean(loc = df['p_dyn_loc'].to_numpy(), 
                                    scale = df['p_dyn_scale'].to_numpy(), 
                                    a = df['p_dyn_a'].to_numpy())
        p_dyn_sigma = skewnorm.std(loc = df['p_dyn_loc'].to_numpy(), 
                                    scale = df['p_dyn_scale'].to_numpy(), 
                                    a = df['p_dyn_a'].to_numpy())
        p_dyn_dist = pm.Truncated.dist(pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = df['p_dyn_a'].to_numpy()),
                                        lower = 0)
        p_dyn_draws = pm.draw(p_dyn_dist, draws=m)
        
        # p_dyn_draws = [positions_df['p_dyn'].to_numpy('float64')] * m
        
        surfaces = []
        for p_dyn_draw in p_dyn_draws:
            # Get n surfaces for this pressure
            # surfaces.extend(get_BoundariesFromPosterior(model_name,
            #                                             [positions_df['t'].to_numpy(), positions_df['p'].to_numpy(), p_dyn_draw], 
            #                                             posterior_path = posterior_path,
            #                                             n=n))
            # surfaces.extend(get_BoundariesFromPosterior(model_name,
            #                                             [df['t'].to_numpy(), df['p'].to_numpy(), p_dyn_draw], 
            #                                             posterior_path = posterior_path,
            #                                             mean=True))
            surface = posterior_fn([df['t'].to_numpy(), df['p'].to_numpy(), p_dyn_draw],
                                   n = n)
            surfaces.extend(surface)

        cm_stats = ['true_negative', 'false_positive', 'false_negative', 'true_positive']
        stats_df = pd.DataFrame(index = np.arange(n), 
                                columns = cm_stats)
        
        outside_boundary_model_list = []
        for i, surface in enumerate(surfaces):
            # breakpoint()
            # What if we add a "buffer" for 1-sigma uncertainties
            # e.g. outside_boundary_model = positions_df['r'].to_numpy() > (surface + ???)
            outside_boundary_model = df['r'].to_numpy() > surface
            outside_boundary_model_list.append(outside_boundary_model)
            
            tn, fp, fn, tp = metrics.confusion_matrix(outside_boundary_data, outside_boundary_model).ravel()
            
            stats_df.loc[i, cm_stats] = tn, fp, fn, tp
            stats_df.loc[i, 'mcc'] = metrics.matthews_corrcoef(outside_boundary_data, outside_boundary_model)
        
        # Calculate derived statistics
        stats_df['true_positive_rate'] = stats_df['true_positive'] / (stats_df['true_positive'] + stats_df['false_negative'])
        stats_df['false_negative_rate'] = stats_df['false_negative'] / (stats_df['true_positive'] + stats_df['false_negative'])
        stats_df['true_negative_rate'] = stats_df['true_negative'] / (stats_df['true_negative'] + stats_df['false_positive'])
        stats_df['false_positive_rate'] = stats_df['false_positive'] / (stats_df['true_negative'] + stats_df['false_positive'])
        
        stats_df['positive_predictive_value'] = stats_df['true_positive'] / (stats_df['true_positive'] + stats_df['false_positive'])
        stats_df['negative_predictive_value'] = stats_df['true_negative'] / (stats_df['true_negative'] + stats_df['false_negative'])
        
        stats_df['accuracy'] = (stats_df['true_positive'] + stats_df['true_negative']) / (np.sum(stats_df[cm_stats].to_numpy(), axis=1))
        
        stats_df['informedness'] = (stats_df['true_positive_rate'] + stats_df['true_negative_rate'] - 1)
        
        # Print some statistics
        print("Confusion Matrix Statistics for {} window".format(df_label))
        for key in stats_df.columns:
            if key in ['true_negative', 'false_positive', 'false_negative', 'true_positive']:
                print("{0}: {1:.1f} +/- {2:.1f}".format(key, 
                                                        100*np.mean(stats_df[key])/total, 
                                                        100*np.std(stats_df[key])/total))
            else:
                print("{0}: {1:.1f} +/- {2:.1f}".format(key, 
                                                        100*np.mean(stats_df[key]), 
                                                        100*np.std(stats_df[key])))
        print("")
    return

def ConfusionMatrix_vs_WindowSize(boundary, model_dict, posterior, constant_posterior=False):
    
    # Get the spacecraft positions and crossings
    # Even if we don't use all of these in the MCMC, we can compare to all of them
    resolution = '10Min'
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    positions_df = postproc.PostprocessCrossings(boundary, spacecraft_to_use = spacecraft_to_use, 
                                                 delta_around_crossing=dt.timedelta(hours=500), 
                                                 other_fraction=0.0)
    n = len(positions_df)
    positions_df = positions_df.query('r_upperbound < 1e3 & r_lowerbound < 1e3')
    
    # Get rid of times when the region is unknown, as these can't be used for testing
    positions_df_original = positions_df.query("region != 'UN'")
    breakpoint()
    
    test_windows = [0.2, 0.4, 0.5, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    len_arr = []
    informedness_arr = []
    for test_window in test_windows:
        positions_df = postproc.balance_Classes(positions_df_original, boundary, dt.timedelta(hours=test_window), other_fraction=0.0)
        len_arr.append(len(positions_df))
    
        if constant_posterior == False:
            # Draw samples to give a sense of the model spread:
            posterior_params_samples = az.extract(posterior, num_samples=100)
            
            posterior_params_mean = []
            posterior_params_vals = []
            
            posterior_sigmas_mean = []
            posterior_sigmas_vals = []
            for param_name in model_dict['param_distributions'].keys():
                
                # Get mean values for each parameter
                posterior_params_mean.append(np.mean(posterior[param_name].values))
                
                # And record the sample values
                posterior_params_vals.append(posterior_params_samples[param_name].values)
                
                # Get sigmas
                if 'sigma_r0' in list(posterior.variables):
                    flag_multi_sigma = 1
                    sigma_param_name = 'sigma_' + param_name
                    if sigma_param_name in list(posterior.variables):
                        posterior_sigmas_mean.append(np.median(posterior[sigma_param_name].values))
                        
                        posterior_sigmas_vals.append(posterior_params_samples[sigma_param_name].values)
                    else:
                        posterior_sigmas_mean.append(np.median(posterior[param_name].values))
                        posterior_sigmas_vals.append(posterior_params_samples[param_name].values)
                else:
                    flag_multi_sigma = 0
                    posterior_sigmas_mean.append(np.median(posterior_params_samples['sigma_fn'].values))
                    posterior_sigmas_vals.append(posterior_params_samples['sigma_fn'].values)
            
            # Transpose so we get a list of params in proper order
            posterior_params_vals = np.array(posterior_params_vals).T
            posterior_sigmas_vals = np.array(posterior_sigmas_vals).T
        
        else:
            # A 'constant posterior' is a 1D vector of model parameters
            flag_multi_sigma = 1
            posterior_params_vals = np.array([posterior])
            posterior_sigmas_vals = np.array([[0 for e in posterior]])
            
        # Get rng for adding sigmas
        rng = np.random.default_rng()
        
        # Get the true locations
        if boundary == 'BS':
            outside_boundary_data = (positions_df['region'] == 'SW').to_numpy()
        elif boundary == 'MP':
            outside_boundary_data = (positions_df['region'] != 'MS').to_numpy()
        else:
            print("WRONG BOUNDARY NAME")
            breakpoint()
            
        outside_boundary_model = []
        true_positives = [] # Both are outside the boundary (i.e., both True)
        true_negatives = [] # Both are inside the boundary (i.e., both False)
        false_positives = [] # Actually inside, model says outside
        false_negatives = [] # Actually outside, model says inside
        
        # For each point in positions_df, get the expected boundary location
        # Do this n times to account for uncertainty
        for params, sigma_params in zip(posterior_params_vals, posterior_sigmas_vals):
            
            # Boundary distance according to model:
            r_b = model_dict['model'](params, positions_df[['t', 'p', 'p_dyn']].to_numpy('float64').T)
            
            # Uncertainty on r_b
            if flag_multi_sigma == 1:
                r_b_sigma = model_dict['model'](sigma_params, positions_df[['t', 'p', 'p_dyn']].to_numpy('float64').T)
                if (r_b_sigma == 0).any():
                    r_b_sigma[np.argwhere(r_b_sigma == 0)] = r_b[np.argwhere(r_b_sigma == 0)] * 1e-6
            else:
                r_b_sigma = np.zeros(len(r_b)) + sigma_params[0]
            
            for i in range(100):
                outside_boundary_bool = positions_df['r'].to_numpy() > rng.normal(loc = r_b, scale = r_b_sigma) 
                outside_boundary_model.append(outside_boundary_bool)
                
                tn, fp, fn, tp = metrics.confusion_matrix(outside_boundary_data, outside_boundary_bool).ravel()
                true_positives.append(tp)
                true_negatives.append(tn)
                false_positives.append(fp)
                false_negatives.append(fn)
            
        # Print some statistics
        true_positives = np.array(true_positives)
        true_negatives = np.array(true_negatives)
        false_positives = np.array(false_positives)
        false_negatives = np.array(false_negatives)
        totals = np.array(true_positives + true_negatives + false_positives + false_negatives)
        
        true_positives_rate = true_positives / (true_positives + false_negatives)  # recall
        false_negatives_rate = false_negatives / (true_positives + false_negatives)
        true_negatives_rate = true_negatives / (true_negatives + false_positives)
        false_positives_rate = false_positives / (true_negatives + false_positives)
        
        positive_predictive_values = true_positives / (true_positives + false_positives) # precision
        negative_predictive_values = true_negatives / (true_negatives + false_negatives)
        
        F1 = (2 * positive_predictive_values * true_positives_rate) / (positive_predictive_values + true_positives_rate)
        
        informedness_arr.append(np.mean(true_positives_rate + true_negatives_rate - 1)*100)
        
        fig, ax = plt.subplots()
        ax.plot(test_windows, len_arr)
        ax1 = ax.twinx()
        ax1.plot(test_windows, informedness_arr, color='C1')
        ax.set(xlabel='Halfwidth of window around crossing [hours]', ylabel = '# of datapoints')
        ax1.set(ylabel='Informedness [%]')
        ax1.spines['left'].set_color('C0')
        ax1.spines['right'].set_color('C1')
        
    breakpoint()
    return