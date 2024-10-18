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

import JoyBoundaryCoords as JBC
import BoundaryModels as BM
import CrossingPreprocessingRoutines as preproc
import CrossingPostprocessingRoutines as postproc

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

def plot_Interpretation(model_dict, posterior, positions_df):
    
    # Draw samples to give a sense of the model spread:
    posterior_params_samples = az.extract(posterior, num_samples=100)
    
    posterior_params_mean = []
    posterior_params_vals = []
    for param_name in model_dict['param_distributions'].keys():
        
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
    
    # model_dict = BM.init('Shuelike_AsymmetryCase1')
    # boundary = 'BS'
    # posterior = 
    # posterior_params_mean = [80, -0.05, -9, 0.8, -0.29]
    # posterior_params_vals = []
    
    # Parse the posterior object
    if posterior is not None:
        # Draw samples to give a sense of the model spread:
        posterior_params_samples = az.extract(posterior, num_samples=100)
        
        posterior_params_mean = []
        posterior_params_vals = []
        for param_name in model_dict['param_distributions'].keys():
            
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
    
    
    # Rotate all crossings to North & scale to constant mean pressure
    # Difference between actual position and boundary position
    boundary_at_point = model_dict['model'](posterior_params_mean, [positions_df['t'], positions_df['p'], positions_df['p_dyn']])
    delta_r = positions_df['r'] - boundary_at_point
    
    # If the point were on the boundary, where would it be rotated to North and w/ constant mean pressure?
    n_df = len(positions_df)
    p_dyn_mean = np.mean(positions_df['p_dyn'])
    p_to_rotate_to = 0 # North
    
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
                            figsize = (6.5, 5))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.7, top=0.98,
                        hspace=0.08)
    
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
    
    axs[0].set(xlim = (200, -600),
               ylim = (0, 500),
               aspect = 1)
    axs[0].annotate('(a)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    axs[1].set(xlim = (200, -600),
                ylim = (0, 500),
                aspect = 1)
    axs[1].annotate('(b)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')

    #   Dummy coords for plotting
    t_plot = np.linspace(0, 0.995*np.pi, 1000)
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
        label_str = r'{0} $\rightarrow$ {1} Crossing'.format(*crossing_regions) 
        label_str += '\n Rotated & Scaled'
        axs[1].scatter(subset_df.loc[boundary_entries_index, 'ell'], 
                   subset_df.loc[boundary_entries_index, 'rho'],
                   label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                   s = 16, color='#0001a7', marker='x', lw = 1,
                   zorder = 10)
        label_str = r'{1} $\rightarrow$ {0} Crossing'.format(*crossing_regions)
        label_str += '\n Rotated & Scaled'
        axs[1].scatter(subset_df.loc[boundary_exits_index, 'ell'], 
                   subset_df.loc[boundary_exits_index, 'rho'],
                   label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                   s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 1,
                   zorder = 10)
        
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
    
    direction_colors = {'NS': 'C0',
                        'DD': 'C1'}
    p_dyn_linestyles = {'16': '-',
                        '50': '--',
                        '84': ':'}

    # Bottom axes: North, to compare with translated crossings
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['NS'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['NS'])
    axs[1].plot(xyz[0], xyz[2], label = 'New Fit', 
                color='#000000', zorder=2)
    
    # Include a sense of sigma
    for params, sigma_m, sigma_b in zip(posterior_params_vals, posterior_sigma_m_vals, posterior_sigma_b_vals):
        r_coord = model_dict['model'](params, [t_coord, p_coords['DD'], p_dyn_coords['50']])
        r_coord = r_coord + (rng.normal(loc = 0, scale = sigma_m) * np.sin(t_coord/2)**2 + rng.normal(loc = 0, scale = sigma_b))
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['DD'])
        axs[1].plot(xyz[0], xyz[1], color='black', alpha=0.05, zorder=-10)
        
    # Compare to Joy+ 2002
    joy2002_params = JBC.get_JoyParameters(boundary = boundary)
    r_coord = BM.Joylike(joy2002_params, [t_coord, p_coords['NS'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['NS'])
    axs[1].plot(xyz[0], xyz[2], label = 'Joy+ (2002)',
                color='C2', zorder=0)
    
    leg = axs[1].legend(scatterpoints=3, handlelength=3,
                    loc='lower right')
    for line in leg.get_lines():
            line.set_linewidth(2.0)

    # Top axes: plot for different pressures, superimposed
    for p_dyn_value in p_dyn_coords.keys():
        for direction in ['NS', 'DD']:
            
            r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords[direction], p_dyn_coords[p_dyn_value]])
            rpl = BM.convert_SphericalSolarToCylindricalSolar(r_coord, t_coord, p_coords[direction])
            axs[0].plot(rpl[2][0:n_coords], rpl[0][0:n_coords],
                        color = direction_colors[direction], ls = p_dyn_linestyles[p_dyn_value],
                        label = '_')
            
    axs[0].plot([0,0], [-100,-100], color = direction_colors['NS'], 
                lw=2, label = 'North-South')
    axs[0].plot([0,0], [-100,-100], color = direction_colors['DD'], 
                lw=2, label = 'Dawn-Dusk')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['16'],
                lw=2, label = r'$p_{dyn}: 16^{th} \%ile$')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['50'],
                lw=2, label = r'$p_{dyn}: 50^{th} \%ile$')
    axs[0].plot([0,0], [-100,-100], color = 'black', ls = p_dyn_linestyles['84'],
                lw=2, label = r'$p_{dyn}: 84^{th} \%ile$')
    
    axs[0].legend(loc='lower right')
            
    
            
    plt.show()
    breakpoint()