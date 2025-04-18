#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:40:14 2024

@author: mrutala
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pandas as pd
pd.options.mode.copy_on_write = True
import datetime as dt
import corner
from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime
from sklearn.metrics import confusion_matrix
import BoundaryForwardModeling as BFM
import BoundaryModels as BM
import JoyBoundaryCoords as JBC
import CrossingPreprocessingRoutines as preproc

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

# Load custom plotting style
try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass
    
# Things that could be function arguments
resolution = '10Min'
end_date = '2018-08-01 00:00:00'
# delta_around_crossing = pd.Timedelta(hours=500)

sc_colors = {'Ulysses': '#910951', 
             'Galileo': '#b6544a', 
             'Cassini': '#d98b3a', 
             'Juno': '#fac205'}

def find_BoundarySurfaceLimits(df, boundary, spacecraft_to_use):
    
    # Check the boundary to set inside/outside indices
    if boundary.lower() in ['bs', 'bow shock']:
        inside_boundary_index = df.query('SH == 1 | MS == 1').index
        outside_boundary_index = df.query('SW == 1').index
    elif boundary.lower() in ['mp', 'magnetopause']:
        inside_boundary_index = df.query('MS == 1').index
        outside_boundary_index = df.query('SW == 1 | SH == 1').index
    else:
        breakpoint()
        print("Incorrect boundary specified in find_UpperLowerLimits().")
        return
        
    #   Create an array of potential standoff distances to try
    possible_standoffs = np.arange(10, 250, 1)
    
    #   For the lower bound: Shue+ model, no pressure dependence, flare exponent fixed at 0.5
    #   Cycle through increasing standoff distances until a solar wind measurement is detected
    #   The previous step sets a lower limit on the bow shock location
    fixed_fb = 0.5
    for i, standoff in enumerate(possible_standoffs):
        
        t_test = df.loc[outside_boundary_index, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test)) + 0.1
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((standoff, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = df.loc[outside_boundary_index, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_true > r_test).all() == False:
            lower_fit_params = (possible_standoffs[i-1], 0, fixed_fb, 0)
            break
        else:
            lower_fit_params = None
    
    #   For the upper bound: Shue+ model, no pressure dependence, flare exponent fixed at 1.0 (assumed by Joy+ 2002)
    #   Cycle through decreasing standoff distances until a non-solar-wind-measurement is detected
    #   The previous step sets an upper limit on the bow shock location
    fixed_fb = 1.0
    for i, standoff in enumerate(np.flip(possible_standoffs)):
        
        t_test = df.loc[inside_boundary_index, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test)) + 0.1
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((standoff, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = df.loc[inside_boundary_index, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_test > r_true).all() == False:
            upper_fit_params = (np.flip(possible_standoffs)[i-1], 0, fixed_fb, 0)
            break    
        else:
            upper_fit_params = None
    
    if (lower_fit_params is None) or (upper_fit_params is None):
        print('Failed finding one set of bounds!')
        print('This is likely due to too restrictive a range of standoff distances to test.')
    
    lower_bound_params = (lower_fit_params[0]*0.8, *lower_fit_params[1:])
    upper_bound_params = (upper_fit_params[0]*1.2, *upper_fit_params[1:])

    params = {'lowerfit': lower_fit_params, 'upperfit': upper_fit_params,
              'lowerbound': lower_bound_params, 'upperbound': upper_bound_params}
    
    return params

def PostprocessCrossings(boundary = 'BS', 
                         spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno'],
                         delta_around_crossing = dt.timedelta(hours=500),
                         other_fraction = 0.2,
                         exclusion_query = None):
    # Read Crossings
    all_positions_df = preproc.read_AllCrossings(resolution = resolution, padding = delta_around_crossing*2)
    all_positions_df = all_positions_df.query("spacecraft in @spacecraft_to_use")
    # Replace datetime index with integers, deal with duplicated rows later
    all_positions_df = all_positions_df.reset_index(names='datetime')
    
    # Drop positions when we don't know which region we're in 
    positions_df = all_positions_df.query("region != 'UN'")
    
    # Calculate the equations describing the bounds
    bound_params = find_BoundarySurfaceLimits(positions_df, boundary, spacecraft_to_use)
    
    # Add the bounds to each entry in the df
    positions_df = add_Bounds(positions_df, boundary, bound_params['upperbound'], bound_params['lowerbound'])

    # Apply exclusion criteria if necessary, *BEFORE* class imbalance handling
    if exclusion_query is not None:
        positions_df = positions_df.query(exclusion_query)
    
    # Address class imbalance by selecting regions near the crossing events
    balanced_positions_df = balance_Classes(positions_df, boundary, delta_around_crossing, other_fraction)
    
    # balanced_positions_df = add_Bounds(balanced_positions_df, boundary, bound_params['upperbound'], bound_params['lowerbound'])
    # boundary_timeseries_df = get_BoundsAsTimeSeries(balanced_positions_df)
    
    # Now add pressure information, since we have one set of bounds per time
    sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
    sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1)
    p_dyn_mme = sw_mme.filter(like = 'p_dyn').resample(resolution).interpolate()
    
    balanced_positions_df.loc[:, 'p_dyn'] = p_dyn_mme.loc[balanced_positions_df['datetime'], 'p_dyn'].to_numpy('float64')
    balanced_positions_df.loc[:, 'p_dyn_a'] = p_dyn_mme.loc[balanced_positions_df['datetime'], 'p_dyn_a'].to_numpy('float64')
    balanced_positions_df.loc[:, 'p_dyn_loc'] = p_dyn_mme.loc[balanced_positions_df['datetime'], 'p_dyn_loc'].to_numpy('float64')
    balanced_positions_df.loc[:, 'p_dyn_scale'] = p_dyn_mme.loc[balanced_positions_df['datetime'], 'p_dyn_scale'].to_numpy('float64')
    
    return balanced_positions_df

def plot_UpperLowerLimits_Spatial():
    import matplotlib.patches as patches
    from matplotlib.patches import ConnectionPatch
    from matplotlib import colors
    
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']   
    resolution = '1Min' # !!! Change back to 1 min
    
    # Read Crossings
    positions_df = preproc.read_AllCrossings(resolution = resolution, padding = dt.timedelta(hours=10)) # 1000
    positions_df = positions_df.query("spacecraft in @spacecraft_to_use")
    # Replace datetime index with integers, deal with duplicated rows later
    positions_df = positions_df.reset_index(names='datetime')
    
    BS_params = find_BoundarySurfaceLimits(positions_df, 'BS', 
                                           spacecraft_to_use)
    
    MP_params = find_BoundarySurfaceLimits(positions_df, 'MP', 
                                           spacecraft_to_use)
    
    # Show the range of possible boundary surface locations
    fig, axs = plt.subplots(nrows = 2, ncols = 2, #sharex='col',
                            figsize = (9, 4.8))
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.80, top=0.98,
                        hspace=0.08, wspace=0.12)
    
    # Set up each set of axes
    ul = (axs[0,0].get_position()._points[0,0], axs[0,0].get_position()._points[1,1]) # upper left
    lr = (axs[-1,-1].get_position()._points[1,0], axs[-1,-1].get_position()._points[0,1]) # lower right
    
    x_label_centered_x, x_label_centered_y = (ul[0] + lr[0])/2, lr[1]/2
    fig.supxlabel(r'$x_{JSS}$ [$R_J$] (+ toward Sun)', 
                  position = (x_label_centered_x, x_label_centered_y),
                  ha = 'center', va = 'top')
    
    y_label_centered_x, y_label_centered_y = ul[0]/4, (ul[1] + lr[1])/2
    fig.supylabel(r'$\rho = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]', 
                  position = (y_label_centered_x, y_label_centered_y),
                  ha = 'right', va = 'center')
    
    for ax in axs[:,0]:
        ax.set(xlim = (250, -650),
               ylim = (0, 600),
               aspect = 1)
    # for ax in axs[:,1]:
    #     ax.set(xlim = (150, -150),
    #            ylim = (0, 200),
    #            aspect = 1)
    axs[0,1].set(xlim = (150, -225),
                 ylim = (0, 250),
                 aspect = 1)
    axs[1,1].set(xlim = (150, -150),
                 ylim = (0, 200),
                 aspect = 1)
   
    for ax, annotation in zip(axs.flatten(), ['a', 'b', 'c', 'd']):
        ax.annotate('({})'.format(annotation), (0,1), (0.5,-1.5), 
                    'axes fraction', 'offset fontsize')
        
    #   Dummy coords for plotting
    t_plot = np.linspace(0, 0.995*np.pi, 1000)
    p_plot = np.zeros(1000) + np.pi/2
    p_dyn_plot = np.zeros(1000)
    coords_plot = (t_plot, p_plot, p_dyn_plot)
    
    for ax_1d, boundary, params in zip(axs, ['BS', 'MP'], [BS_params, MP_params]):
        for ax in ax_1d:
            # Plug in the params to get fit lines
            fits = {}
            for name in params.keys():
                r_fit = BM.Shuelike(params[name], coords_plot)
                rpl_fit = BM.convert_SphericalSolarToCylindricalSolar(r_fit, t_plot, p_plot)
                
                fits[name] = rpl_fit
    
            # Plot orbital trajectories for each spacecraft individually
            for spacecraft in spacecraft_to_use:
                
                subset_df = positions_df.query('spacecraft == @spacecraft')
                
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
                
                # Make DataFrames with NaNs where not in region, to make plotting easy
                inside_df = subset_df[['spacecraft', 'rho', 'phi', 'ell']]
                inside_df.loc[~inside_mask, ['rho', 'phi', 'ell']] = np.nan
                
                outside_df = subset_df[['spacecraft', 'rho', 'phi', 'ell']]
                outside_df.loc[~outside_mask, ['rho', 'phi', 'ell']] = np.nan
    
                # Ensure that we don't flip back and forth between coincident spacecraft
                ax.plot(inside_df.query("spacecraft == @spacecraft")['ell'],
                        inside_df.query("spacecraft == @spacecraft")['rho'],
                        # label = '',
                        color = sc_colors[spacecraft], lw = 1, ls = '--',
                           zorder = 9)
                ax.plot(outside_df.query("spacecraft == @spacecraft")['ell'],
                        outside_df.query("spacecraft == @spacecraft")['rho'],
                        label = '{0} Trajectory'.format(spacecraft),
                        color = sc_colors[spacecraft], lw = 1, ls = '-',
                        zorder = 9)
                
                # Crossings
                crossing_regions = ['SW', 'SH'] if boundary == 'BS' else ['SH', 'MS']
                ax.scatter(subset_df.loc[boundary_entries_index, 'ell'], 
                           subset_df.loc[boundary_entries_index, 'rho'],
                           label = r'{0} $\rightarrow$ {1} Crossing'.format(*crossing_regions) if spacecraft == spacecraft_to_use[-1] else '',
                           s = 16, color='#0001a7', marker='x', lw = 1,
                           zorder = 10)
                ax.scatter(subset_df.loc[boundary_exits_index, 'ell'], 
                           subset_df.loc[boundary_exits_index, 'rho'],
                           label = r'{1} $\rightarrow$ {0} Crossing'.format(*crossing_regions) if spacecraft == spacecraft_to_use[-1] else '',
                           s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 1,
                           zorder = 10)
            
            # Plot the fit boundaries
            ax.plot(fits['upperfit'][2], fits['upperfit'][0],
                    color = 'black', linestyle = (0, (3, 1, 1, 1)),
                    # label = 'Inclusive Fit: \n' + r'${0:.0f} \times \left( \frac{{2}}{{1 + cos(\theta)}} \right)^{{{2}}}$'.format(*params['upperfit']))
                    label = r'$v_{{S97}} = ({}, {}, {}, {})$'.format(*params['upperfit']))
            ax.plot(fits['lowerfit'][2], fits['lowerfit'][0],
                    color = 'black', linestyle = (0, (3, 1, 1, 1, 1, 1)),
                    # label = 'Exclusive Fit: \n' + r'${0:.0f} \times \left( \frac{{2}}{{1 + cos(\theta)}} \right)^{{{2}}}$'.format(*params['lowerfit']))
                    label = r'$v_{{S97}} = ({}, {}, {}, {})$'.format(*params['lowerfit']))
            
            # Shade the padded, valid region
            ax.fill(np.append(fits['upperbound'][2], np.flip(fits['lowerbound'][2])), 
                    np.append(fits['upperbound'][0], np.flip(fits['lowerbound'][0])),
                    color = 'black', alpha = 0.2, edgecolor = None,
                    label = 'Range of Possible \n{} Surfaces'.format('Bow Shock' if boundary == 'BS' else 'Magnetopause'))
            ax.plot(np.append(fits['upperbound'][2], np.flip(fits['lowerbound'][2])),
                    np.append(fits['upperbound'][0], np.flip(fits['lowerbound'][0])),
                    color = 'xkcd:black', alpha=0.5, linestyle = '-',
                    label = r'_$\pm20\%$ around fits')

        # Add the patch to the Axes
        if boundary == 'BS':
            rect = patches.Rectangle((-225, 0), 375, 250, linewidth=1, edgecolor='xkcd:cornflower blue', facecolor=colors.to_rgba("xkcd:cornflower blue", alpha=0.25), zorder=-16)
            leg_title = 'Bow Shock'
        else:
            rect = patches.Rectangle((-150, 0), 300, 200, linewidth=1, edgecolor='xkcd:cornflower blue', facecolor=colors.to_rgba("xkcd:cornflower blue", alpha=0.25), zorder=-16)
            leg_title = 'Magnetopause'
        ax_1d[0].add_patch(rect)
        # ax_1d[0].annotate(leg_title, (0, 0.5), (-4, 0), 
        #                   xycoords='axes fraction', textcoords='offset fontsize', 
        #                   rotation = 'vertical', va='center', ha='center')
        ax_1d[0].set_ylabel(leg_title, size='large')
        
        # Connect the cut-out plot and make the same color
        connector1 = ConnectionPatch(xyA=(rect._x0,0), xyB=(0,0), 
                                     coordsA="data", coordsB="axes fraction", 
                                     axesA=ax_1d[0], axesB=ax_1d[1],
                                     color="xkcd:cornflower blue", zorder=-16)
        connector2 = ConnectionPatch(xyA=(rect._x0,rect._y0+rect._height), xyB=(0,1), 
                                     coordsA="data", coordsB="axes fraction", 
                                     axesA=ax_1d[0], axesB=ax_1d[1], 
                                     color="xkcd:cornflower blue", zorder=-16)
        ax_1d[1].add_artist(connector1)
        ax_1d[1].add_artist(connector2)
        ax_1d[1].set_axisbelow(False)
        
        for spine in ['left', 'right', 'bottom', 'top']:
            ax_1d[1].spines[spine].set_color('xkcd:cornflower blue')
        
        leg = ax_1d[1].legend(scatterpoints=3, handlelength=3,
                              loc='upper left', bbox_to_anchor=(1, 1.03, 0.62, 0.0), 
                              mode = 'expand')
        for line in leg.get_lines():
                line.set_linewidth(2.0)
    
            # x_joy = np.linspace(-150, 600, 10000)
            # ps_dyn_joy = [0.08]
            # for p_dyn_joy in ps_dyn_joy:
            #     #   N-S boundary
            #     z_joy = JBC.find_JoyBoundaries(p_dyn_joy, 'BS', x = x_joy, y = 0)
            #     ax.plot(x_joy, np.abs(z_joy[0]), color = 'C1', linestyle='--', lw=1.5,
            #             label = r'N-S reference (Joy+ 2002)' ,zorder=8)
            #     y_joy = JBC.find_JoyBoundaries(p_dyn_joy, 'BS', x = x_joy, z = 0)
            #     ax.plot(x_joy, np.abs(y_joy[0]), color = 'C3', linestyle='--',
            #             label = r'Dusk reference (Joy+ 2002)', zorder=8)
            #     ax.plot(x_joy, y_joy[1], color = 'C5', linestyle='--',
            #             label = r'Dawn reference (Joy+ 2002)', zorder=8)
    
    plt.savefig("/Users/mrutala/projects/JupiterBoundaries/paper/figures/UpperLowerLimits_Spatial.png",
                dpi=300)
    plt.show()
    
    breakpoint()
    
def plot_UpperLowerLimits_Temporal(): 
    
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    delta_around_crossing = dt.timedelta(hours=500)
    other_fraction = 0.2
    
    spacecraft_color_dict = {'Ulysses': 'C0',
                             'Galileo': 'C1',
                             'Cassini': 'C3',
                             'Juno': 'C5'}
    
    bs_df = PostprocessCrossings(boundary = 'BS', 
                                 spacecraft_to_use = spacecraft_to_use,
                                 delta_around_crossing = delta_around_crossing,
                                 other_fraction = other_fraction)
    
    mp_df = PostprocessCrossings(boundary = 'MP', 
                                 spacecraft_to_use = spacecraft_to_use,
                                 delta_around_crossing = delta_around_crossing,
                                 other_fraction = other_fraction)
    
    #   Visualize
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 5))
    subfigs = fig.subfigures(nrows=2, hspace=0.04)
    
    for boundary_label, df, subfig in zip(['Bow Shock', 'Magnetopause'], [bs_df, mp_df], subfigs):
        
        # Identify gaps in the timeseries
        # This gives the index of the step BEFORE the bigger gap, you'll want to add one before subscripting the df
        diffs_in_seconds = np.diff(df['datetime']).astype('float64') * 1e-9
        all_gaps = np.argwhere(diffs_in_seconds > 600).flatten() + 1
        long_gaps = np.argwhere(diffs_in_seconds > 10*24*60*60).flatten() + 1
    
        width_ratios = np.diff(long_gaps, prepend=0, append=len(df))
    
        axs = subfig.subplots(ncols = len(long_gaps)+1, nrows = 1,
                              width_ratios = width_ratios,
                              squeeze = False)
 
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9,
                            wspace = 0.04)
    
        # Top row: bow shocks
        for start_indx, stop_indx, ax in zip(np.insert(long_gaps, 0, 0), np.append(long_gaps, len(df)), axs[0]):
            
            subset_df = df.iloc[start_indx:stop_indx]
            
            for sc in ['Ulysses', 'Galileo', 'Cassini', 'Juno']:
                ax.fill_between(subset_df.query("spacecraft == @sc")['datetime'], 
                                subset_df.query("spacecraft == @sc")['r_upperbound'].to_numpy('float64'), 
                                subset_df.query("spacecraft == @sc")['r_lowerbound'].to_numpy('float64'),
                                color='black', alpha=0.33, edgecolor=None,
                                label = 'Range of Potential \n{} Surfaces'.format(boundary_label) if sc == 'Ulysses' else '_')
            
                ax.scatter(subset_df.query("spacecraft == @sc")['datetime'], 
                           subset_df.query("spacecraft == @sc")['r'],
                           color = spacecraft_color_dict[sc], marker = '.', s = 1, 
                           label = '{} Trajectory'.format(sc))
            
            ax.set(xlim = (subset_df['datetime'].iloc[0], subset_df['datetime'].iloc[-1]))
           
            # Custom x-axis ticks
            major_tick_interval = 150 # days
            first_date = subset_df['datetime'].iloc[0]
            res = [mdates.date2num(d) for d in subset_df['datetime'] if ((d - first_date).total_seconds() % int(major_tick_interval*24*60*60)) == 0]
        
            ax.xaxis.set_major_locator(mticker.MultipleLocator(150))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%j\n%Y'))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(50))
        
            if start_indx != 0:
                ax.spines['left'].set_visible(False)
                ax.tick_params(which = 'both', left = False, labelleft = False)
                
                for coords in [(0, 0), (0, 1)]:
                    ax.annotate('/', coords, (0, -0.1), 
                                'axes fraction', 'offset fontsize', 
                                ha = 'center', va = 'center')
                
                # ax.minorticks_off()
            if stop_indx != len(df):
                ax.spines['right'].set_visible(False)
                ax.tick_params(which = 'both', right = False, labelright = False)
                
                for coords in [(1, 0), (1, 1)]:
                    ax.annotate('/', coords, (0.1,-0.1), 
                                'axes fraction', 'offset fontsize', 
                                ha = 'center', va = 'center')
                # ax.minorticks_off()
        
        for ax in axs[0]:
            ax.set(ylim = [0, 500])
            
        axs[0,-1].legend(loc = 'upper right', scatterpoints = 9)
            
        
        # ax.legend()
        fig.supxlabel('Date [Day of Year / Year]')
        fig.supylabel(r'$r_{JSS} = \sqrt{x_{JSS}^2 + y_{JSS}^2 + z_{JSS}^2} = \sqrt{x_{JSS}^2 + \rho_{JSS}^2} [R_J]$')
        # ax.set(xlabel = 'Time', 
        #        ylabel = r'$r_{JSS} = \sqrt{x_{JSS}^2 + y_{JSS}^2 + z_{JSS}^2} = \sqrt{x_{JSS}^2 + \rho_{JSS}^2} [R_J]$', ylim = (0, 500))
    
    plt.show()

    
    
    breakpoint()
def plot_Crossings():
    
    spacecraft_to_use = ['Ulysses', 'Galileo', 'Cassini', 'Juno']
    # spacecraft_to_use = ['Galileo', 'Juno']    
    
    # Read Crossings
    positions_df = preproc.read_AllCrossings(resolution = resolution, padding = dt.timedelta(hours=3000))
    positions_df = positions_df.query("spacecraft in @spacecraft_to_use")
    # Replace datetime index with integers, deal with duplicated rows later
    positions_df = positions_df.reset_index(names='datetime')
    
    # 
    fig, axs = plt.subplots(nrows = 2, ncols = 2,
                            figsize = (6.5, 5))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.7, top=0.98,
                        hspace=0.08, wspace = 0.4)

    for ax in axs:
        ax[0].set(xlabel = r'$x_{JSS}$ (+ Sunward)', xlim = (-150, +150), 
                  ylabel = r'$z_{JSS}$ (+ Northward)', ylim = (-75, +75), aspect = 2)
        ax[1].set(xlabel = r'$x_{JSS}$ (+ Sunward)', xlim = (-250, +250), 
                  ylabel = r'$y_{JSS}$ (+ Duskward)', ylim = (-250, +250), aspect = 1)

    #   Dummy coords for plotting
    t_plot = np.linspace(0, 0.995*np.pi, 1000)
    p_plot = np.zeros(1000) + np.pi/2
    p_dyn_plot = np.zeros(1000)
    coords_plot = (t_plot, p_plot, p_dyn_plot)
    
    for ax, boundary in zip(axs, ['BS', 'MP']):

        # Plot orbital trajectories for each spacecraft individually
        for spacecraft in spacecraft_to_use:
            
            subset_df = positions_df.query('spacecraft == @spacecraft')
            
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
            
            # Make DataFrames with NaNs where not in region, to make plotting easy
            inside_df = subset_df.copy(deep=True)
            inside_df.loc[~inside_mask, ['rho', 'phi', 'ell']] = np.nan
            
            outside_df = subset_df.copy(deep=True)
            outside_df.loc[~outside_mask, ['rho', 'phi', 'ell']] = np.nan

            # Ensure that we don't flip back and forth between coincident spacecraft
            ax[0].plot(inside_df.query("spacecraft == @spacecraft")['x'],
                    inside_df.query("spacecraft == @spacecraft")['z'],
                    # label = '',
                    color = sc_colors[spacecraft], lw = 0.5, ls = '--',
                       zorder = 9)
            ax[0].plot(outside_df.query("spacecraft == @spacecraft")['x'],
                    outside_df.query("spacecraft == @spacecraft")['z'],
                    label = '{}'.format(spacecraft),
                    color = sc_colors[spacecraft], lw = 0.5, ls = '-',
                    zorder = 9)
            ax[1].plot(inside_df.query("spacecraft == @spacecraft")['x'],
                    inside_df.query("spacecraft == @spacecraft")['y'],
                    # label = '',
                    color = sc_colors[spacecraft], lw = 0.5, ls = '--',
                       zorder = 9)
            ax[1].plot(outside_df.query("spacecraft == @spacecraft")['x'],
                    outside_df.query("spacecraft == @spacecraft")['y'],
                    label = '{}'.format(spacecraft),
                    color = sc_colors[spacecraft], lw = 0.5, ls = '-',
                    zorder = 9)
            
            # Crossings
            crossing_regions = ['SW', 'SH'] if boundary == 'BS' else ['SH', 'MS']
            label_str = r'{0} $\rightarrow$ {1}'.format(*crossing_regions)
            ax[0].scatter(subset_df.loc[boundary_entries_index, 'x'], 
                       subset_df.loc[boundary_entries_index, 'z'],
                       label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                       s = 16, color='#0001a7', marker='x', lw = 0.5,
                       zorder = 10)
            ax[1].scatter(subset_df.loc[boundary_entries_index, 'x'], 
                       subset_df.loc[boundary_entries_index, 'y'],
                       label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                       s = 16, color='#0001a7', marker='x', lw = 0.5,
                       zorder = 10)
            label_str = r'{1} $\rightarrow$ {0}'.format(*crossing_regions)
            ax[0].scatter(subset_df.loc[boundary_exits_index, 'x'], 
                       subset_df.loc[boundary_exits_index, 'z'],
                       label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                       s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 0.5,
                       zorder = 10)
            ax[1].scatter(subset_df.loc[boundary_exits_index, 'x'], 
                       subset_df.loc[boundary_exits_index, 'y'],
                       label = label_str if spacecraft == spacecraft_to_use[-1] else '',
                       s = 16, edgecolor='#563ae2', facecolor='None', marker='o', lw = 0.5,
                       zorder = 10)
    
    for ax in axs.flatten():
        leg = ax.legend(scatterpoints=3, handlelength=3,
                        loc='lower right', fontsize = 'x-small')
        for line in leg.get_lines():
                line.set_linewidth(2.0)
    
    plt.show()
    
    breakpoint()

def balance_Classes(df, boundary, delta, other_fraction=0.1):
    """
    Roughly balance the number of measurements inside and outside a 
    magnetospheric boundary by choosing those +/- delta from each crossing
    plus a specified fraction of all remaining measurements.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing measurements with at least two columns: 
        'datetime' for timestamps and either 'SW' for solar wind or 'MS' 
        for magnetospheric state.
    boundary : str
        The type of magnetospheric boundary to consider, either 'bs' 
        (bow shock) or 'mp' (magnetopause).
    delta : pd.Timedelta
        The time window around each boundary crossing from which to select 
        measurements.
    other_fraction : float, optional
        The fraction of remaining measurements to randomly include, default is 0.1.

    Returns
    -------
    pandas.DataFrame
        A balanced DataFrame containing only the measurements within the 
        specified boundaries and time windows, plus a random selection of 
        additional measurements to achieve balance.
    """
    # Have to do this by spacecraft, else simultaneous spacecraft mess it up
    crossings_datetimes = []
    for sc in set(df['spacecraft']):
        subset_df = df.query("spacecraft == @sc")
        if len(subset_df.query("region == 'UN'")) > 0:
            print("There shouldn't be any unknown regions to balance classes...")
            breakpoint()
            
        if boundary.lower() in ['bs', 'bow shock']:
            crossings_bool = np.diff(subset_df['SW'].to_numpy(), prepend=1) != 0
        elif boundary.lower() in ['mp', 'magnetopause']:
            crossings_bool = np.diff(subset_df['MS'].to_numpy(), prepend=0) != 0
        else:
            print("Invalid boundary selection!")
            breakpoint()
        
        crossings_datetimes.extend(subset_df.loc[crossings_bool]['datetime'])
    
    balanced_arr = np.zeros(len(df), dtype='int32')
    for datetime in crossings_datetimes:
        
        # Querying doesn't work here for some reason...
        indx = (df['datetime'] >= datetime - delta) & (df['datetime'] < datetime + delta)
        balanced_arr += indx.to_numpy('int32')
    
    balanced_bool = balanced_arr > 0
    
    # Add in some remaining measurements
    other_indx = np.argwhere(balanced_bool == False).flatten()
    rng = np.random.default_rng()
    to_add_indx = rng.choice(other_indx, int(other_fraction * len(other_indx)), replace=False)
    balanced_bool[to_add_indx] = True
    
    return df.loc[balanced_bool]

def add_Bounds(df, boundary, upperbound, lowerbound):
    
    # Double-check that the df index is unique, otherwise this won't work
    if not df.index.is_unique:
        print("Invalid dataframe index values")
        breakpoint()
        
    # Coefficients affecting pressure should be 0 (no p_dyn dependence)
    coords = [df['t'].to_numpy('float64'), df['p'].to_numpy('float64'), np.zeros(len(df))+0.1]
    df.loc[:, 'r_upperbound'] = BM.Shuelike(upperbound, coords)
    df.loc[:, 'r_lowerbound'] = BM.Shuelike(lowerbound, coords)
    
    if boundary.lower() in ['bs', 'bow shock']:
        # in_sw_below_upperbound_index = df.query("r < r_upperbound & region == 'SW'").index # set upper bound
        # not_in_sw_above_lowerbound_index = df.query("r > r_lowerbound & region != 'SW'").index # set lower bound
        # df.loc[in_sw_below_upperbound_index, 'r_upperbound'] = df.loc[in_sw_below_upperbound_index, 'r']
        # df.loc[not_in_sw_above_lowerbound_index, 'r_lowerbound'] = df.loc[not_in_sw_above_lowerbound_index, 'r']
        
        # Above the BS, but below the upperbound -> r_upperbound = r
        qry = "(region == 'SW') & (r < r_upperbound)"
        df.loc[df.query(qry).index, 'r_upperbound'] = df.loc[df.query(qry).index, 'r']
        # Below the BS, but above the lowerbound -> r_lowerbound = r
        qry = "(region == 'SH') & (r > r_lowerbound)"
        df.loc[df.query(qry).index, 'r_lowerbound'] = df.loc[df.query(qry).index, 'r']
        
    elif boundary.lower() in ['mp', 'magnetopause']:
        # Above the MP, but below the upperbound -> r_upperbound = r
        qry = "(region == 'SH') & (r < r_upperbound)"
        df.loc[df.query(qry).index, 'r_upperbound'] = df.loc[df.query(qry).index, 'r']
        # Below the MP, but above the lowerbound -> r_lowerbound = r
        qry = "(region == 'MS') & (r > r_lowerbound)"
        df.loc[df.query(qry).index, 'r_lowerbound'] = df.loc[df.query(qry).index, 'r']
    else:
        print("Invalid boundary selection!")
        breakpoint()
    
    return df

# def get_BoundsAsTimeSeries(positions_df):
#     # =============================================================================
#     # Converting the boundaries to time series
#     # =============================================================================
#     # inorout_arr = positions_df['within_bs'].to_numpy() 
#     # Make a new dataframe as a time series
#     datetimes = positions_df['datetime'].loc[positions_df['datetime'].duplicated() == False]
#     boundary_timeseries_df = pd.DataFrame(index = datetimes,
#                                           columns = ['r_lowerbound', 'r_upperbound',
#                                                       'r', 't', 'p'])
#     # Find duplicated and unique indices in positions_df
#     duplicated_index = positions_df['datetime'].duplicated(keep=False)
#     unique_index = ~duplicated_index    
#     # Where the datetime indices are unique, there's only one spacecraft
#     # So when that spacecraft is in the SW, it gives an upperbound
#     # when it's in the SH or MS, it give a lowerbound
#     # and when it's UN, we will ignore it (leave as NaN)
#     in_sw_df = positions_df[unique_index].query('SW == 1')
#     unique_upperbounds = in_sw_df['r']
#     unique_lowerbounds = BM.Shuelike(BS_lowerbound, [in_sw_df['t'], in_sw_df['p'], np.zeros(len(in_sw_df))])
#     indx = pd.Index(in_sw_df['datetime'])
#     boundary_timeseries_df.loc[indx, 'r_upperbound'] = unique_upperbounds.to_numpy('float64')
#     boundary_timeseries_df.loc[indx, 'r_lowerbound'] = unique_lowerbounds.to_numpy('float64')  
#     in_shms_df = positions_df[unique_index].query('SH == 1 | MS == 1')
#     unique_lowerbounds = in_shms_df['r']
#     unique_upperbounds = BM.Shuelike(upper_bound_params, [in_shms_df['t'], in_shms_df['p'], np.zeros(len(in_shms_df))])
#     indx = pd.Index(in_shms_df['datetime'])
#     boundary_timeseries_df.loc[indx, 'r_upperbound'] = unique_upperbounds.to_numpy('float64')
#     boundary_timeseries_df.loc[indx, 'r_lowerbound'] = unique_lowerbounds.to_numpy('float64')  
#     boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 'r'] = positions_df[unique_index]['r'].to_numpy('float64')
#     boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 't'] = positions_df[unique_index]['t'].to_numpy('float64')
#     boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 'p'] = positions_df[unique_index]['p'].to_numpy('float64')
#     # Where the datetime indices are not unique, there's multiple spacecraft
#     # We will only keep one set of these datetimes:
#     # 
#     # Here we have to loop over datetimes
#     duplicated_datetimes = pd.Index(positions_df.loc[duplicated_index, 'datetime'].drop_duplicates())
#     for datetime in duplicated_datetimes:      
#         rows = positions_df.loc[positions_df['datetime'] == datetime]       
#         if len(rows) > 2:
#             print("Cannot handle more than two simultaneous spacecraft!")
#             breakpoint()         
#         # Scale the location of the second s/c to that of the first
#         # where the 'first' is the s/c with the smallest t(heta), nearest the nose
#         prime_sc_indx = np.argmin(rows['t'])
#         prime_row = rows.iloc[prime_sc_indx]
#         secondary_row = rows.drop(rows.index[prime_sc_indx], axis='rows').iloc[0]  
#         boundary_timeseries_df.loc[datetime, 'r'] = prime_row['r']
#         boundary_timeseries_df.loc[datetime, 't'] = prime_row['t']
#         boundary_timeseries_df.loc[datetime, 'p'] = prime_row['p']
#         if prime_row['SW'] == 1:
#             prime_upperbound = prime_row['r']
#             prime_lowerbound = BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
#         elif (prime_row['SH'] == 1) | (prime_row['MS']) == 1:
#             prime_upperbound = BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])
#             prime_lowerbound = prime_row['r']
#         else:
#             prime_upperbound, prime_lowerbound = np.nan, np.nan
#         # Scale the secondary
#         if secondary_row['SW'] == 1:
#             secondary_upperbound = secondary_row['r']
#             secondary_lowerbound = BM.Shuelike(lower_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
#             # # Scale with the flaring of the lower bound ensures that the bound is higher than the prime_lowerbound
#             # custom_scaling_params = [upper_bound_params[0], 0, lower_bound_params[2], 0]
#             # scale_factor = secondary_upperbound/BM.Shuelike(upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
#             # secondary_upperbound = scale_factor * BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])  
#         elif (secondary_row['SH'] == 1) | (secondary_row['MS'] == 1):
#             secondary_upperbound = BM.Shuelike(upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
#             secondary_lowerbound = secondary_row['r']
#             # # Scale with the flaring of the upper bound ensures that the bound is lower than the prime_upperbound
#             # custom_scaling_params = [lower_bound_params[0], 0, upper_bound_params[2], 0]
#             # scale_factor = secondary_lowerbound/BM.Shuelike(custom_scaling_params, [secondary_row['t'], secondary_row['p'], 0.0])
#             # secondary_lowerbound = scale_factor * BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
#         else:
#             secondary_upperbound, secondary_lowerbound = np.nan, np.nan
#         # Scale the secondary
#         mod_upper_bound_params = [upper_bound_params[0], 0, lower_bound_params[2], 0]
#         mod_lower_bound_params = [lower_bound_params[0], 0, upper_bound_params[2], 0]
#         secondary_upperbound = (secondary_upperbound / BM.Shuelike(mod_upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])) * BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])
#         secondary_lowerbound = (secondary_lowerbound / BM.Shuelike(mod_lower_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])) * BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
#         possible_upperbounds = [prime_upperbound, secondary_upperbound]
#         possible_lowerbounds = [prime_lowerbound, secondary_lowerbound]
#         if np.isnan(possible_upperbounds).all() == False:
#             boundary_timeseries_df.loc[datetime, 'r_upperbound'] = np.nanmin(possible_upperbounds)
#         else:
#             breakpoint()
#         if np.isnan(possible_lowerbounds).all() == False:
#             boundary_timeseries_df.loc[datetime, 'r_lowerbound'] = np.nanmax(possible_lowerbounds)
#         else:
#             breakpoint()
#         if np.nanmax([prime_lowerbound, secondary_lowerbound]) > np.nanmin([prime_upperbound, secondary_upperbound]):
#             print('ISSUE')
#             breakpoint()
#     # FINALLY, drop NaNs-- these occur when we are uncertain about the s/c region
#     boundary_timeseries_df = boundary_timeseries_df.dropna(axis='index', how='any') 
#     diff_test = boundary_timeseries_df['r_upperbound'] - boundary_timeseries_df['r_lowerbound']
#     if (diff_test < 0).any():
#         print('Something went wrong!')
#         breakpoint()   
#     return boundary_timeseries_df