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

import CrossingPreprocessingRoutines as CPR

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

# Load custom plotting style
try:
    plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
except:
    pass

resolution = '10Min'

# Read Crossings
positions_df = CPR.read_AllCrossings(resolution = resolution, padding = dt.timedelta(hours=1000))

positions_df = positions_df.query('spacecraft != "Galileo"')

# # Add solar wind dynamic pressure from MMESH
# sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
# sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1)
# p_dyn_mme = sw_mme.filter(like = 'p_dyn').resample(resolution).interpolate()

# positions_df = pd.concat([positions_df, p_dyn_mme.loc[positions_df.index]], axis='columns')

# Finally, replace datetime index with integers, to deal with duplicated rows
positions_df = positions_df.reset_index(names='datetime')


model_name = 'Shuelike'
model_dict = BM.init(model_name)
model_number = model_dict['model_number']

def find_UpperLowerLimits(positions_df):
        
    # Reset index to deal with duplicated datetimes
    df = positions_df
    
    # Find when the spacecraft is (not) in the solar wind
    in_sw_index = df.query('SW == 1').index
    not_in_sw_index = df.query('SH == 1 | MS == 1').index
    
    #   Create an array of potential standoff distances to try
    possible_standoffs = np.arange(50, 150, 0.5)
    
    #   For the lower bound: Shue+ model, no pressure dependence, flare exponent fixed at 0.5
    #   Cycle through increasing standoff distances until a solar wind measurement is detected
    #   The previous step sets a lower limit on the bow shock location
    fixed_fb = 0.5
    for i, standoff in enumerate(possible_standoffs):
        
        t_test = df.loc[in_sw_index, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test))
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((standoff, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = df.loc[in_sw_index, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_true > r_test).all() == False:
            lower_bound_params = (possible_standoffs[i-1], 0, fixed_fb, 0)
            break
        else:
            lower_bound_params = None
    
    #   For the upper bound: Shue+ model, no pressure dependence, flare exponent fixed at 0.8
    #   !!!! Set this flare based on a fit of the Shue model to Joy model
    #   Cycle through decreasing standoff distances until a non-solar-wind-measurement is detected
    #   The previous step sets an upper limit on the bow shock location
    fixed_fb = 0.8
    for i, standoff in enumerate(np.flip(possible_standoffs)):
        
        t_test = df.loc[not_in_sw_index, 't'].to_numpy()
        p_test = np.zeros(len(t_test))
        p_dyn_test = np.zeros(len(t_test))
        
        #   For the test coordinates, where the boundary surface r is located
        r_test = BM.Shuelike((standoff, 0, fixed_fb, 0), 
                             (t_test, p_test, p_dyn_test))
        
        #   Where the spacecraft actually is
        r_true = df.loc[not_in_sw_index, 'r'].to_numpy()
        
        #   We know we're in the solar wind, so r_true must be greater than r_test
        #   If it's not, then r_test is too large-- so the previous value
        #   of r_totry is the innermost valid boundary surface
        if (r_test > r_true).all() == False:
            upper_bound_params = (np.flip(possible_standoffs)[i-1], 0, fixed_fb, 0)
            break    
        else:
            upper_bound_params = None
    
    if (lower_bound_params is None) or (upper_bound_params is None):
        print('Failed finding one set of bounds!')
        print('This is likely due to too restrictive a range of standoff distances to test.')
        
    return lower_bound_params, upper_bound_params
    
lower_fit_params, upper_fit_params = find_UpperLowerLimits(positions_df)

lower_bound_params = (lower_fit_params[0]*0.8, *lower_fit_params[1:])
upper_bound_params = (upper_fit_params[0]*1.2, *upper_fit_params[1:])
    
#   Show the range of possible boundary surface locations
fig, ax = plt.subplots()

#   Dummy coords for plotting
t_fc = np.linspace(0, 0.95*np.pi, 1000)
p_fc = np.zeros(1000) + np.pi/2
p_dyn_fc = np.zeros(1000)

#   radial distance at lower fit, lower bound, upper fit, and upper bound
r_lower_fit = model_dict['model'](lower_fit_params, (t_fc, p_fc, p_dyn_fc))
rpl_lower_fit = BM.convert_SphericalSolarToCylindricalSolar(r_lower_fit, t_fc, p_fc)

r_lower_bound = model_dict['model'](lower_bound_params, (t_fc, p_fc, p_dyn_fc))
rpl_lower_bound = BM.convert_SphericalSolarToCylindricalSolar(r_lower_bound, t_fc, p_fc)

r_upper_fit = model_dict['model'](upper_fit_params, (t_fc, p_fc, p_dyn_fc))
rpl_upper_fit = BM.convert_SphericalSolarToCylindricalSolar(r_upper_fit, t_fc, p_fc)

r_upper_bound = model_dict['model'](upper_bound_params, (t_fc, p_fc, p_dyn_fc))
rpl_upper_bound = BM.convert_SphericalSolarToCylindricalSolar(r_upper_bound, t_fc, p_fc)

#   Plot the fit boundaries
ax.plot(rpl_upper_fit[2], rpl_upper_fit[0],
        color = 'black', linestyle = (0, (3, 1, 1, 1)),
        label = r'Inclusive Fit: ${:.2f} \left( \frac{{2}}{{1 + cos(\theta)}} \right)^{{{}}}$'.format(upper_bound_params[0], upper_bound_params[2]))
ax.plot(rpl_lower_fit[2], rpl_lower_fit[0],
        color = 'black', linestyle = (0, (3, 1, 1, 1, 1, 1)),
        label = r'Exclusive Fit: ${:.2f} \left( \frac{{2}}{{1 + cos(\theta)}} \right)^{{{}}}$'.format(lower_bound_params[0], lower_bound_params[2]))

#   Shade the bound region
ax.fill(np.append(rpl_upper_bound[2], rpl_lower_bound[2][::-1]), 
        np.append(rpl_upper_bound[0], rpl_lower_bound[0][::-1]),
        color = 'black', alpha = 0.2, edgecolor = None,
        label = 'Considered Range of Possible \nBow Shock Surfaces')
ax.plot(np.append(rpl_upper_bound[2], np.flip(rpl_lower_bound[2])),
        np.append(rpl_upper_bound[0], np.flip(rpl_lower_bound[0])),
        color = 'xkcd:black', alpha=0.5, linestyle = '-',
        label = r'$\pm20\%$ around fits')

#   Plot locations of solar wind measurements
#   Times when the spacecraft is in the solar wind
in_sw_index = positions_df.query('SW == 1').index

sw_entries_index = (positions_df['SW'].shift(1) == 0) & (positions_df['SW'] == 1)
sw_exits_index = (positions_df['SW'].shift(1) == 1) & (positions_df['SW'] == 0)

ax.scatter(positions_df.loc[sw_entries_index, 'ell'], 
           positions_df.loc[sw_entries_index, 'rho'],
           label = 'Entry into the Solar Wind (Outbound BS Crossing)',
           s = 24, color='C0', marker='o',
           zorder = 10)
ax.scatter(positions_df.loc[sw_exits_index, 'ell'], 
           positions_df.loc[sw_exits_index, 'rho'],
           label = 'Exit from the Solar Wind (Inbound BS Crossing)',
           s = 24, color='C4', marker='x',
           zorder = 10)

in_sh_index = positions_df.query('SH == 1').index
in_ms_index = positions_df.query('MS == 1').index
ax.scatter(positions_df.loc[in_sw_index, 'ell'],
           positions_df.loc[in_sw_index, 'rho'],
           label = 'Solar Wind Measurements',
           s = 0.5, color = '#777777', marker = '.', lw=0.5,
           zorder = 9)
ax.scatter(positions_df.loc[in_sh_index, 'ell'],
           positions_df.loc[in_sh_index, 'rho'],
           label = 'Magnetosheath Measurements',
           s = 0.5, color = '#aaaaaa', marker = '.', lw=0.5,
           zorder = 9)
ax.scatter(positions_df.loc[in_ms_index, 'ell'],
           positions_df.loc[in_ms_index, 'rho'],
           label = 'Magnetosphere Measurements',
           s = 0.5, color = '#cccccc', marker = '.', lw=0.5,
           zorder = 9)

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
    

ax.set(xlim = (200, -600), xlabel = r'$x_{JSS}$ [$R_J$] (+ toward Sun)',
       ylim = (0, 600), ylabel = r'$\rho_{JSS} = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]',
       aspect = 1)

ax.legend(scatterpoints=3, handlelength=3)
plt.show()

def get_BoundsAsTimeSeries(positions_df):
    # =============================================================================
    # Converting the boundaries to time series
    # =============================================================================
    # inorout_arr = positions_df['within_bs'].to_numpy()
    
    # Make a new dataframe as a time series
    datetimes = positions_df['datetime'].loc[positions_df['datetime'].duplicated() == False]
    boundary_timeseries_df = pd.DataFrame(index = datetimes,
                                          columns = ['r_lowerbound', 'r_upperbound',
                                                     'r', 't', 'p'])

    # Find duplicated and unique indices in positions_df
    duplicated_index = positions_df['datetime'].duplicated(keep=False)
    unique_index = ~duplicated_index
    
    # Where the datetime indices are unique, there's only one spacecraft
    # So when that spacecraft is in the SW, it gives an upperbound
    # when it's in the SH or MS, it give a lowerbound
    # and when it's UN, we will ignore it (leave as NaN)
    in_sw_df = positions_df[unique_index].query('SW == 1')
    unique_upperbounds = in_sw_df['r']
    unique_lowerbounds = BM.Shuelike(lower_bound_params, [in_sw_df['t'], in_sw_df['p'], np.zeros(len(in_sw_df))])
    indx = pd.Index(in_sw_df['datetime'])
    boundary_timeseries_df.loc[indx, 'r_upperbound'] = unique_upperbounds.to_numpy('float64')
    boundary_timeseries_df.loc[indx, 'r_lowerbound'] = unique_lowerbounds.to_numpy('float64')
    
    in_shms_df = positions_df[unique_index].query('SH == 1 | MS == 1')
    unique_lowerbounds = in_shms_df['r']
    unique_upperbounds = BM.Shuelike(upper_bound_params, [in_shms_df['t'], in_shms_df['p'], np.zeros(len(in_shms_df))])
    indx = pd.Index(in_shms_df['datetime'])
    boundary_timeseries_df.loc[indx, 'r_upperbound'] = unique_upperbounds.to_numpy('float64')
    boundary_timeseries_df.loc[indx, 'r_lowerbound'] = unique_lowerbounds.to_numpy('float64')
    
    boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 'r'] = positions_df[unique_index]['r'].to_numpy('float64')
    boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 't'] = positions_df[unique_index]['t'].to_numpy('float64')
    boundary_timeseries_df.loc[positions_df[unique_index]['datetime'], 'p'] = positions_df[unique_index]['p'].to_numpy('float64')

    # Where the datetime indices are not unique, there's multiple spacecraft
    # We will only keep one set of these datetimes:
    # 
    # Here we have to loop over datetimes
    duplicated_datetimes = pd.Index(positions_df.loc[duplicated_index, 'datetime'].drop_duplicates())
    for datetime in duplicated_datetimes:
        
        rows = positions_df.loc[positions_df['datetime'] == datetime]
        
        if len(rows) > 2:
            print("Cannot handle more than two simultaneous spacecraft!")
            breakpoint()
            
        # Scale the location of the second s/c to that of the first
        # where the 'first' is the s/c with the smallest t(heta), nearest the nose
        prime_sc_indx = np.argmin(rows['t'])
        prime_row = rows.iloc[prime_sc_indx]
        secondary_row = rows.drop(rows.index[prime_sc_indx], axis='rows').iloc[0]
        
        boundary_timeseries_df.loc[datetime, 'r'] = prime_row['r']
        boundary_timeseries_df.loc[datetime, 't'] = prime_row['t']
        boundary_timeseries_df.loc[datetime, 'p'] = prime_row['p']
        
        if prime_row['SW'] == 1:
            prime_upperbound = prime_row['r']
            prime_lowerbound = BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
        elif (prime_row['SH'] == 1) | (prime_row['MS']) == 1:
            prime_upperbound = BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])
            prime_lowerbound = prime_row['r']
        else:
            prime_upperbound, prime_lowerbound = np.nan, np.nan
            
        # Scale the secondary
        if secondary_row['SW'] == 1:
            secondary_upperbound = secondary_row['r']
            secondary_lowerbound = BM.Shuelike(lower_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
            
            # # Scale with the flaring of the lower bound ensures that the bound is higher than the prime_lowerbound
            # custom_scaling_params = [upper_bound_params[0], 0, lower_bound_params[2], 0]
            # scale_factor = secondary_upperbound/BM.Shuelike(upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
            # secondary_upperbound = scale_factor * BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])
            
        elif (secondary_row['SH'] == 1) | (secondary_row['MS'] == 1):
            secondary_upperbound = BM.Shuelike(upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])
            secondary_lowerbound = secondary_row['r']
            
            # # Scale with the flaring of the upper bound ensures that the bound is lower than the prime_upperbound
            # custom_scaling_params = [lower_bound_params[0], 0, upper_bound_params[2], 0]
            # scale_factor = secondary_lowerbound/BM.Shuelike(custom_scaling_params, [secondary_row['t'], secondary_row['p'], 0.0])
            # secondary_lowerbound = scale_factor * BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
            
        else:
            secondary_upperbound, secondary_lowerbound = np.nan, np.nan
            
        # Scale the secondary
        mod_upper_bound_params = [upper_bound_params[0], 0, lower_bound_params[2], 0]
        mod_lower_bound_params = [lower_bound_params[0], 0, upper_bound_params[2], 0]
        secondary_upperbound = (secondary_upperbound / BM.Shuelike(mod_upper_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])) * BM.Shuelike(upper_bound_params, [prime_row['t'], prime_row['p'], 0.0])
        secondary_lowerbound = (secondary_lowerbound / BM.Shuelike(mod_lower_bound_params, [secondary_row['t'], secondary_row['p'], 0.0])) * BM.Shuelike(lower_bound_params, [prime_row['t'], prime_row['p'], 0.0])
        
        possible_upperbounds = [prime_upperbound, secondary_upperbound]
        possible_lowerbounds = [prime_lowerbound, secondary_lowerbound]
        
        if np.isnan(possible_upperbounds).all() == False:
            boundary_timeseries_df.loc[datetime, 'r_upperbound'] = np.nanmin(possible_upperbounds)
        else:
            breakpoint()
            
        if np.isnan(possible_lowerbounds).all() == False:
            boundary_timeseries_df.loc[datetime, 'r_lowerbound'] = np.nanmax(possible_lowerbounds)
        else:
            breakpoint()
        
        if np.nanmax([prime_lowerbound, secondary_lowerbound]) > np.nanmin([prime_upperbound, secondary_upperbound]):
            print('ISSUE')
            breakpoint()
    
    # FINALLY, drop NaNs-- these occur when we are uncertain about the s/c region
    boundary_timeseries_df = boundary_timeseries_df.dropna(axis='index', how='any')
    
    diff_test = boundary_timeseries_df['r_upperbound'] - boundary_timeseries_df['r_lowerbound']
    if (diff_test < 0).any():
        print('Something went wrong!')
        breakpoint()
    
    return boundary_timeseries_df

# Before getting bounds as time series, address class imbalance
solarwind_crossings_bool = np.diff(positions_df['SW'].to_numpy(), prepend=1) != 0
solarwind_crossing_datetimes = positions_df.loc[solarwind_crossings_bool]['datetime']

delta = pd.Timedelta(hours=1000)

balanced_bool = np.full(len(positions_df), False)
for datetime in solarwind_crossing_datetimes:
    
    # Querying doesn't work here for some reason...
    indx = (positions_df['datetime'] >= datetime - delta) & (positions_df['datetime'] < datetime + delta)
    balanced_bool += indx.to_numpy()

balanced_positions_df = positions_df.loc[balanced_bool]
    
boundary_timeseries_df = get_BoundsAsTimeSeries(balanced_positions_df)

# Now add pressure information, since we have one set of bounds per time
sw_fullfilepath = '/Users/mrutala/projects/MMESH_runs/JupiterMME_ForBoundaries/output/JupiterMME_ForJupiterBoundaries.csv'
sw_mme = MMESH_reader.fromFile(sw_fullfilepath).xs('ensemble', level=0, axis=1)
p_dyn_mme = sw_mme.filter(like = 'p_dyn').resample(resolution).interpolate()

boundary_timeseries_df = pd.concat([boundary_timeseries_df, p_dyn_mme.loc[boundary_timeseries_df.index]], axis='columns')


#   Visualize
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Identify gaps in the timeseries
# This gives the index of the step BEFORE the bigger gap, you'll want to add one before subscripting the df
diffs_in_seconds = np.diff(boundary_timeseries_df.index).astype('float64') * 1e-9
all_gaps = np.argwhere(diffs_in_seconds > 600).flatten() + 1
long_gaps = np.argwhere(diffs_in_seconds > 10*24*60*60).flatten() + 1

width_ratios = np.diff(long_gaps, prepend=0, append=len(boundary_timeseries_df))

fig, axs = plt.subplots(ncols = len(long_gaps)+1,
                        nrows = 2,
                        sharey='row',
                        width_ratios = width_ratios,
                        figsize = (10, 5),
                        squeeze = False)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.985, top=0.975,
                    hspace = 0.2, wspace = 0.15)

spacecraft_color_dict = {'Ulysses': 'C0',
                         'Galileo': 'C1',
                         'Cassini': 'C3',
                         'Juno': 'C5'}
# color_by_spacecraft = [spacecraft_color_dict[sc] for sc in positions_df['spacecraft']]

# Top row: bow shocks
for start_indx, stop_indx, ax in zip(np.insert(long_gaps, 0, 0), np.append(long_gaps, len(boundary_timeseries_df)), axs[0]):
    
    subset_df = boundary_timeseries_df.iloc[start_indx:stop_indx]
    
    ax.fill_between(subset_df.index, 
                    subset_df['r_upperbound'].to_numpy('float64'), 
                    subset_df['r_lowerbound'].to_numpy('float64'),
                    color='black', alpha=0.33, edgecolor=None,
                    label = 'Range of Potential \nBow Shock Surfaces')
    
    for sc in ['Ulysses', 'Galileo', 'Cassini', 'Juno']:
        ax.scatter(positions_df.query("spacecraft == @sc")['datetime'], 
                   positions_df.query("spacecraft == @sc")['r'],
                   color = spacecraft_color_dict[sc], marker = '.', s = 1, 
                   label = '{} Trajectory'.format(sc))
    
    ax.set(xlim = (subset_df.index[0], subset_df.index[-1]))
    
    # Custom x-axis ticks
    major_tick_interval = 50 # days
    first_date = subset_df.index[0]
    res = [mdates.date2num(d) for d in subset_df.index if ((d - first_date).total_seconds() % int(major_tick_interval*24*60*60)) == 0]

    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%j\n%Y'))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    if start_indx != 0:
        ax.spines['left'].set_visible(False)
        ax.tick_params(which = 'both', left = False, labelleft = False)
        
        for coords in [(0, 0), (0, 1)]:
            ax.annotate('/', coords, (0, -0.1), 
                        'axes fraction', 'offset fontsize', 
                        ha = 'center', va = 'center')
        
        
        # ax.minorticks_off()
    if stop_indx != len(boundary_timeseries_df):
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

# breakpoint()

# #   Trim data to account for class imbalance
# #   We have way more times inside than outside (1s vs 0s)
# #   We're only going to look at times when we're outside, 
# #   plus the times inside before and after of equal lengths 
# #   Isolate chunks where we are outside the boundary
# outside = np.argwhere(inorout_arr == 0).flatten()
# list_of_outsides = np.split(outside, np.argwhere(np.diff(outside) != 1).flatten()+1)
# balanced_indices = []
# for outside_chunk in list_of_outsides:
#     duration = len(outside_chunk)
#     inside_chunk_left_indx = np.arange(outside_chunk[0]-duration-1,
#                                           outside_chunk[0]-1)
#     inside_chunk_right_indx = np.arange(outside_chunk[-1]+1,
#                                         outside_chunk[-1]+duration+1)
#     balanced_indices.extend(inside_chunk_left_indx)
#     balanced_indices.extend(outside_chunk)
#     balanced_indices.extend(inside_chunk_right_indx)
# # #   Quasi-balanced, but with more sampling further down tail:
# # #   Sample only near apojove
# # balanced_indices = np.argwhere(r > 70).flatten()
# # balanced_df = coordinate_df.query('r > 90 & ell > -100')
# # balanced_indices.extend([coordinate_df.index.get_loc(i) for i in balanced_df.index])=
# apojove_indx = np.argwhere((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:])).flatten() + 1
# balanced_indices.extend([i + step for i in apojove_indx for step in np.arange(-72, 72)])    #   +/- 12 hours (in ten minute intervals)
# #   Make sure theres no negative or repeated indices
# balanced_indices = np.array(list(set(balanced_indices)))
# balanced_indices = balanced_indices[balanced_indices >= 0]
# balanced_indices = np.sort(balanced_indices)
# n_balanced = len(balanced_indices)
# time_balanced = location_df.index[balanced_indices]
# r_balanced = r[balanced_indices]
# t_balanced = t[balanced_indices]
# p_balanced = p[balanced_indices]  
# p_dyn_loc_balanced = p_dyn_loc[balanced_indices]
# r_surface_lower_balanced = r_surface_lower[balanced_indices]
# r_surface_upper_balanced = r_surface_upper[balanced_indices]
# breakpoint()

# p_dyn_alpha =  boundary_timeseries_df['p_dyn_a'].to_numpy('float64')
from scipy.stats import skewnorm
# p_dyn_mu = skewnorm.mean(loc = boundary_timeseries_df['p_dyn_loc'].to_numpy('float64'), 
#                          scale = boundary_timeseries_df['p_dyn_scale'].to_numpy('float64'), 
#                          a = boundary_timeseries_df['p_dyn_a'].to_numpy('float64'))

# p_dyn_sigma = skewnorm.std(loc = boundary_timeseries_df['p_dyn_loc'].to_numpy('float64'), 
#                            scale = boundary_timeseries_df['p_dyn_scale'].to_numpy('float64'), 
#                            a = boundary_timeseries_df['p_dyn_a'].to_numpy('float64'))

# test = pm.Truncated.dist(pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha), lower=0.0001)
# test_draws = pm.draw(test, draws=100)

# breakpoint()

# possible_p_dyn_indx = np.sum(test_draws < 0, axis=0) == 0

# boundary_timeseries_df = boundary_timeseries_df.loc[possible_p_dyn_indx]

# coords = positions_df.loc[:, ['r', 't', 'p', 'p_dyn']].to_numpy()
# r, t, p, p_dyn_loc = coords.T
#   Think about what this is!
sigma = 1

# FOR TESTING, TO MAKE THE MCMC SAMPLER RUN FASTER
# boundary_timeseries_df = boundary_timeseries_df.sample(frac=0.05, axis='rows')

r = boundary_timeseries_df['r'].to_numpy('float64')
t = boundary_timeseries_df['t'].to_numpy('float64')
p = boundary_timeseries_df['p'].to_numpy('float64')

r_lower = boundary_timeseries_df['r_lowerbound'].to_numpy('float64')
r_upper = boundary_timeseries_df['r_upperbound'].to_numpy('float64')

p_dyn_loc   = boundary_timeseries_df['p_dyn_loc'].to_numpy('float64')
p_dyn_scale = boundary_timeseries_df['p_dyn_scale'].to_numpy('float64')
p_dyn_alpha = boundary_timeseries_df['p_dyn_a'].to_numpy('float64')
p_dyn_mu    = skewnorm.mean(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)
p_dyn_sigma = skewnorm.std(loc = p_dyn_loc, scale = p_dyn_scale, a = p_dyn_alpha)

test_pressure_dist = pm.SkewNormal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha)
test_pressure_draws = pm.draw(test_pressure_dist, draws=1000)

with pm.Model() as test_potential:
    
    # p_dyn_dist = pm.Normal.dist(mu = p_dyn_mu, sigma = p_dyn_sigma) #, alpha = p_dyn_alpha)
    # p_dyn = pm.Truncated("p_dyn", p_dyn_dist, lower = 0)
    
    # p_dyn_mu_latent = pm.Gamma('p_dyn_mu_latent', mu = np.mean(p_dyn_mu), sigma = np.std(p_dyn_mu), shape=p_dyn_mu.shape[0], initval = p_dyn_mu)
    # p_dyn_sigma_latent = pm.HalfNormal('p_dyn_sigma_latent', sigma = np.mean(p_dyn_sigma), shape=p_dyn_sigma.shape[0], initval = p_dyn_sigma)
    
    # p_dyn = pm.Normal('p_dyn', mu=p_dyn_mu_latent, sigma=p_dyn_sigma_latent, observed=p_dyn_mu, shape=p_dyn_mu.shape[0])
    # p_dyn_obs = pm.SkewNormal('p_dyn_obs', mu = p_dyn_mu_latent, sigma = p_dyn_sigma_latent, alpha = p_dyn_alpha, initval = p_dyn_mu)
    # negative_p_dyn_penalty = pm.Potential("negative_p_dyn_penalty", pm.math.switch(p_dyn_obs < 0, -np.inf, 0))
    # negative_p_dyn_penalty = pm.Potential("negative_p_dyn_penalty", pm.math.switch(p_dyn_obs < 0, (p_dyn_obs * 10), 0))
    
    # p_dyn_obs = pm.SkewNormal('p_dyn', mu = p_dyn_mu, sigma = p_dyn_sigma, alpha = p_dyn_alpha, shape = p_dyn_mu.shape[0])
    p_dyn_obs = pm.Normal('p_dyn', mu = p_dyn_mu, sigma = p_dyn_sigma, shape = p_dyn_mu.shape[0])
    negative_p_dyn_penalty = pm.Potential("negative_p_dyn_penalty", pm.math.switch(p_dyn_obs < 0, (p_dyn_obs * 10), 0))
    
    r0 = pm.Uniform("r0", 30, 200)
    r1 = pm.Uniform("r1", -1/2, 1/2)
    r2 = pm.Uniform("r2", -20, 0)
    r3 = pm.Uniform("r3", -20, 20)
    a0 = pm.Uniform("a0", 0, 2)
    a1 = pm.Uniform("a1", -2, 2)
    
    # r_b = pm.Deterministic("r_b", r0 * (p_dyn_obs) ** r1)
    r_b = pm.Deterministic("r_b", (r0 + r2*np.cos(p)**2 + r3*np.sin(p)*np.sin(t))*((p_dyn_obs)**r1))
    a_b = pm.Deterministic("a_b", a0 + a1 * (p_dyn_obs))
    
    mu = pm.Deterministic("mu", r_b * (2/(1 + np.cos(t)))**a_b)
    
    r_obs = pm.Uniform("r_obs", lower = r_lower, upper = r_upper)
    
    sigma = pm.HalfNormal("sigma", sigma = 10)
    
    likelihood = pm.Potential("likelihood", pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), value=r_obs))

with test_potential:
    idata = pm.sample(tune=1000, draws=1000, chains=4, cores=3, 
                      var_names = ['r0', 'r1', 'r2', 'r3', 'a0', 'a1', 'sigma'],
                      target_accept=0.99,
                      init = 'adapt_diag') # prevents 'jitter', which might move points init vals around too much here
    
#   Get the posterior from the inference data
posterior = idata.posterior

# #   Select only the variables which don't change with time/spacecraft position
# mod_posterior = posterior.drop_vars(['p_dyn_mu_latent', 'p_dyn_sigma_latent', 
#                                      'r_b', 'a_b', 'mu', 'r_obs'])

#   Plot a trace to check for convergence
az.plot_trace(posterior)

#   Plot a corner plot to investigate covariance
figure = corner.corner(posterior, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True,)
figure.set_size_inches(8,8)

plt.show()
    
breakpoint()
    
    
with pm.Model() as potential_model:
     
    # p_dyn = p_dyn_mu
    p_dyn = pm.Uniform('p_dyn', lower = p_dyn_mu - p_dyn_sigma, upper = p_dyn_mu + p_dyn_sigma)
    # p_dyn_dist = pm.SkewNormal.dist(mu=p_dyn_mu, sigma=p_dyn_sigma, alpha=p_dyn_alpha)
    # p_dyn_dist = pm.Normal.dist(mu=p_dyn_mu, sigma=p_dyn_sigma)
    # p_dyn = pm.Truncated("p_dyn", p_dyn_dist, lower=0.0001)
    # p_dyn = pm.Normal('p_dyn', mu = p_dyn_mu, sigma = p_dyn_sigma)
    # p_dyn = pm.SkewNormal('p_dyn', mu = p_dyn_mu, sigma = p_dyn_var**2, alpha = p_dyn_alpha)
    
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
    r_b = pm.Deterministic("r_b", param_dict['r0'] * ((p_dyn)**param_dict['r1']))
    f_b = pm.Deterministic("f_b", param_dict['a0'] + param_dict['a1']*p_dyn)
    mu = pm.Deterministic("mu", r_b * (2/(1 + pm.math.cos(t)))**f_b)
   
    # mu = pm.Deterministic("mu", model_dict['model'](params, [boundary_timeseries_df['t'].to_numpy('float64'), boundary_timeseries_df['p'].to_numpy('float64'), p_dyn]))
    
    
    # r_b = param_dict['r0'] * ((p_dyn)**param_dict['r1'])
    # f_b = param_dict['a0'] + param_dict['a1']*p_dyn
    # mu = r_b * (2/(1 + pm.math.cos(boundary_timeseries_df['t'].to_numpy('float64'))))**f_b
    
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
                            lower = boundary_timeseries_df['r_lowerbound'].to_numpy('float64'),
                            upper = boundary_timeseries_df['r_upperbound'].to_numpy('float64'))
    
    # observed_r = (boundary_timeseries_df['r_lowerbound'].to_numpy('float64') + boundary_timeseries_df['r_upperbound'].to_numpy('float64'))/2.
    
    # Define likelihood
    # likelihood = pm.Normal("obs", mu = mu - observed_r, 
    #                         sigma=sigma_dist, 
    #                         observed = np.zeros(len(boundary_timeseries_df)))
    
    likelihood = pm.Potential("obs", pm.logp(pm.Normal.dist(mu=mu, sigma=sigma_dist), value=observed_r))


    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    # idata = sample(3000)
    
    idata = pm.sample(tune=1000, draws=1000, chains=4, cores=4, target_accept=0.95) #, var_names = ['r0', 'r1', 'a0', 'a1'])
    
    # posterior = pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    
    # coords_dist = coords.T
    
    # # use a Potential instead of a CustomDist
    # pm.Potential("likelihood", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    
    # # pred = pm.Potential("pred", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    # # #pred = pm.Deterministic("pred", custom_dist_loglike(data, params, sigma_dist, coords_dist, model_number))
    
    # # likelihood = pm.Normal("likelihood", pred, sigma_dist, observed=coords.T[0])

breakpoint()

#   Get the posterior from the inference data
posterior = idata.posterior

#   Select only the variables which don't change with time/spacecraft position
mod_posterior = posterior.drop_vars(['observed_r', 'f_b', 'r_b', 'mu'])

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
                              figsize = (9,6))
#   Plot a posterior predictive of the time series in the bottom panel
c_limit = axd['c'].scatter(np.array([time_balanced]*50), obs_r_s.T,
                           color='gray', alpha=0.05, marker='o', ec=None, s=1, zorder=-10,
                           label = 'Refined Range of Possible \nBow Shock Surfaces')
c_bound = axd['c'].scatter(np.array([time_balanced]*50), mu_s.T, 
                           color='C0', alpha=1, marker='.', s=1, zorder=10,
                           label = 'Modeled Bow Shock Locations')

c_orbit = axd['c'].plot(coordinate_df.index, coordinate_df['r'], 
                        color='C4', lw=1, zorder=2, ls='--',
                        label = 'Spacecraft Position')

axd['c'].set(xlabel='Date', 
             ylim=(80, 200), ylabel=r'Radial Distance $[R_J]$', yscale='log')
    
axd['c'].legend(loc='upper right')
axd['c'].set_xlim(dt.datetime(2016,5,1), dt.datetime(2025,1,1))

#   Plot the actual shape of this boundary
p_dyn_s_10, p_dyn_s_50, p_dyn_s_90 = np.percentile(p_dyn_s.flatten(), [25, 50, 75])

t_ = np.array([np.linspace(0, 0.75 * np.pi, 1000)]*50).T
# p_ = np.zeros((1000, 50))
p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_10
# r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
# rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)

# a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
#                         color='C2', lw = 1, alpha=1/5)
# a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
#                         color='C2', lw = 1, alpha=1)

p_dyn_ = np.zeros((1000, 50)) + p_dyn_s_50
p_ = np.zeros((1000, 50)) + 1*np.pi/2.
r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
# a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
#                         color='C4', lw = 1, alpha=1/5)
a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
                        color='C2', lw = 1, alpha=1,
                        label = 'Mean Boundary Location \n@ Median Pressure')

r_upper_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_)) + sigma_s
rpl_upper_ = BM.convert_SphericalSolarToCylindricalSolar(r_upper_, t_, p_)
r_lower_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_)) - sigma_s
rpl_lower_ = BM.convert_SphericalSolarToCylindricalSolar(r_lower_, t_, p_)

# axd['a'].fill(np.append(np.mean(rpl_upper_[2].T, 0), np.mean(rpl_lower_[2], 0)[::-1]), 
#               np.append(np.mean(rpl_upper_[0].T, 0), np.mean(rpl_lower_[0], 0)[::-1]),
#               color = 'C2', alpha = 0.4, edgecolor = None,
#               label = r'Mean Boundary Location $\pm1\sigma$' + '\n@ Median Pressure')

a_nor_m = axd['a'].plot(np.mean(rpl_upper_[2].T, 0), np.mean(rpl_upper_[0].T, 0),
                        color='C2', lw = 1, alpha=1, ls=':',
                        label = r'Mean Boundary Location $+1\sigma$')

a_nor_m = axd['a'].plot(np.mean(rpl_lower_[2].T, 0), np.mean(rpl_lower_[0].T, 0),
                        color='C2', lw = 1, alpha=1, ls=':', 
                        label = r'Mean Boundary Location $-1\sigma$')
axd['a'].legend()

# p_ = np.zeros((1000, 50)) + p_dyn_s_90
# r_ = model_dict['model']((r0_s, r1_s, a0_s, a1_s), (t_, p_, p_dyn_))
# rpl_ = BM.convert_SphericalSolarToCylindricalSolar(r_, t_, p_)
# a_north = axd['a'].plot(rpl_[2].T, rpl_[0].T,
#                         color='C5', lw = 1, alpha=1/5)
# a_nor_m = axd['a'].plot(np.mean(rpl_[2].T, 0), np.mean(rpl_[0].T, 0),
#                         color='C5', lw = 1, alpha=1)


axd['a'].set(xlim=(150,-400),
             # ylim=(000, 200),
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

import matplotlib.patheffects as pe
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,0]/(n_balanced*50)*100),
                  (1/4, 3/4), ha='center', va='center', color='white',
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[0,1]/(n_balanced*50)*100),
                  (3/4, 3/4), ha='center', va='center', color='white',
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,0]/(n_balanced*50)*100),
                  (1/4, 1/4), ha='center', va='center', color='white',
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])
axd['b'].annotate(r"$\mu$ = {:.1f}%".format(cm[1,1]/(n_balanced*50)*100),
                  (3/4, 1/4), ha='center', va='center', color='white',
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])

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

