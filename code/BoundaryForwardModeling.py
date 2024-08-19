#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:45:53 2024

@author: mrutala
"""

import datetime
import numpy as np
import pandas as pd
import scipy

import matplotlib as mpl
import datetime as dt

from tqdm.auto import tqdm
import matplotlib.pyplot as plt


import JunoPreprocessingRoutines as JPR
import BoundaryModels as BM
import JoyBoundaryCoords as JBC


def _ready_DataFrames(starttime, stoptime, resolution=1):
    #starttime = datetime.datetime(2016, 8, 1)
    #stoptime =  datetime.datetime(2016, 11, 1) # datetime.datetime(2024, 2, 28)
    #resolution = 10  #  'minutes'
    
    #bs_crossings = JPR.make_CombinedCrossingsList(boundary='bs', which=['Louis', 'Kurth'])
    bs_crossings = JPR.make_CombinedCrossingsList(boundary='bs', which=['Louis', 'Kurth'])#, 'Ebert'])
    mp_crossings = JPR.make_CombinedCrossingsList(boundary='mp', which='Louis')
    location_df = JPR.convert_DirectionToLocation(starttime, stoptime, bs_crossings, mp_crossings, resolution=resolution)
    location_df = location_df.dropna(subset = ['in_sw', 'in_msh', 'in_msp'], axis = 'index', how = 'any')
    
    #   Drop location info to get coordinate_df
    coordinate_df = location_df.drop(['in_sw', 'in_msh', 'in_msp', 'location'], axis='columns')
    
    #   Add additional coordinates to coordinate_df
    rpl = BM.convert_CartesianToCylindricalSolar(*coordinate_df.loc[:, ['x', 'y', 'z']].to_numpy().T)
    coordinate_df.loc[:, ['rho', 'phi', 'ell']] = rpl.T
    
    rtp = BM.convert_CartesianToSphericalSolar(*coordinate_df.loc[:, ['x', 'y', 'z']].to_numpy().T)
    coordinate_df.loc[:, ['r', 't', 'p']] = rtp.T
    
    # #   Add MMESH pressure to coordinate_df
    #coordinate_df['p_dyn'] = 0.03
    import sys
    sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
    import MMESH_reader
    sw_mme_filepath = '/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv'
    
    # =============================================================================
    #     !!!! Here we'll need to repeatedly resample pressure w/ uncertainties
    # =============================================================================
    #   Load the solar wind data and select the ensesmble
    sw_models = MMESH_reader.fromFile(sw_mme_filepath)
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
    #   Upsample the pressure using linear interpolation
    p_dyn = sw_mme.loc[:, 'p_dyn'].resample('{}min'.format(resolution)).interpolate()
    
    coordinate_df['p_dyn'] = p_dyn.loc[coordinate_df.index]
    
    location_df['within_bs'] = location_df['in_msh'] + location_df['in_msp']
    location_df['within_mp'] = location_df['in_msp']
    
    return location_df, coordinate_df

def find_ModelOrbitIntersections(coordinate_df, boundary = 'bs',
                                 model = None, params = None, as_numpy = False):
    #   Get the string names of variables expected by the model
    input_var_names, output_var_name = model(variables = True)
    
    #   In spherical coordinates: get boundary 'r' position from 't, p'    
    if output_var_name == 'r':
        model_r = model(parameters = params, coordinates = coordinate_df.loc[:, input_var_names].to_numpy().T)
    
    else:
        breakpoint()
        
    within_boundary = (model_r >= coordinate_df['r'].to_numpy()).astype(int)
    
    location_df = pd.DataFrame(index = coordinate_df.index)
    location_df['within_' + boundary] = within_boundary

    return location_df
    

def _find_ModelOrbitIntersections(coordinate_df, boundary = 'bs', 
                                 model = None, params = None, 
                                 sigma=10, n=1e3,
                                 disable_tqdm = False):
    #   Specify a 4D boundary model, the type of boundary, the pressure time series, and a (constant***) internal sigma
    #   Return spacecraft orbital locations
    
    #   Get the string names of variables expected by the model
    input_var_names, output_var_name = model(variables = True)
    
    
    #   In spherical coordinates: get boundary 'r' position from 't, p'    
    if output_var_name == 'r':
        model_r = model(parameters = params, coordinates = coordinate_df.loc[:, input_var_names].to_numpy().T)
    
    else:
        breakpoint()
        
    #   Get random variations in boundary r with sigma=sigma, n times
    if (sigma > 0) & (n > 0):
        rng = np.random.default_rng()
        
        _within_boundary = []
        for _ in tqdm(range(int(n)), disable=disable_tqdm):

            perturbed_model_r = rng.normal(model_r, sigma)
                
            _within_boundary.append((perturbed_model_r >= coordinate_df['r'].to_numpy()).astype(int))
        
        within_boundary = np.mean(_within_boundary, axis=0)
        #breakpoint()
        
    else:
        within_boundary = (model_r >= coordinate_df['r'].to_numpy()).astype(int)
    
    location_df = pd.DataFrame(index = coordinate_df.index)
    location_df['within_' + boundary] = within_boundary

    return location_df
    
def _compare_(location_df, model_location_df=None, boundary='bs'):
    import matplotlib.cbook as cbook
    import matplotlib.dates as mdates
    
    column_name = 'within_' + boundary.lower()
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    ax.plot(location_df[column_name], color='black', 
            label = 'Juno')
    
    if model_location_df is not None:
        ax.plot(model_location_df[column_name], linestyle='-', linewidth = 1.0,
                label = 'Forward Model',)
    
    ax.legend(loc = 'lower center', bbox_to_anchor = [0.5, 1], ncols=2)
    
    ax.set(xlabel = 'Date [Year-DoY]',
           ylabel = 'Fraction of Time within Bow Shock \n (Instantaneous)')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval = 50))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval = 5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%j'))
    
    plt.show()
  

def loop_over_grid(starttime, stoptime, resolution = 1, model_name = 'Shuelike'):
    
    location_df, coordinate_df = _ready_DataFrames(starttime, stoptime, resolution=resolution)
    
    
    #   
    model_params = boundary_model_init(model_name)
    
    #   Make grids of every permutation of parameters
    all_params = []
    for key, val in model_params['param_dict'].items():
        all_params.append(val)
    
    
    param_ngrid = np.meshgrid(*all_params)
    param_nlist = []
    for grid in param_ngrid:
        param_nlist.append(grid.flatten())
        
    n_param = len(param_ngrid)
    n_grid = param_ngrid[0].size
    
    # r2_list = []
    rsdl_list = []
    rmsd_list = []
    mae_list = []
    rmsd_0_list, rmsd_1_list = [], []
    mae_0_list, mae_1_list = [], []
    for i in tqdm(range(n_grid)):
        params = []
        for permutation in param_ngrid:
            params.append(permutation.flatten()[i])
            
            
        model_df = find_ModelOrbitIntersections(coordinate_df, boundary = 'bs', 
                                                model = model_params['model'], 
                                                params = params, 
                                                sigma=10, n=1e3,
                                                disable_tqdm = True)
        
        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(location_df['within_bs'], model_df['within_bs'])
        # r2_list.append(r_value**2)
        
        rsdl_list.append(np.mean(location_df['within_bs'] - model_df['within_bs']))
        
        rmsd_list.append(np.sqrt(np.mean((location_df['within_bs'] - model_df['within_bs'])**2)))
        mae_list.append(np.mean(np.abs(location_df['within_bs'] - model_df['within_bs'])))
        
        #   Maybe measure the fit where location = 0 and location = 1 separately?
        indx_eq1 = location_df['within_bs'] == 1
        indx_eq0 = location_df['within_bs'] == 0
        
        rmsd_0_list.append(np.sqrt(np.mean((location_df.loc[indx_eq0, 'within_bs'] - model_df.loc[indx_eq0, 'within_bs'])**2)))
        rmsd_1_list.append(np.sqrt(np.mean((location_df.loc[indx_eq1, 'within_bs'] - model_df.loc[indx_eq1, 'within_bs'])**2)))
        
        mae_0_list.append(np.mean(np.abs(location_df.loc[indx_eq0, 'within_bs'] - model_df.loc[indx_eq0, 'within_bs'])))
        mae_1_list.append(np.mean(np.abs(location_df.loc[indx_eq1, 'within_bs'] - model_df.loc[indx_eq1, 'within_bs'])))
        
    
    combined_mae = mae_list # 0.5 * np.array(mae_0_list) + 0.5 * np.array(mae_1_list)
    min_mae_indx = np.where(combined_mae <= np.percentile(combined_mae, 10))[0]
    
    if len(min_mae_indx) == 0:
        breakpoint()
    
    return np.array(param_nlist)[:, min_mae_indx]
    
def fit_Subsets(starttime = dt.datetime(2016, 5, 1),
                stoptime = dt.datetime(2018, 5, 1),
                resolution = 10,
                subset_length = 360, subset_offset = None):
    
    #   Break [starttime, stoptime) into subset_length chunks, spaced by subset_offset
    #   If subset_offset = None, then assumer non-overlapping chunks
    #   (e.g. subset_length = subset_offset)
    if subset_offset == None:
        subset_offset = subset_length
    
    subset_starts = np.arange(starttime, stoptime-dt.timedelta(days=subset_length), dt.timedelta(days=subset_offset)).astype(dt.datetime)
    subset_stops = np.arange(starttime+dt.timedelta(days=subset_length), stoptime, dt.timedelta(days=subset_offset)).astype(dt.datetime)
    
    best_fits = []
    mean_x, mean_y, mean_z = [], [], []
    for subset_start, subset_stop in zip(subset_starts, subset_stops):
        
        location_df, coordinate_df = _ready_DataFrames(subset_start,
                                                        subset_stop,
                                                        resolution=resolution)
        mean_x.append(np.mean(coordinate_df['x']))
        mean_y.append(np.mean(coordinate_df['y']))
        mean_z.append(np.mean(coordinate_df['z']))
        
        res = loop_over_grid(subset_start, subset_stop, resolution = resolution, model_name = 'Shuelike')
        
        
        best_fits.append(res)

    
    fig, axs = plt.subplots(nrows=2)
    for x, best_fit in zip(mean_x, best_fits):
        n_vals = len(best_fit[0])
        axs[0].scatter([x] * n_vals, best_fit[0])
        axs[1].scatter([x] * n_vals, best_fit[1])
    
    plt.show()
    
    fig, ax = plt.subplots()
    norm = mpl.colors.Normalize(vmin=np.min(mean_y), vmax=np.max(mean_y))
    cmap = mpl.colormaps['Spectral']


    for best_fit, coord in zip(best_fits, mean_y):
        
        rgba = cmap(norm(coord))

        ax.plot(best_fit[0], best_fit[1], color=rgba)
        
    plt.show()
        
    breakpoint()
    
    
    
    
    
        
def _steps_illustration():
    
    #   Load the real crossing locations & coordinates w/ pressures
    location_df, coordinate_df = _ready_DataFrames()
    
    #   For the example, copy the coordinate_df and overrite the pressures
    coordinate_constpdyn_df = coordinate_df.copy()
    coordinate_constpdyn_df['p_dyn'] = 1    #   1 is not physical, but we're aiming for pseudo-unitless
    
    #   Plot the location_df info alone
    _compare_(location_df, boundary='bs')
    
    #   Now fit a simple, pressureless model with no internal variability
    model_info = boundary_model_init('Shuelike')
    
    #loop_over_grid(model_name = 'Shuelike')
    
    model_df = find_ModelOrbitIntersections(coordinate_constpdyn_df, boundary = 'bs', 
                                            model = model_info['model'], params = [30, -0.25, 0.78, 0.37], 
                                            sigma=0, n=1e3,
                                            disable_tqdm = False)
    
    _compare_(location_df, model_df)
    
    #   Next fit a simple, pressureless model with internal variability
    model_df = find_ModelOrbitIntersections(coordinate_constpdyn_df, boundary = 'bs',
                                            model = model_info['model'], params = [30, -0.25, 0.78, 0.37], 
                                            sigma = 8, n = 1e4)
    
    _compare_(location_df, model_df)
    
    #   Now add the modeled pressures back in
    model_df = find_ModelOrbitIntersections(coordinate_df, boundary = 'bs', 
                                            model = model_info['model'], params = [30, -0.25, 0.78, 0.37], 
                                            sigma = 0, n = 1e4)
    _compare_(location_df, model_df)
    
    #   Visualize
    t = np.linspace(0, 150, 1000) * np.pi/180
    p = np.zeros(1000)
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    p_dyn = np.zeros(1000) + 1
    r = model_info['model'](parameters = [30, -0.25, 0.78, 0.37], coordinates = np.array([t, p, p_dyn]))
    ax.plot(r * np.cos(t), r * np.sin(t), label = r'$p_{dyn}$ = 1 nPa', linestyle=':')
    
    p_dyn = np.zeros(1000) + 0.13
    r = model_info['model'](parameters = [30, -0.25, 0.78, 0.37], coordinates = np.array([t, p, p_dyn]))
    ax.plot(r * np.cos(t), r * np.sin(t), label = r'$p_{dyn}$ = 0.13 nPa')
    
    p_dyn = np.zeros(1000) + 0.08
    r = model_info['model'](parameters = [30, -0.25, 0.78, 0.37], coordinates = np.array([t, p, p_dyn]))
    ax.plot(r * np.cos(t), r * np.sin(t), label = r'$p_{dyn}$ = 0.08 nPa')
    
    p_dyn = np.zeros(1000) + 0.03
    r = model_info['model'](parameters = [30, -0.25, 0.78, 0.37], coordinates = np.array([t, p, p_dyn]))
    ax.plot(r * np.cos(t), r * np.sin(t), label = r'$p_{dyn}$ = 0.03 nPa')
    
    df = coordinate_df.loc[location_df.query('inside_bs == 0').index, :]
    ax.scatter(df['r']*np.cos(df['t']), df['r']*np.sin(df['t']), s=1, marker='x', color='black', label='Juno outside BS')
    
    ax.legend()
    ax.set(xlim = [-200, 200], ylim = [0, 200])
    ax.set(aspect=1)
    
    plt.show()
    
    