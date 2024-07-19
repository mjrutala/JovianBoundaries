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

from tqdm.auto import tqdm
import matplotlib.pyplot as plt


import JunoPreprocessingRoutines as JPR
import BoundaryModels as BM
import JoyBoundaryCoords as JBC


def _ready_DataFrames():
    starttime = datetime.datetime(2016, 5, 1)
    stoptime = datetime.datetime(2017, 2, 1)
    resolution = 10  #  'minutes'
    
    bs_crossings = JPR.make_CombinedCrossingsList(boundary='bs', which='Louis')
    mp_crossings = JPR.make_CombinedCrossingsList(boundary='mp', which='Louis')
    location_df = JPR.convert_DirectionToLocation(starttime, stoptime, bs_crossings, mp_crossings, resolution=resolution)
    
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
    
    location_df['inside_bs'] = location_df['in_msh'] + location_df['in_msp']
    location_df['inside_mp'] = location_df['in_msp']
    
    return location_df, coordinate_df


def find_ModelOrbitIntersections(coordinate_df, boundary = 'bs', 
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
        
        _inside_boundary = []
        for _ in tqdm(range(int(n)), disable=disable_tqdm):

            perturbed_model_r = rng.normal(model_r, sigma)
                
            _inside_boundary.append((perturbed_model_r >= coordinate_df['r'].to_numpy()).astype(int))
        
        inside_boundary = np.mean(_inside_boundary, axis=0)
        #breakpoint()
        
    else:
        inside_boundary = (model_r >= coordinate_df['r'].to_numpy()).astype(int)
    
    location_df = pd.DataFrame(index = coordinate_df.index)
    location_df['inside_' + boundary] = inside_boundary

    return location_df
    
def _compare_(location_df, model_location_df, boundary = 'bs'):
    import matplotlib.cbook as cbook
    import matplotlib.dates as mdates
    
    column_name = 'inside_' + boundary.lower()
    
    fig, ax = plt.subplots(figsize = (6,4))
    
    ax.plot(location_df[column_name], color='black', 
            label = 'Juno')
    
    ax.plot(model_location_df[column_name], linestyle='-', 
            label = 'Forward Model')
    
    ax.legend(loc = 'lower center', bbox_to_anchor = [0.5, 1], ncols=2)
    
    ax.set(xlabel = 'Date [Year-DoY]',
           ylabel = 'Fraction of Time within Bow Shock \n (Instantaneous)')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval = 50))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval = 5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%j'))
    
    
  
    
def boundary_model_init(model_name):
    #   Select boundary model
    bm = {'Shuelike': {'model': BM.Shuelike, 
                       'param_dict': {'r0': [30, 40, 50, 60, 70, 80],
                                      'r1': [-0.1, -0.2, -0.3, -0.4],
                                      'a0': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                      'a1': [0.01, 0.1, 1.0, 10.0]}
                       },
          }
    return bm[model_name]
          
def loop_over_grid(model_name = 'Shuelike'):
    
    location_df, coordinate_df = _ready_DataFrames()
    
    
    #   
    model_params = boundary_model_init(model_name)
    
    #   Make grids of every permutation of parameters
    all_params = []
    for key, val in model_params['param_dict'].items():
        all_params.append(val)
    
    
    param_ngrid = np.meshgrid(*all_params)
    n_param = len(param_ngrid)
    n_grid = param_ngrid[0].size
    
    r2_list = []
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
                                                sigma=10, n=1e2,
                                                disable_tqdm = True)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(location_df['inside_bs'], model_df['inside_bs'])
        r2_list.append(r_value**2)
        
        rmsd_list.append(np.sqrt(np.mean((location_df['inside_bs'] - model_df['inside_bs'])**2)))
        mae_list.append(np.mean(np.abs(location_df['inside_bs'] - model_df['inside_bs'])))
        
        #   Maybe measure the fit where location = 0 and location = 1 separately?
        indx_eq1 = location_df['inside_bs'] == 1
        indx_eq0 = location_df['inside_bs'] == 0
        
        rmsd_0_list.append(np.sqrt(np.mean((location_df.loc[indx_eq0, 'inside_bs'] - model_df.loc[indx_eq0, 'inside_bs'])**2)))
        rmsd_1_list.append(np.sqrt(np.mean((location_df.loc[indx_eq1, 'inside_bs'] - model_df.loc[indx_eq1, 'inside_bs'])**2)))
        
        mae_0_list.append(np.mean(np.abs(location_df.loc[indx_eq0, 'inside_bs'] - model_df.loc[indx_eq0, 'inside_bs'])))
        mae_1_list.append(np.mean(np.abs(location_df.loc[indx_eq1, 'inside_bs'] - model_df.loc[indx_eq1, 'inside_bs'])))
        
    breakpoint()
        
        
    
    