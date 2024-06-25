#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:33:41 2024

@author: mrutala
"""
from pathlib import Path
import pandas as pd
import datetime as dt

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

import BoundaryModels as BM

def get_paths():
    paths_dict = {'data': Path('/Users/mrutala/projects/JupiterBoundaries/data/'),
                  'SPICE': Path('/Users/mrutala/SPICE/')}
    paths_dict = paths_dict | {'JunoMetakernel': paths_dict['SPICE']/'juno/metakernel_juno.txt',
                               'PlanetaryMetakernel': paths_dict['SPICE']/'generic/metakernel_planetary.txt', 
                               'Louis2023_magnetopause': paths_dict['data'] / 'Louis2023/boundary_crossings_caracteristics_MP.csv', 
                               'Louis2023_bowshock': paths_dict['data'] / 'Louis2023/boundary_crossings_caracteristics_BS.csv',
                               'Kurth_bowshock': paths_dict['data'] / 'Kurth_Waves/Kurth_BowShocks_formatted.csv',
                               'Ebert_magnetopause': paths_dict['data'] / 'Ebert_JADE/Magnetopause_Crossings_Ebert2024_v5.csv',
                               'Ebert_bowshock': paths_dict['data'] / 'Ebert_JADE/Bowshock_Crossings_Ebert2024.csv'}
    
    return paths_dict
    
def make_CombinedCrossingsList(boundary = 'BS'):
    import spiceypy as spice
    
    match boundary.lower():
        case ('bs' | 'bowshock' | 'bow shock'):
            crossing_data_Louis2023 = read_Louis2023_CrossingList(bs=True)
            
            crossing_data_Kurth = read_Kurth_CrossingList(bs=True)
            #   Dropping items which didn't have a clearly listed crossing direction
            crossing_data_Kurth = crossing_data_Kurth[crossing_data_Kurth['direction'].notna()]
            
            crossing_data_Ebert = read_Ebert_CrossingList(bs=True)
            
            crossings_df = pd.concat([crossing_data_Louis2023, crossing_data_Kurth, crossing_data_Ebert], axis=0, join="outer")
        case ('mp' | 'magnetopause'):
            crossing_data_Louis2023 = read_Louis2023_CrossingList(mp=True)
            
            crossing_data_Ebert = read_Ebert_CrossingList(mp=True)
            
            crossings_df = pd.concat([crossing_data_Louis2023, crossing_data_Ebert], axis=0, join="outer")
        case _:
            crossings_df = None
    
    crossings_df = crossings_df.drop(['date', 'time', 'DoY'], axis='columns')
    crossings_df = crossings_df.drop(['x_IAU', 'y_IAU', 'z_IAU', 'r_IAU', 'theta_IAU', 'phi_IAU'], axis='columns')
     
    #   Load SPICE kernels for positions of Juno
    spice.furnsh(get_paths()['PlanetaryMetakernel'].as_posix())
    spice.furnsh(get_paths()['JunoMetakernel'].as_posix())
    
    R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
    
    #   Get the positions at all time stamps
    df_ets = spice.datetime2et(crossings_df.index)
    pos, lt = spice.spkpos('Juno', df_ets , 'Juno_JSS', 'None', 'Jupiter')
    
    spice.kclear()
    
    crossings_df[['x_JSS', 'y_JSS', 'z_JSS']] = pos / R_J
        
    return crossings_df

def make_HourlyCrossingList(df):
    """
    

    Parameters
    ----------
    df : pandas DataFrame
        Expected to have datetime index and two columns: direction and notes
         - direction defines whether the corssing is into or out of the boundary
         - notes describes if conditions of the crossing-- e.g., if the crossing
        was poorly defined
    resolution : string, optional
        A pandas text string defining the desired temporal resolution. The default is '60Min'.

    Returns
    -------
    None.

    """
    import spiceypy as spice
    
    R_J = 71492. # km to Jupiter radius
    
    df['notes'] = df['notes'].fillna('null')
    
    #   Get the nearest hour to hourly precision
    df['hourly_datetimes'] = get_HourlyFromDatetimes(df.index)
    
    #   Check if this hour is unique; if not, combine with others
    df = df.groupby(['hourly_datetimes']).agg(direction=('direction', lambda x: x.to_numpy()), 
                                              notes=('notes', lambda x: x.to_numpy()), 
                                              origin=('origin', lambda x: x.to_numpy()))
    
    #   Load SPICE kernels for positions of Juno
    spice.furnsh(get_paths()['PlanetaryMetakernel'].as_posix())
    spice.furnsh(get_paths()['JunoMetakernel'].as_posix())
    
    #   Get the positions at the hour, then half an hour before and after
    df_ets = spice.datetime2et(df.index)
    pos_before, lt_before = spice.spkpos('Juno', df_ets - (30*60), 'Juno_JSS', 'None', 'Jupiter')
    pos_middle, lt_middle = spice.spkpos('Juno', df_ets , 'Juno_JSS', 'None', 'Jupiter')
    pos_after, lt_after = spice.spkpos('Juno', df_ets  + (30*60), 'Juno_JSS', 'None', 'Jupiter')
    
    spice.kclear()
    
    #   Get the differences
    left_half = abs(pos_middle - pos_before)
    right_half = abs(pos_after - pos_middle)
    
    #   Take the larger of the differences as the 3 sigma uncertainty
    uncertainties = left_half
    uncertainties[left_half < right_half] = right_half[left_half < right_half]
    
    
    df[['x_JSS', 'y_JSS', 'z_JSS']] = pos_middle/R_J
    df[['x_unc_JSS', 'y_unc_JSS', 'z_unc_JSS']] = uncertainties/3/R_J
    
    return df

def get_HourlyFromDatetimes(ts):
    #   Get the nearest hour to hourly precision
    res = [t.replace(minute=0, second=0, microsecond=0) + 
               dt.timedelta(hours=t.minute//30) for t in ts]
    return res

def read_Louis2023_CrossingList(mp=False, bs=True):
    if mp == True:
        crossing_list = get_paths()['Louis2023_magnetopause']
    else:
        crossing_list = get_paths()['Louis2023_bowshock']
        
    crossing_list_names = ['#', 'DoY', 'date', 'time', 'boundary',  'direction', 'notes', 
                           'x_JSS', 'y_JSS', 'z_JSS', 
                           'x_IAU', 'y_IAU', 'z_IAU', 'r_IAU', 'theta_IAU', 'phi_IAU', 
                           'p_dyn', 'r_mp', 'r_bs']
    crossings = pd.read_csv(crossing_list, sep=';', names=crossing_list_names, header=0)
    crossings.index = [dt.datetime.strptime(row['date']+'T'+row['time'], '%Y/%m/%dT%H:%M') for index, row in crossings.iterrows()]
    crossings = crossings.sort_index()
    crossings['origin'] = 'Louis+ (2023)'
    return crossings

def read_Kurth_CrossingList(bs=True):
    if bs == True:
        crossing_list = get_paths()['Kurth_bowshock']
    else:
        print("No magnetopause crossings available from this function!")
        return
    
    crossing_list_names = ['apojoves', 'date', 'direction', 'notes']
    crossings = pd.read_csv(crossing_list, sep = ',', names = crossing_list_names, header = 0)
    crossings.index = [dt.datetime.strptime(row['date'], '%Y-%jT%H:%M') for index, row in crossings.iterrows()]
    crossings = crossings.sort_index()
    crossings['origin'] = 'Kurth, p.c.'
    return crossings

def read_Ebert_CrossingList(mp=False, bs=True):
    if mp == True:
        crossing_list = get_paths()['Ebert_magnetopause']
        crossing_list_names = ['#', 'date', 'time', 
                               'r', 'x_JSO', 'y_JSO', 'z_JSO', 
                               'lat', 'mlat', 'lon', 'LT', 
                               'notes']
    else:
        crossing_list = get_paths()['Ebert_bowshock']
        crossing_list_names = ['#', 'date', 'time',
                               'r', 'lat', 'mlat', 'MLT']
    #   Read the correct list, and drop rows without events
    crossings = pd.read_csv(crossing_list, 
                            sep=',', names = crossing_list_names, header=0,
                            skipinitialspace=True)
    crossings = crossings.dropna(axis = 'index', how = 'all', subset = ['date', 'time'])
    
    #   Set the index to the datetime, rather than an int
    crossings.index = [dt.datetime.strptime(row['date']+'T'+row['time'], '%Y-%jT%H:%M') for _, row in crossings.iterrows()]
    crossings = crossings.sort_index()
    
    #   Add the origin of these crossings to the df
    crossings['origin'] = 'Ebert (+ Montgomery), p.c.'
    
    return crossings

def plot_CrossingsAndTrajectories():
    import spiceypy as spice
    import numpy as np
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    mp_crossing_data = read_Louis2023_CrossingList(mp=True)
    mp_crossing_data = make_HourlyCrossingList(mp_crossing_data)
    
    #   Bow Shock Crossings
    bs_crossing_data_Louis2023 = read_Louis2023_CrossingList(bs=True)
    
    bs_crossing_data_Kurth = read_Kurth_CrossingList(bs=True)
    bs_crossing_data_Kurth = bs_crossing_data_Kurth[bs_crossing_data_Kurth['direction'].notna()]
    
    bs_crossing_data = pd.concat([bs_crossing_data_Louis2023, bs_crossing_data_Kurth], axis=0, join="outer")
    bs_crossing_data = make_HourlyCrossingList(bs_crossing_data)
    
    #   Load SPICE kernels for positions of Juno
    spice.furnsh(get_paths()['PlanetaryMetakernel'].as_posix())
    spice.furnsh(get_paths()['JunoMetakernel'].as_posix())
    
    R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
    
    #   Get all hourly trajectory info
    earliest_time = min(np.append(bs_crossing_data.index, mp_crossing_data.index))
    latest_time = max(np.append(bs_crossing_data.index, mp_crossing_data.index))
    
    datetimes = np.arange(pd.Timestamp(earliest_time).replace(minute=0, second=0, nanosecond=0),
                          pd.Timestamp(latest_time).replace(minute=0, second=0, nanosecond=0) + dt.timedelta(hours=1),
                          dt.timedelta(hours=1)).astype(dt.datetime)
    
    ets = spice.datetime2et(datetimes)
    
    pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
    pos = pos.T / R_J
    pos_df = pd.DataFrame({'x': pos[0], 'y': pos[1], 'z':pos[2]}, 
                          index=datetimes)
    
    spice.kclear()
    
    sw_models = MMESH_reader.fromFile('/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv')
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
    for crossing_type in ['bs', 'mp']:
        if crossing_type == 'bs':
            crossing_data = bs_crossing_data
            marker = 'x'
        if crossing_type == 'mp':
            crossing_data = mp_crossing_data
            marker = 'o'
        
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6,5))
        plt.subplots_adjust(left=0.1, bottom=0.14, right=0.8, top=0.98, 
                            wspace=0.05, hspace=0.05)
        ax2 = fig.add_axes([0.82, 0.14, 0.04, 0.84])
        
        axs[1,0].plot(pos_df['y'], pos_df['z'], 
                      color='black', linewidth=0.5, zorder=1)
        axs[1,0].annotate('Y-Z Plane', (0, 1), (1, -1), ha='left',
                          xycoords='axes fraction', textcoords='offset fontsize')
        
        axs[1,1].plot(pos_df['x'], pos_df['z'],
                      color='black', linewidth=0.5, zorder=1)
        axs[1,1].annotate('X-Z Plane', (0, 1), (1, -1), ha='left',
                          xycoords='axes fraction', textcoords='offset fontsize')
        
        axs[0,0].plot(pos_df['y'], pos_df['x'], 
                      color='black', linewidth=0.5, zorder=1)
        axs[0,0].annotate('Y-X Plane', (0, 1), (1, -1), ha='left',
                          xycoords='axes fraction', textcoords='offset fontsize')
        
        axs[0,1].set_axis_off()
        # axs[0,1].annotate('3D View WIP', (0, 1), (1, -1), ha='left',
        #                   xycoords='axes fraction', textcoords='offset fontsize')
        
        #
        bounds = np.logspace(-2, -1, 11)
        cmap = plt.get_cmap('magma', 11)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        cs = axs[1,0].scatter(*pos_df.loc[crossing_data.index, ['y', 'z']].to_numpy().T,
                              c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
                              cmap = cmap, norm = norm,
                              marker = marker, zorder=2)
        axs[1,1].scatter(*pos_df.loc[crossing_data.index, ['x', 'z']].to_numpy().T,
                              c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
                              cmap = cmap, norm = norm,
                              marker = marker, zorder=2)
        axs[0,0].scatter(*pos_df.loc[crossing_data.index, ['y', 'x']].to_numpy().T,
                              c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
                              cmap = cmap, norm = norm,
                              marker = marker, zorder=2)
        
        axs[1,0].set(xlabel = r'$Y_{JSS}$ (+ duskward) [$R_J$]', xlim = [-120, 80], 
                     ylabel = r'$Z_{JSS}$ (+ north) [$R_J$]', ylim = [-120, 80])
        axs[0,0].set(ylabel = r'$X_{JSS}$ (+ sunward) [$R_J$]', ylim = [40,-160])
        axs[1,1].set(xlabel = r'$X_{JSS}$ (+ sunward) [$R_J$]', xlim = [40,-160])
        
        cb = plt.colorbar(cs, cax=ax2)
        #cb.ax.set_yticklabels([r'$10^{{{:.1f}}}$'.format(np.log10(num)) for num in cb.get_ticks()])
        cb.ax.set_yticklabels([r'{:.1f}'.format(np.log10(num)) for num in bounds])
        ax2.set(ylabel=r'MME Solar Wind Pressure ($p_{dyn}$) [log(nPa)]')
        
        plt.show()
    
    breakpoint()

