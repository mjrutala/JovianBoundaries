#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:48:23 2024

@author: mrutala
"""

import pandas as pd
import spiceypy as spice
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

import sys
import JoyBoundaryCoords as JBC

plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')

R_J = 71492.

def read_Louis2023_CrossingList():
    bowshock_crossing_list = '/Users/mrutala/Data/Published/Louis2023/boundary_crossings_caracteristics_BS.csv'
    bowshock_crossing_list_names = ['#', 'DoY', 'date', 'time', 'boundary',  'direction', 'notes', 
                                    'x_JSS', 'y_JSS', 'z_JSS', 
                                    'x_IAU', 'y_IAU', 'z_IAU', 'r_IAU', 'theta_IAU', 'phi_IAU', 
                                    'p_dyn', 'r_mp', 'r_bs']
    bs_crossings = pd.read_csv(bowshock_crossing_list, sep=';', names=bowshock_crossing_list_names, header=0)
    bs_crossings.index = [dt.datetime.strptime(row['date']+'T'+row['time'], '%Y/%m/%dT%H:%M') for index, row in bs_crossings.iterrows()]
    return(bs_crossings)

def read_MMESH_Outputs():
    MMESH_csv = '/Users/mrutala/projects/MMESH/JupiterMME_withConstituentModels.csv'
    sw_models = pd.read_csv(MMESH_csv, comment='#', header=[0,1], index_col=[0])
    sw_models.index = pd.to_datetime(sw_models.index)
    return(sw_models)

#   The first bowshock encounter in this list is during cruise, which is not currently present in the MMESH outputs
sw_models = read_MMESH_Outputs()
bowshock_encounters = read_Louis2023_CrossingList()

#   Assign perijove labels to each bowshock
#   PJ00 starts at PJ00 and ends at PJ01
perijove_dict = {'PJ00': (dt.datetime(2016, 7, 5, 2,47,32), dt.datetime(2016, 8,27,12,50,44)),
                 'PJ01': (dt.datetime(2016, 8,27,12,50,44), dt.datetime(2016,10,19,18,10,54)),
                 'PJ02': (dt.datetime(2016,10,19,18,10,54), dt.datetime(2016,12,11,17, 3,41)),
                 'PJ03': (dt.datetime(2016,12,11,17, 3,41), dt.datetime(2017, 2, 2,12,57, 9)),
                 'PJ04': (dt.datetime(2017, 2, 2,12,57, 9), dt.datetime(2017, 3,27, 8,51,52))}
bowshock_encounters['perijove'] = 'none'
for perijove_label, dates in perijove_dict.items():
    bowshock_encounters.loc[dates[0]:dates[1], 'perijove'] = perijove_label

#   Loop over perijoves
for perijove in set(bowshock_encounters['perijove'].values):
    
    bs_subset = bowshock_encounters.loc[bowshock_encounters['perijove'] == perijove]
    direction_colors = np.array(['xkcd:aqua' if row['direction'] == 'out' else 'xkcd:salmon' 
                                 for index, row in bs_subset.iterrows()])
    direction_markers = np.array(['x' if row['direction'] == 'out' else 'o'
                                  for index, row in bs_subset.iterrows()])
    
    #   Quick function to round datetimes to the nearest hour
    def nearest_hour(datetimes):
        result = [t.replace(minute=0, second=0, microsecond=0) + 
                  dt.timedelta(hours=t.minute//30) for t in datetimes]
        return np.array(result)
    
    padding = dt.timedelta(hours=24*4)
    datetime_range = nearest_hour(bs_subset.index[[0,-1]].to_pydatetime())
    datetime_range = datetime_range + np.array([-padding, padding])
    
    extended_datetime_range = datetime_range + np.array([-4*padding, 4*padding])
    
    datetimes = np.arange(*datetime_range, dt.timedelta(hours=1)).astype(dt.datetime)
    extended_datetimes = np.arange(*extended_datetime_range, dt.timedelta(hours=1)).astype(dt.datetime)
    
    fig, axd = plt.subplot_mosaic(
        """
        abc
        ddd
        eee
        """,
        layout="constrained", figsize=(6,6))
    
    #   Juno global positiong from SPICE
    with spice.KernelPool('/Users/mrutala/SPICE/juno/metakernel_juno.txt'):
        ets = spice.datetime2et(extended_datetimes)
        extended_juno_pos, lts = spice.spkpos('Juno', ets, 
                                     'Juno_JSS', 'LT', 'Jupiter')
        extended_juno_pos = extended_juno_pos.T / R_J
        juno_pos = extended_juno_pos[:, (extended_datetimes >= datetime_range[0]) & (extended_datetimes <= datetime_range[1])]
        
        crossing_pos, lts = spice.spkpos('Juno', spice.datetime2et(bs_subset.index.to_pydatetime()), 
                                         'Juno_JSS', 'LT', 'Jupiter')
        crossing_pos = crossing_pos.T / R_J
    
    #   Start plotting
    #   X-Y plane 
    for marker in set(direction_markers):
        mask = np.array(direction_markers) == marker
        axd['a'].scatter(crossing_pos[0,mask], crossing_pos[1,mask],
                         marker=marker, c=direction_colors[mask], s=32, linewidth=1)
    axd['a'].scatter(0,0,marker='*', color='xkcd:peach')
    
    axd['a'].plot(extended_juno_pos[0], extended_juno_pos[1], 
                  color='black', linewidth=0.5)
    axd['a'].plot(juno_pos[0], juno_pos[1], 
                  color='xkcd:blue', linewidth=1)
    
    x_bs = np.arange(-150,150.1,.1)
    y_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=x_bs, z=np.mean(bs_subset['z_JSS']))
    axd['a'].plot(x_bs, y_bs[0], color='gray', linewidth=1)
    axd['a'].plot(x_bs, y_bs[1], color='gray', linewidth=1)
    axd['a'].annotate('z = {0:.1f}'.format(np.mean(bs_subset['z_JSS'])), 
                      (1,1), (-1,-1), 
                      xycoords='axes fraction', textcoords='offset fontsize',
                      ha='right', va='center')
    
    axd['a'].set_aspect(1)
    axd['a'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Y_{JSS}$')
    
    #   X-Z plane
    axd['b'].scatter(crossing_pos[0],crossing_pos[2],
                     marker='x', c=direction_colors, s=32, linewidth=0.5)
    axd['b'].scatter(0,0,marker='*', color='xkcd:peach')
    xlim, ylim = axd['b'].get_xlim(), axd['b'].get_ylim()
    
    axd['b'].plot(extended_juno_pos[0], extended_juno_pos[2], 
                  color='black', linewidth=0.5)
    axd['b'].plot(juno_pos[0], juno_pos[2], 
                 color='xkcd:blue', linewidth=1)
    
    x_bs = np.arange(-150,150.1,.1)
    z_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=x_bs, y=np.mean(bs_subset['y_JSS']))
    axd['b'].plot(x_bs, z_bs[0], color='gray', linewidth=1)
    axd['b'].plot(x_bs, z_bs[1], color='gray', linewidth=1)
    axd['b'].annotate('y = {0:.1f}'.format(np.mean(bs_subset['y_JSS'])), 
                      (1,1), (-1,-1), 
                      xycoords='axes fraction', textcoords='offset fontsize',
                      ha='right', va='center')
    
    #axd['b'].set(xlim=xlim, ylim=ylim)
    axd['b'].set_aspect(1)
    axd['b'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Z_{JSS}$')
    
    #   Y-Z Plane
    axd['c'].scatter(crossing_pos[1],crossing_pos[2],
                     marker='x', c=direction_colors, s=32, linewidth=0.5)
    axd['c'].scatter(0,0,marker='*', color='xkcd:peach')
    
    axd['c'].plot(extended_juno_pos[1], extended_juno_pos[2], 
                  color='black', linewidth=1)
    axd['c'].plot(juno_pos[1], juno_pos[2], 
                  color='xkcd:blue', linewidth=2)
    
    y_bs = np.arange(-150,150.1,.1)
    z_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=np.mean(bs_subset['x_JSS']), y=y_bs)
    axd['c'].plot(y_bs, z_bs[0], color='gray', linewidth=1)
    axd['c'].plot(y_bs, z_bs[1], color='gray', linewidth=1)
    axd['c'].annotate('x = {0:.1f}'.format(np.mean(bs_subset['x_JSS'])), 
                      (1,1), (-1,-1), 
                      xycoords='axes fraction', textcoords='offset fontsize',
                      ha='right', va='center')
    
    axd['c'].set_aspect(1)
    axd['c'].set(xlabel=r'$Y_{JSS}$', ylabel=r'$Z_{JSS}$')
    
    #   MMESH
    colors = ['xkcd:baby blue', 'xkcd:kelly green', 'xkcd:lavender', 'xkcd:navy']
    sw_model_names = list(set(sw_models.columns.get_level_values(0)))
    
    try:
        for num, sw_model in enumerate(sw_model_names):
            
            y = sw_models.loc[datetimes, (sw_model, 'p_dyn')]
            y_upper = sw_models.loc[datetimes, (sw_model, 'p_dyn_pos_unc')]
            y_lower = sw_models.loc[datetimes, (sw_model, 'p_dyn_neg_unc')]               
            
            axd['d'].plot(y.index, y,
                          color=colors[num], label=sw_model)
            axd['d'].fill_between(y.index, y-y_lower, y+y_upper,
                                  alpha=0.5, color=colors[num])
            axd['d'].set_yscale('log')
            
            
        axd['d'].legend()

    except:
        axd['d'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
                          xycoords='axes fraction', textcoords='offset fontsize')
    
    for marker in set(direction_markers):
        mask = np.array(direction_markers) == marker
        axd['d'].scatter(bs_subset.index[mask], bs_subset['p_dyn'].values[mask],
                         marker=marker, c=direction_colors[mask], s=32, linewidth=1)
    axd['d'].set(ylabel='Pressure [nPa]', xlabel='Date')
        
    try:
        for num, sw_model in enumerate(sw_model_names):
            
            y = sw_models.loc[datetimes, (sw_model, 'u_mag')]
            y_upper = sw_models.loc[datetimes, (sw_model, 'u_mag_pos_unc')]
            y_lower = sw_models.loc[datetimes, (sw_model, 'u_mag_neg_unc')]               
            
            axd['e'].plot(y.index, y,
                          color=colors[num], label=sw_model)
            axd['e'].fill_between(y.index, y-y_lower, y+y_upper,
                                  alpha=0.5, color=colors[num])
            axd['e'].set_yscale('log')
            
            
        axd['e'].legend()

    except:
        axd['e'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
                          xycoords='axes fraction', textcoords='offset fontsize')
    axd['e'].set(ylabel='Flow Speed [km/s]', xlabel='Date')
    
    plt.show()
     
    #breakpoint()
    
# for index, row in bowshock_encounters.iterrows():
    
#     nearest_hour = row['datetime'].replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=row['datetime'].minute//30)
    
#     delta = dt.timedelta(hours=24*5)
    
#     datetimes = np.arange(row.index-delta, row.index+delta, dt.timedelta(hours=1)).astype(dt.datetime)
#     datetimes_hourly = np.arange(nearest_hour-delta, nearest_hour+delta, dt.timedelta(hours=1)).astype(dt.datetime)
    
#     fig, axd = plt.subplot_mosaic(
#         """
#         abc
#         ddd
#         """,
#         layout="constrained",)
    
#     with spice.KernelPool('/Users/mrutala/SPICE/juno/metakernel_juno.txt'):
        
#         ets = spice.datetime2et(datetimes)
        
#         pos, lts = spice.spkpos('Juno', ets, 'Juno_JSS', 'LT', 'Jupiter')
#         pos = pos.T / RJ
        
#         crossing, crossing_lt = spice.spkpos('Juno', spice.datetime2et(row['datetime']), 'Juno_JSS', 'LT', 'Jupiter')
#         crossing = crossing.T / RJ
        
    
        
#     #   X-Y plane 
#     axd['a'].plot(pos[0], pos[1])
#     axd['a'].scatter(crossing[0],crossing[1],marker='x', color='black')
#     axd['a'].scatter(0,0,marker='*', color='xkcd:peach')
#     axd['a'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Y_{JSS}$')
    
#     x_bs = np.arange(-100,100.1,.1)
#     y_bs = JBC.find_JoyBowShock(row['p_dyn'], x=x_bs, z=row['z_JSS'])
#     axd['a'].plot(x_bs, y_bs[0], color='gray', linewidth=1)
#     axd['a'].plot(x_bs, y_bs[1], color='gray', linewidth=1)
    
#     #   X-Z plane
#     axd['b'].plot(pos[0], pos[2])
#     axd['b'].scatter(crossing[0],crossing[2],marker='x', color='black')
#     axd['b'].scatter(0,0,marker='*', color='xkcd:peach')
#     axd['b'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Z_{JSS}$')
    
#     x_bs = np.arange(-100,100.1,.1)
#     z_bs = JBC.find_JoyBowShock(row['p_dyn'], x=x_bs, y=row['y_JSS'])
#     axd['b'].plot(x_bs, z_bs[0], color='gray', linewidth=1)
#     axd['b'].plot(x_bs, z_bs[1], color='gray', linewidth=1)
    
#     #   Y-Z Plane
#     axd['c'].plot(pos[1], pos[2])
#     axd['c'].scatter(crossing[1],crossing[2],marker='x', color='black')
#     axd['c'].scatter(0,0,marker='*', color='xkcd:peach')
#     axd['c'].set(xlabel=r'$Y_{JSS}$', ylabel=r'$Z_{JSS}$')
    
#     y_bs = np.arange(-100,100.1,.1)
#     z_bs = JBC.find_JoyBowShock(row['p_dyn'], x=row['x_JSS'], y=y_bs)
#     axd['c'].plot(y_bs, z_bs[0], color='gray', linewidth=1)
#     axd['c'].plot(y_bs, z_bs[1], color='gray', linewidth=1)
    
#     #   MMESH
#     try:
#         y = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn')]
#         y_upper = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn_pos_unc')]
#         y_lower = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn_neg_unc')]               
        
#         axd['d'].plot(sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn')])
#         axd['d'].fill_between(y.index, y-y_lower, y+y_upper,
#                               alpha=0.5)
#         axd['d'].set_yscale('log')
#         axd['d'].axvline(row['datetime'], color='red')
#         axd['d'].axhline(row['p_dyn'], color='red', linestyle='--')
#         axd['d'].set(ylabel='Pressure [nPa]', xlabel='Date')
#     except:
#         axd['d'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
#                           xycoords='axes fraction', textcoords='offset fontsize')
    
#     #axd['e'].plot(sw_models.loc[datetimes_hourly, ('ensemble', 'u_mag')],)
    
#     plt.show()
        
        

    