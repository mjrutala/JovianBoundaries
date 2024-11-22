#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:33:41 2024

@author: mrutala
"""
from pathlib import Path
import pandas as pd
import datetime as dt
import numpy as np

import sys
sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

import matplotlib.pyplot as plt
plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')

import BoundaryModels as BM

def get_paths():
    paths_dict = {'data': Path('/Users/mrutala/projects/JupiterBoundaries/data/'),
                  'SPICE': Path('/Users/mrutala/SPICE/')}
    
    paths_dict = paths_dict | {'UlyssesMetakernel': paths_dict['SPICE']/'ulysses/metakernel_ulysses.txt',
                               'GalileoMetakernel': paths_dict['SPICE']/'galileo/metakernel_galileo.txt',
                               'CassiniMetakernel': paths_dict['SPICE']/'cassini/metakernel_cassini.txt',
                               'JunoMetakernel': paths_dict['SPICE']/'juno/metakernel_juno.txt',
                               'PlanetaryMetakernel': paths_dict['SPICE']/'generic/metakernel_planetary.txt'}
    
    
    paths_dict = paths_dict | {'Louis2023_magnetopause': paths_dict['data'] / 'Louis2023/boundary_crossings_caracteristics_MP.csv', 
                               'Louis2023_bowshock': paths_dict['data'] / 'Louis2023/boundary_crossings_caracteristics_BS.csv',
                               'Kurth_bowshock': paths_dict['data'] / 'Kurth_Waves/Kurth_BowShocks_formatted.csv',
                               'Ebert_magnetopause': paths_dict['data'] / 'Ebert_JADE/Magnetopause_Crossings_Ebert2024_v5.csv',
                               'Ebert_bowshock': paths_dict['data'] / 'Ebert_JADE/Bowshock_Crossings_Ebert2024.csv',
                               'Achilleos2004_bowshock': paths_dict['data']/'Achilleos2004/BowShock_Crossings_Achilleos2004.csv',
                               'Galileo_both': paths_dict['data']/'Galileo/GalileoCrossings.csv',
                               'Bame1992_bowshock': paths_dict['data']/'Bame1992/Crossings_Bame1992.csv'}
    
    return paths_dict

# =============================================================================
# Main routines
# =============================================================================
def convert_CrossingsToRegions(df, resolution = '60Min', padding = None):
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
        
    if padding == None:
        padding = [dt.timedelta(hours=0), dt.timedelta(hours=0)]
    if type(padding[0]) != dt.timedelta:
        breakpoint()
    
    df['notes'] = df['notes'].fillna('null')
    
    # Create a new dataframe with the desired resolution
    loc_columns = ['region', 'region_num', 'SW', 'SH', 'MS', 'UN', 
                   'notes', 'source', 'spacecraft']
    loc_index = pd.date_range(start = df.index[0].floor(resolution) - padding[0],
                              end = df.index[-1].ceil(resolution) + padding[1],
                              freq = resolution)
    region_df = pd.DataFrame(columns = loc_columns, index = loc_index)
    
    # For each row in the crossings df, set the times before and after
    for time, row in df.iterrows():
        
        # As long as the input df is continuous, we only n
        # we only need to do this the first time
        if time == df.index[0]:
            region_df.loc[region_df.query("index < @time").index, 'region'] = row['origin']
            region_df.loc[region_df.query("index < @time").index, 'notes'] = row['notes']
            region_df.loc[region_df.query("index < @time").index, 'source'] = row['source']
            region_df.loc[region_df.query("index < @time").index, 'spacecraft'] = row['spacecraft']
            
        region_df.loc[region_df.query("index >= @time").index, 'region'] = row['destination']
        #   Assign the notes and sources in the same way
        region_df.loc[region_df.query("index >= @time").index, 'notes'] = row['notes']
        region_df.loc[region_df.query("index >= @time").index, 'source'] = row['source']
        region_df.loc[region_df.query("index >= @time").index, 'spacecraft'] = row['spacecraft']
    # Assign regions
    for region in ['SW', 'SH', 'MS', 'UN']:
        region_df.loc[:, region] = 0
        region_df.loc[region_df.query("region== @region").index, region] = 1
        
    region_df['source'] = df['source'].iloc[0]
    
    region_to_num_dict = {'UN': -1,
                          'SW': 0,
                          'SH': 1,
                          'MS': 2}
    region_df['region_num'] = region_df['region'].map(region_to_num_dict)
    
    return region_df
# =============================================================================
# Convenience functions
# =============================================================================
def get_SpacecraftPositions(df, target):
    import spiceypy as spice
    
    mk_paths = [get_paths()['UlyssesMetakernel'].as_posix(), 
                get_paths()['GalileoMetakernel'].as_posix(),
                get_paths()['CassiniMetakernel'].as_posix(),
                get_paths()['JunoMetakernel'].as_posix(),
                get_paths()['PlanetaryMetakernel'].as_posix()]
    
    with spice.KernelPool(mk_paths):
        
        R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
        
        ets = spice.datetime2et(df.index.to_pydatetime())
        
        xyz_pos, lt = spice.spkpos(target, ets, 'Juno_JSS', 'None', 'Jupiter')
        xyz_pos = xyz_pos.T / R_J
        
        rpl_pos = BM.convert_CartesianToCylindricalSolar(*xyz_pos)
        rtp_pos = BM.convert_CartesianToSphericalSolar(*xyz_pos)
        
    df[['x', 'y', 'z']] = xyz_pos.T
    df[['rho', 'phi', 'ell']] = rpl_pos.T
    df[['r', 't', 'p']] = rtp_pos.T
        
    return df

def convert_BoundariesToCrossings(df):
    
    #   Convert destination a Crossing List: add 'origin' and 'destination' columns
    df['origin'] = ''
    df['destination'] = ''
    
    #   origin Solar Wind -> crossing the Bow Shock inward
    origin_sw_indx = df.query("boundary == 'bow shock' & direction == 'in'").index
    df.loc[origin_sw_indx, 'origin'] = 'SW'
    df.loc[origin_sw_indx, 'destination'] = 'SH'
    
    #   destination Solar Wind -> crossing the Bow Shock outward
    destination_sw_indx = df.query("boundary == 'bow shock' & direction == 'out'").index
    df.loc[destination_sw_indx, 'origin'] = 'SH'
    df.loc[destination_sw_indx, 'destination'] = 'SW'
    
    #   origin Magnedestinationsphere (MSP) -> crossing the magnetopause outward
    origin_msp_indx = df.query("boundary == 'magnetopause' & direction == 'out'").index
    df.loc[origin_msp_indx, 'origin'] = 'MS'
    df.loc[origin_msp_indx, 'destination'] = 'SH'
    
    #   destination Magnedestinationsphere (MSP) -> crossing the magnetopause inward
    destination_msp_indx = df.query("boundary == 'magnetopause' & direction == 'in'").index
    df.loc[destination_msp_indx, 'origin'] = 'SH'
    df.loc[destination_msp_indx, 'destination'] = 'MS'
    
    #   For the time being, remove partial crossings
    df = df.query("direction == 'in' | direction == 'out'")
    
    return df[['origin', 'destination', 'notes', 'source', 'spacecraft']]

# =============================================================================
# Routines which read .csv files and return a DataFrame in the preferred format
#   - "Boundary Lists" include boundary locations and directions of crossing
#   - "Crossing Lists" include the m'spheric region the s/c was in before and
#       after a crossing
# =============================================================================
def read_Bame1992_CrossingList(mp=False, bs=True):
    
    crossing_filepath = get_paths()['Bame1992_bowshock']
    
    #   Interpreted from the .csv file
    column_names = ['year', 'doy', 'time', 'origin', 'destination']
    
    #   Read the boundary file, parse the index to datetimes
    crossing_df = pd.read_csv(crossing_filepath, sep = ',', names = column_names, header = 0)
    crossing_df.index = [dt.datetime.strptime('{}-{}T{}'.format(row['year'], row['doy'], row['time']), '%Y-%jT%H:%M') for _, row in crossing_df.iterrows()]
    crossing_df = crossing_df.sort_index()
    
    #   Add required columns
    crossing_df['notes'] = ''
    crossing_df['source'] = 'Bame+ (1992)'
    crossing_df['spacecraft'] = 'Ulysses'
        
    #   Drop unneeded columns
    crossing_df = crossing_df[['origin', 'destination', 'notes', 'source', 'spacecraft']]
    
    return crossing_df

def read_Galileo_CrossingList():
    
    crossing_filepath = get_paths()['Galileo_both']
    
    #   Interpreted from the .csv file
    column_names = ['date', 'origin', 'destination', 'instrument', 'issues', 'inferred']
    
    #   Read the boundary file, parse the index to datetimes
    crossing_df = pd.read_csv(crossing_filepath, sep = ',', names = column_names, header = 1)
    crossing_df.index = [dt.datetime.strptime(row['date'], '%Y-%jT%H:%M') for _, row in crossing_df.iterrows()]
    crossing_df = crossing_df.sort_index()
    
    #   Add required columns
    crossing_df['notes'] = crossing_df['instrument'].fillna('') + '; ' + crossing_df['issues'].fillna('') + crossing_df['inferred'].fillna('')
    crossing_df['source'] = 'Galileo Spreadsheet'
    crossing_df['spacecraft'] = 'Galileo'
    
    # if mp == True:
    #     crossing_df = crossing_df.query("origin == 'MS' | destination == 'MS'")
    # else:
    #     crossing_df = crossing_df.query("origin == 'SW' | destination == 'SW'")
        
    #   Drop unneeded columns
    crossing_df = crossing_df[['origin', 'destination', 'notes', 'source', 'spacecraft']]
    
    return crossing_df

def read_Achilleos2004_CrossingList(bs=True):
    if bs == True:
        crossing_filepath = get_paths()['Achilleos2004_bowshock']
    else:
        print("No magnetopause crossings available from this function!")
        return
    
    #   Interpreted from the .csv file
    column_names = ['year', 'DoY', 'time', 'origin', 'destination']
   
    #   Read the boundary file, parse the index to datetimes
    crossing_df = pd.read_csv(crossing_filepath, sep = ',', names = column_names, header = 0)
    crossing_df.index = [dt.datetime.strptime('{}-{}T{}'.format(row['year'], row['DoY'], row['time']), '%Y-%jT%H:%M') for index, row in crossing_df.iterrows()]
    crossing_df = crossing_df.sort_index()
    
    #   Add required columns
    crossing_df['notes'] = ''
    crossing_df['source'] = 'Achilleos+ (2004)'
    crossing_df['spacecraft'] = 'Cassini'
    
    #   Convert from Boundaries to Crossings
    # crossing_df = convert_BoundariesToCrossings(boundary_df)
    return crossing_df

def read_Louis2023_CrossingList(stoptime = None):
    mp_boundary_filepath = get_paths()['Louis2023_magnetopause']
    bs_boundary_filepath = get_paths()['Louis2023_bowshock']
    
    #   Interpreted from the .csv file
    column_names = ['#', 'DoY', 'date', 'time', 'boundary',  'direction', 'notes', 
                    'x_JSS', 'y_JSS', 'z_JSS', 
                    'x_IAU', 'y_IAU', 'z_IAU', 'r_IAU', 'theta_IAU', 'phi_IAU', 
                    'p_dyn', 'r_mp', 'r_bs']
    
    #   Read the boundary file, parse the index to datetimes
    mp_df = pd.read_csv(mp_boundary_filepath, sep=';', names=column_names, 
                        header=0)
    bs_df = pd.read_csv(bs_boundary_filepath, sep=';', names=column_names, 
                        header=0)
    boundary_df = pd.concat([mp_df, bs_df])
    boundary_df.index = [dt.datetime.strptime(row['date']+'T'+row['time'], '%Y/%m/%dT%H:%M') for index, row in boundary_df.iterrows()]
    boundary_df = boundary_df.sort_index()
    boundary_df['source'] = 'Louis+ (2023)' # Keep track of the source
    boundary_df['spacecraft'] = 'Juno'

    #   Convert from Boundaries to Crossings
    crossing_df = convert_BoundariesToCrossings(boundary_df)
    
    #   Obey stoptime if needed
    if stoptime is not None:
        crossing_df = crossing_df.query("index < @stoptime")
    return crossing_df

def read_Kurth_CrossingList(bs=True):
    if bs == True:
        boundary_filepath = get_paths()['Kurth_bowshock']
    else:
        print("No magnetopause crossings available from this function!")
        return
    
    #   Interpreted from the .csv file
    column_names = ['apojoves', 'date', 'direction', 'notes']
    
    #   Read the boundary file, parse the index to datetimes
    boundary_df = pd.read_csv(boundary_filepath, sep = ',', names = column_names, header = None)
    boundary_df.index = [dt.datetime.strptime(row['date'], '%Y-%jT%H:%M') for index, row in boundary_df.iterrows()]
    boundary_df = boundary_df.sort_index()
    
    #   Add required columns
    boundary_df['boundary'] = 'bow shock'
    boundary_df['source'] = 'Kurth, p.c.'
    
    #   Convert from Boundaries to Crossings
    crossing_df = convert_BoundariesToCrossings(boundary_df)
    return crossing_df

def read_Ebert_CrossingList(mp=False, bs=True):
    if mp == True:
        crossing_list = get_paths()['Ebert_magnetopause']
        crossing_list_names = ['#', 'date', 'time', 
                               'r', 'x_JSO', 'y_JSO', 'z_JSO', 
                               'lat', 'mlat', 'lon', 'LT', 
                               'notes']
        crossing_label = 'magnetopause'
    else:
        crossing_list = get_paths()['Ebert_bowshock']
        crossing_list_names = ['#', 'date', 'time',
                               'r', 'lat', 'mlat', 'MLT']
        crossing_label = 'bow shock'
        
    #   Read the correct list, and drop rows without events
    crossings = pd.read_csv(crossing_list, 
                            sep=',', names = crossing_list_names, header=0,
                            skipinitialspace=True)
    crossings = crossings.dropna(axis = 'index', how = 'all', subset = ['date', 'time'])
    
    #   Set the index to the datetime, rather than an int
    crossings.index = [dt.datetime.strptime(row['date']+'T'+row['time'], '%Y-%jT%H:%M') for _, row in crossings.iterrows()]
    crossings = crossings.sort_index()
    
    #   Add the origin of these crossings to the df
    crossings['boundary'] = crossing_label
    crossings['source'] = 'Ebert (+ Montgomery), p.c.'
    
    return crossings

# =============================================================================
# Routines which concatenate DataFrames in useful ways
# =============================================================================
def make_CombinedCrossingsList(boundary = 'BS', which = ['Louis', 'Kurth', 'Ebert']):
    import spiceypy as spice
    
    match boundary.lower():
        case ('bs' | 'bowshock' | 'bow shock'):
            crossing_data = []
            
            if 'Louis' in which: 
                crossing_data.append(read_Louis2023_CrossingList(bs=True))
            
            if 'Kurth' in which:
                crossing_data_Kurth = read_Kurth_CrossingList(bs=True)
                #   Dropping items which didn't have a clearly listed crossing direction
                crossing_data.append(crossing_data_Kurth[crossing_data_Kurth['direction'].notna()])
            
            if 'Ebert' in which:
                crossing_data.append(read_Ebert_CrossingList(bs=True))
                
            if 'Achilleos' in which:
                crossing_data.append(read_Achilleos2004_CrossingList(bs=True))
            
            crossings_df = pd.concat(crossing_data, axis=0, join="outer")
        case ('mp' | 'magnetopause'):
            crossing_data = []
            
            if 'Louis' in which:
                crossing_data.append(read_Louis2023_CrossingList(mp=True))
            
            if 'Ebert' in which:
                crossing_data.append(read_Ebert_CrossingList(mp=True))
            
            crossings_df = pd.concat(crossing_data, axis=0, join="outer")
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



def convert_DirectionToLocation(starttime, stoptime, bs_crossings, mp_crossings, resolution=1):
    import spiceypy as spice
    import numpy as np
    import tqdm
    
    # if bs_df['direction'].isnull().values.any():
    #     print('All crossing directions are needed to analyze the spacecraft location!')
    #     breakpoint()
    
    datetimes = np.arange(starttime, stoptime, dt.timedelta(minutes=resolution), dtype=dt.datetime)
    
    with spice.KernelPool([get_paths()['PlanetaryMetakernel'].as_posix(), get_paths()['JunoMetakernel'].as_posix()]):
        
        R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
        
        ets = spice.datetime2et(datetimes)
        
        pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
        pos = pos.T / R_J
    
    df = pd.DataFrame({'x': pos[0], 'y': pos[1], 'z':pos[2]}, 
                      index=datetimes)
    
    #   Subset the bow shock df and magnetopause df such that they lie between starttime and stoptime
    
    #   Loop over entries, starting with the bow shock
    df['in_sw'] = np.nan
    df['in_msh'] = np.nan
    df['in_msp'] = np.nan
    
    # last_indx = pd.Timestamp(starttime)
    # #   These intervals should be [last_indx, indx)
    # for indx, row in bs_crossings.iterrows():
        
    #     #   If the spacecraft just crossed *in* to the bow shock, it had been in the solar wind since last_indx
    #     if row['direction'].lower() == 'in':
    #         df.loc[last_indx:indx, 'in_sw'] = 1
    #         df.loc[last_indx:indx, 'in_msh'] = 0
    #         df.loc[last_indx:indx, 'in_msp'] = 0
        
    #     #   If the spacecraft just crossed *out* of the bow shock, it had been in the magnetosheath or magnetosphere since last_indx
    #     if row['direction'].lower() == 'out':
    #         df.loc[last_indx:indx, 'in_sw'] = 0
    #         df.loc[last_indx:indx, 'in_msh'] = 1
    #         df.loc[last_indx:indx, 'in_msp'] = 1
        
    #     last_indx = indx
    
    
    # last_indx = pd.Timestamp(starttime)
    # #   These intervals should be [last_indx, indx)
    # for indx, row in mp_crossings.iterrows():
        
    #     #   If the spacecraft just crossed *in* to the magnetopause, it had been in the magnetosheath alone
    #     if row['direction'].lower() == 'in':
    #         #df.loc[last_indx:indx, 'in_msh'] = 1
    #         df.loc[last_indx:indx, 'in_msp'] = 0
        
    #     #   If the spacecraft just crossed *out* of the magnetopause, it had been in the magnetosphere alone
    #     if row['direction'].lower() == 'out':
    #         df.loc[last_indx:indx, 'in_msh'] = 0
    #         #df.loc[last_indx:indx, 'in_msp'] = 1
            
    #     last_indx = indx
    
    last_indx = pd.Timestamp(starttime)
    combined_crossings = pd.concat([bs_crossings, mp_crossings], axis='index')
    interleaved_crossings = combined_crossings.sort_index()
    
    for this_indx, row in interleaved_crossings.iterrows():
        
        qry = 'index >= @last_indx & index < @this_indx'
        
        if row['boundary'] == 'bow shock':
            if row['direction'] == 'in':
                df.loc[df.query(qry).index, 'in_sw'] = 1
                df.loc[df.query(qry).index, 'in_msh'] = 0
                df.loc[df.query(qry).index, 'in_msp'] = 0
            if row['direction'] == 'out':
                df.loc[df.query(qry).index, 'in_sw'] = 0
                df.loc[df.query(qry).index, 'in_msh'] = 1
                df.loc[df.query(qry).index, 'in_msp'] = 0
        if row['boundary'] == 'magnetopause':
            if row['direction'] == 'in':
                df.loc[df.query(qry).index, 'in_sw'] = 0
                df.loc[df.query(qry).index, 'in_msh'] = 1
                df.loc[df.query(qry).index, 'in_msp'] = 0
            if row['direction'] == 'out':
                df.loc[df.query(qry).index, 'in_sw'] = 0
                df.loc[df.query(qry).index, 'in_msh'] = 0
                df.loc[df.query(qry).index, 'in_msp']= 1
        
        if row['direction'] != 'partial':
            last_indx = this_indx
        
    df['location'] = np.nan
    df.loc[df['in_sw'] == 1, 'location'] = 1
    df.loc[df['in_msh'] == 1, 'location'] = 2
    df.loc[df['in_msp'] == 1, 'location'] = 3
            
    return df
        
def get_HourlyFromDatetimes(ts):
    #   Get the nearest hour to hourly precision
    res = [t.replace(minute=0, second=0, microsecond=0) + 
               dt.timedelta(hours=t.minute//30) for t in ts]
    return res
    

def convert_BoundaryList_to_CrossingList(boundary_df):
    """
    Given a boundary list (indicating which boundary was crossed, and in which 
    direction), generate a crossing list (indicating where the spacecraft was
    on either side of the crossing)

    Parameters
    ----------
    boundary_df : pandas DataFrame
        Minimially contains a datetime index, boundary name, and crossing direction

    Returns
    -------
    crossing_df : pandas DataFrame
        Minimally contains a datetime index, initial location, and final location of spacecraft

    """
    import copy
    
    crossing_df = copy.deepcopy(boundary_df)
    crossing_df['initial'] = ''
    crossing_df['final'] = ''
    
    for indx, row in boundary_df.iterrows():
        
        if ('bow shock' or 'bs') in row['boundary'].lower():
            if row['direction'] == 'in':
                initial, final = 'SW', 'SH'
            else:
                initial, final = 'SH', 'SW'
        elif ('magnetopause' or 'mp') in row['boundary'].lower():
            if row['direction'] == 'in':
                initial, final = 'SH', 'MS'
            else:
                initial, final = 'MS', 'SH'
                
        crossing_df.loc[indx, 'initial'] = initial
        crossing_df.loc[indx, 'final'] = final
    
    return crossing_df    
    
def read_AllCrossings(resolution='10Min', 
                      padding=dt.timedelta(hours=100)):
    
    symmetrical_padding = [padding, padding]
    asymmetrical_padding = [padding, dt.timedelta(hours=0)]
    
    # Load Ulysses crossings
    ulysses_crossings = read_Bame1992_CrossingList()
    ulysses_regions = convert_CrossingsToRegions(ulysses_crossings, resolution, symmetrical_padding)
    ulysses_positions = get_SpacecraftPositions(ulysses_regions, 'Ulysses')
    
    # Load Galileo crossings
    galileo_crossings = read_Galileo_CrossingList()
    galileo_regions = convert_CrossingsToRegions(galileo_crossings, resolution, asymmetrical_padding)
    galileo_positions = get_SpacecraftPositions(galileo_regions, 'GLL')
    
    # Load Cassini crossings
    cassini_crossings = read_Achilleos2004_CrossingList()
    cassini_regions = convert_CrossingsToRegions(cassini_crossings, resolution, symmetrical_padding)
    cassini_positions = get_SpacecraftPositions(cassini_regions, 'Cassini')
    
    # Load Juno crossings 
    juno_crossings = read_Louis2023_CrossingList(stoptime = dt.datetime(2021,7,1))
    juno_regions = convert_CrossingsToRegions(juno_crossings, resolution, asymmetrical_padding)
    juno_positions = get_SpacecraftPositions(juno_regions, 'Juno')
    
    positions_list = [ulysses_positions,
                      galileo_positions,
                      cassini_positions,
                      juno_positions]
    
    df = pd.concat(positions_list)
    df = df.sort_index(axis='index')
    
    return df
    

def plot_CrossingsAndTrajectories():
    import matplotlib.pyplot as plt
    
    padding = dt.timedelta(hours=1000) # Only for flybys
    res = '10Min' # lower resolution than what we actually use, otherwise slow?
    
    #   Load Ulysses data
    ulysses_crossings = read_Bame1992_CrossingList()
    ulysses_regions = convert_CrossingsToRegions(ulysses_crossings, res, [padding, padding])
    ulysses_positions = get_SpacecraftPositions(ulysses_regions, 'Ulysses')
    
    #   Load Cassini data
    cassini_crossings = read_Achilleos2004_CrossingList()
    cassini_regions = convert_CrossingsToRegions(cassini_crossings, res, [padding, padding])
    cassini_positions = get_SpacecraftPositions(cassini_regions, 'Cassini')
    
    galileo_crossings = read_Galileo_CrossingList()
    galileo_regions = convert_CrossingsToRegions(galileo_crossings, res)
    galileo_positions = get_SpacecraftPositions(galileo_regions, 'GLL')
    
    juno_crossings = read_Louis2023_CrossingList(stoptime = dt.datetime(2021,7,1))
    juno_regions = convert_CrossingsToRegions(juno_crossings, res, [padding, dt.timedelta(hours=0)])
    juno_positions = get_SpacecraftPositions(juno_regions, 'Juno')
    
    # Set up colors
    region_colors = {'UN': 'xkcd:gray',
                     'SW': 'xkcd:pale orange',
                     'SH': 'xkcd:turquoise',
                     'MS': 'xkcd:magenta'}
    positions_list = [ulysses_positions,
                      galileo_positions,
                      cassini_positions,
                      juno_positions]
    all_positions = pd.concat(positions_list)
    
    rbin_step = 20
    rbins = np.arange(0, 320, rbin_step)
    abin_step = 2*np.pi/48
    abins = np.arange(0, 2*np.pi + abin_step, abin_step)
    
    z_limit = [-50,50]
    
    # hist, _, _ = np.histogram2d(juno_positions['t'], juno_positions['r'], bins = (abins, rbins))
    A, R = np.meshgrid(abins, rbins)
    
    all_positions['a_LST'] = (all_positions['t'] * np.sign(all_positions['y'])) + np.pi
    equatorial_positions = all_positions.query("@z_limit[0] <= z <= @z_limit[1]")
    
    shape = np.array(np.shape(A)) - 1
    SW_res = np.zeros(shape) - 1
    SH_res = np.zeros(shape) - 1
    MS_res = np.zeros(shape) - 1
    
    for rbin_left in rbins[:-1]:
        rbin_right = rbin_left + rbin_step
        for abin_left in abins[:-1]:
            abin_right = abin_left + abin_step
            
            qry = "@rbin_left <= r < @rbin_right & @abin_left <= a_LST < @abin_right"
            
            r_indx, a_indx = np.where((A == abin_left) & (R == rbin_left))
            
            #   How many total measurements are in this bin?
            total_res = len(equatorial_positions.query(qry).query("UN == 0"))
            
            if total_res != 0:
                
                #   How many SW points are in this bin?
                SW_res[r_indx, a_indx] = len(equatorial_positions.query(qry).query("SW == 1"))/total_res
                
                #   How many SW points are in this bin?
                SH_res[r_indx, a_indx] = len(equatorial_positions.query(qry).query("SH == 1"))/total_res
                
                #   How many SW points are in this bin?
                MS_res[r_indx, a_indx] = len(equatorial_positions.query(qry).query("MS == 1"))/total_res
     
            
    SW_res_masked = np.ma.masked_less(SW_res, 0)
    SH_res_masked = np.ma.masked_less(SH_res, 0)
    MS_res_masked = np.ma.masked_less(MS_res, 0)
    
    rbin_step = 20
    rbins = np.arange(0, 320, rbin_step)
    pbin_step = 2*np.pi/48
    pbins = np.arange(-np.pi, np.pi + pbin_step, pbin_step)
    x_limit = [-50,50]
    
    # hist, _, _ = np.histogram2d(juno_positions['t'], juno_positions['r'], bins = (abins, rbins))
    P, R = np.meshgrid(pbins, rbins)
    
    meridional_positions = all_positions.query("@x_limit[0] <= x <= @x_limit[1]")
    
    shape = np.array(np.shape(P)) - 1
    SW_res_xz = np.zeros(shape) - 1
    SH_res_xz = np.zeros(shape) - 1
    MS_res_xz = np.zeros(shape) - 1
    
    for rbin_left in rbins[:-1]:
        rbin_right = rbin_left + rbin_step
        for pbin_left in pbins[:-1]:
            pbin_right = pbin_left + pbin_step
            
            qry = "@rbin_left <= r < @rbin_right & @pbin_left <= p < @pbin_right"
            
            r_indx, p_indx = np.where((P == pbin_left) & (R == rbin_left))
            
            #   How many total measurements are in this bin?
            total_res = len(meridional_positions.query(qry).query("UN == 0"))
            
            if total_res != 0:
                
                #   How many SW points are in this bin?
                SW_res_xz[r_indx, p_indx] = len(meridional_positions.query(qry).query("SW == 1"))/total_res
                
                #   How many SW points are in this bin?
                SH_res_xz[r_indx, p_indx] = len(meridional_positions.query(qry).query("SH == 1"))/total_res
                
                #   How many SW points are in this bin?
                MS_res_xz[r_indx, p_indx] = len(meridional_positions.query(qry).query("MS == 1"))/total_res
     
    SW_res_xz_masked = np.ma.masked_less(SW_res_xz, 0)
    SH_res_xz_masked = np.ma.masked_less(SH_res_xz, 0)
    MS_res_xz_masked = np.ma.masked_less(MS_res_xz, 0)
    
    cmap = plt.get_cmap('plasma')
    cmap.set_bad('xkcd:light gray', alpha=0.5)
    
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize = (6,5),
                            subplot_kw={'projection':'polar'})
    plt.subplots_adjust(left=0.12, bottom=0.04, right=0.96, top=0.78, 
                        hspace=0.25, wspace=0.4)
    
    im = axs[0,0].pcolormesh(A, R, SW_res_masked, cmap=cmap, vmin=0, vmax=1)      
    # axs[0,0].set_title('In the Solar Wind')
    axs[0,0].annotate('In the Solar Wind', (0.5, 1.25), (0,0),
                      'axes fraction', 'offset fontsize',
                      rotation = 'horizontal', ha = 'center', va = 'center')
    axs[0,0].annotate('Local Solar Time [Hours]', (-0.25,0.5), (0,0),
                      'axes fraction', 'offset fontsize', 
                      rotation = 'vertical', ha = 'center', va = 'center')
    axs[1,0].annotate('Meridional Angle [$\circ$]', (-0.25,0.5), (0,0),
                      'axes fraction', 'offset fontsize', 
                      rotation = 'vertical', ha = 'center', va = 'center')
    
    axs[0,1].pcolormesh(A, R, SH_res_masked, cmap=cmap, vmin=0, vmax=1)     
    # axs[0,1].set_title('In the Magnetosheath')
    axs[0,1].annotate('In the Magnetosheath', (0.5, 1.25), (0,0),
                      'axes fraction', 'offset fontsize',
                      rotation = 'horizontal', ha = 'center', va = 'center')
     
    axs[0,2].pcolormesh(A, R, MS_res_masked, cmap=cmap, vmin=0, vmax=1)     
    # axs[0,2].set_title('In the Magnetosphere')
    axs[0,2].annotate('In the Magnetosphere', (0.5, 1.25), (0,0),
                      'axes fraction', 'offset fontsize',
                      rotation = 'horizontal', ha = 'center', va = 'center')

    axs[1,0].pcolormesh(P, R, SW_res_xz_masked, cmap=cmap, vmin=0, vmax=1)      
    
    axs[1,1].pcolormesh(P, R, SH_res_xz_masked, cmap=cmap, vmin=0, vmax=1)     
     
    axs[1,2].pcolormesh(P, R, MS_res_xz_masked, cmap=cmap, vmin=0, vmax=1)     

    for ax in axs.flatten():
        ax.tick_params(axis='x', which='major', pad=0)
        ax.set_rlabel_position(-30)
    for ax in axs[0].flatten():
        ax.set(xticks = np.radians(np.arange(0, 360, 45)), 
               xticklabels = ['{:02d}'.format(t) for t in np.roll(np.arange(0, 24, 3), 0)])
    for ax in axs[1].flatten():
        ax.set_theta_offset(np.pi/2.)
    
    cbar_ax = fig.add_axes([0.2, 0.9, 0.68, 0.04])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticklocation='bottom')
    cbar_ax.set_title('Fraction of all Spacecraft Measurements')
    
    plt.show()

    
def orbital_plots():
    padding = dt.timedelta(hours=1000) # Only for flybys
    res = '10Min' # lower resolution than what we actually use, otherwise slow?
    
    #   Load Ulysses data
    ulysses_crossings = read_Bame1992_CrossingList()
    ulysses_regions = convert_CrossingsToRegions(ulysses_crossings, res, [padding, padding])
    ulysses_positions = get_SpacecraftPositions(ulysses_regions, 'Ulysses')
    
    #   Load Cassini data
    cassini_crossings = read_Achilleos2004_CrossingList()
    cassini_regions = convert_CrossingsToRegions(cassini_crossings, res, [padding, padding])
    cassini_positions = get_SpacecraftPositions(cassini_regions, 'Cassini')
    
    galileo_crossings = read_Galileo_CrossingList()
    galileo_regions = convert_CrossingsToRegions(galileo_crossings, res)
    galileo_positions = get_SpacecraftPositions(galileo_regions, 'GLL')
    
    juno_crossings = read_Louis2023_CrossingList(stoptime = dt.datetime(2021,7,1))
    juno_regions = convert_CrossingsToRegions(juno_crossings, res, [padding, dt.timedelta(hours=0)])
    juno_positions = get_SpacecraftPositions(juno_regions, 'Juno')
    
    # Set up colors
    region_colors = {'UN': 'xkcd:gray',
                     'SW': 'xkcd:pale orange',
                     'SH': 'xkcd:turquoise',
                     'MS': 'xkcd:magenta'}
    
    # Set up plot
    fig, axs = plt.subplots(figsize = (6,4), nrows=2, ncols=2, 
                            sharex='col', width_ratios=[2,1])
    plt.subplots_adjust(left=0.1, right=0.9)
    
    fig2, axs2 = plt.subplots(figsize = (6,4), nrows=2, ncols=2, 
                            sharex='col', width_ratios=[2,1])
    plt.subplots_adjust(left=0.1, right=0.9)
    
    positions_list = [ulysses_positions, 
                       cassini_positions,
                       galileo_positions,
                       juno_positions]
    names = ['Ulysses', 'Cassini', 'Galileo', 'Juno']
    
    for positions, name in zip(positions_list, names):
        n = 10
        # Plot Ulysses Flyby
        axs[0,0].scatter(positions['x'][::n], positions['y'][::n],
                       c = positions['region'][::n].map(region_colors),
                       marker = '.', s = 0.5, alpha=0.01)
        axs[0,1].scatter(positions['x'][::n], positions['y'][::n],
                       c = positions['region'][::n].map(region_colors),
                       marker = '.', s = 0.5, alpha=0.01)
        
        axs[1,0].scatter(positions['x'][::n], positions['z'][::n],
                       c = positions['region'][::n].map(region_colors),
                       marker = '.', s = 0.5, alpha=0.01)
        axs[1,1].scatter(positions['x'][::n], positions['z'][::n],
                       c = positions['region'][::n].map(region_colors),
                       marker = '.', s = 0.5, alpha=0.01)
    
        
        axs2[0,0].scatter(positions['x'], positions['y'],
                          label = name,
                          marker = '.', s = 0.5)
        axs2[0,1].scatter(positions['x'], positions['y'],
                          label = name,
                          marker = '.', s = 0.5)
        
        axs2[1,0].scatter(positions['x'], positions['z'],
                          label = name,
                          marker = '.', s = 0.5)
        axs2[1,1].scatter(positions['x'], positions['z'],
                          label = name,
                          marker = '.', s = 0.5)
    
    axs2[0,0].legend()
    
    axs[0,0].set(ylim = [-200, 200], ylabel = r'$y_{JSS} [R_J]$ (+ Duskward)',
               aspect=1)
    axs[0,1].set(ylim = [-100, 100], aspect=1)
    
    axs[1,0].set(xlim = [300, -300], xlabel = r'$x_{JSS} [R_J]$ (+ Sunward)', 
               ylim = [-200, 200], ylabel = r'$z_{JSS} [R_J]$ (+ Northward)',
               aspect=1)
    axs[1,1].set(xlim = [100, -100],
               ylim = [-100, 100], 
               aspect=1)
    
    axs2[0,0].set(ylim = [-200, 200], ylabel = r'$y_{JSS} [R_J]$ (+ Duskward)',
               aspect=1)
    axs2[0,1].set(ylim = [-100, 100], aspect=1)
    
    axs2[1,0].set(xlim = [300, -300], xlabel = r'$x_{JSS} [R_J]$ (+ Sunward)', 
               ylim = [-200, 200], ylabel = r'$z_{JSS} [R_J]$ (+ Northward)',
               aspect=1)
    axs2[1,1].set(xlim = [100, -100],
               ylim = [-100, 100], 
               aspect=1)
    
    plt.show()    

# def _plot():
    
#     boundaries
    
#     crossings_galileo = read_Galileo_CrossingList()
    
#     boundaries_cassini = read_Achilleos2004_BoundaryList()
#     crossings_cassini = convert_BoundaryList_to_CrossingList(boundaries_cassini)
    
    


# def plot_CrossingsAndTrajectories_XYPlane(joy=False, mme=False):
#     import spiceypy as spice
#     import numpy as np
    
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
    
#     import JoyBoundaryCoords as JBC

#     mp_crossing_data = make_CombinedCrossingsList(boundary = 'MP')
#     mp_crossing_data = make_HourlyCrossingList(mp_crossing_data)
    
#     #   Bow Shock Crossings
#     bs_crossing_data = make_CombinedCrossingsList(boundary = 'BS')
#     bs_crossing_data = make_HourlyCrossingList(bs_crossing_data)
    
#     #   Load SPICE kernels for positions of Juno
#     spice.furnsh(get_paths()['PlanetaryMetakernel'].as_posix())
#     spice.furnsh(get_paths()['JunoMetakernel'].as_posix())
    
#     R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
    
#     #   Get all hourly trajectory info
#     earliest_time = min(np.append(bs_crossing_data.index, mp_crossing_data.index))
#     latest_time = max(np.append(bs_crossing_data.index, mp_crossing_data.index))
    
#     datetimes = np.arange(pd.Timestamp(earliest_time).replace(minute=0, second=0, nanosecond=0),
#                           pd.Timestamp(latest_time).replace(minute=0, second=0, nanosecond=0) + dt.timedelta(hours=1),
#                           dt.timedelta(hours=1)).astype(dt.datetime)
    
#     ets = spice.datetime2et(datetimes)
    
#     pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
#     pos = pos.T / R_J
#     pos_df = pd.DataFrame({'x': pos[0], 'y': pos[1], 'z':pos[2]}, 
#                           index=datetimes)

#     pos_df['r'] = np.sqrt(np.sum(pos_df[['x', 'y']].to_numpy()**2, 1))

    
#     spice.kclear()
    
#     sw_models = MMESH_reader.fromFile('/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv')
#     sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
#     with plt.style.context('/Users/mrutala/code/python/mjr_presentation.mplstyle'):
#         fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True, 
#                                 figsize=(8,6))
#         bottom, left, top, right = 0.275, 0.125, 0.825, 0.975
#         plt.subplots_adjust(bottom=bottom, left=left, top=top, right=right,
#                             wspace=0.075)
        
        
#         for ax, crossing_type in zip(axs, ['bs', 'mp']):
            
#             y_joy = np.linspace(-500, 500, 10000)
            
#             if crossing_type == 'bs':
#                 crossing_data = bs_crossing_data
#                 marker = 'x'
#                 x_joy = JBC.find_JoyBowShock(0.1, y=y_joy, z=0)
#                 color_joy, linestyle_joy = 'C0', 'solid'
                
#             if crossing_type == 'mp':
#                 crossing_data = mp_crossing_data
#                 marker = 'o'
#                 x_joy = JBC.find_JoyMagnetopause(0.1, y=y_joy, z=0)
#                 color_joy, linestyle_joy = 'C4', '--'
            
#             #fig, ax = plt.subplots(figsize=(8,8))
            
#             ax.plot(pos_df['y'], pos_df['x'], 
#                     color='xkcd:light gray', linewidth=0.5, 
#                     zorder=1)
            
#             #   Scatter plot for crossings
#             if mme:
#                 if crossing_type == 'bs':
#                     crossing_data = bs_crossing_data
#                     marker = 'x'
#                 if crossing_type == 'mp':
#                     crossing_data = mp_crossing_data
#                     marker = 'o'
                    
#                 bounds = np.logspace(-2, -1, 11)
#                 cmap = plt.get_cmap('magma', 11)
#                 norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                
#                 ax.scatter(*pos_df.loc[crossing_data.index, ['y', 'x']].to_numpy().T,
#                            c = sw_mme.loc[crossing_data.index, 'p_dyn'],
#                            cmap = cmap, norm = norm, s = 12, 
#                            marker = marker, zorder=2)
#             else:
            
#                 ax.scatter(*pos_df.loc[crossing_data.index, ['y', 'x']].to_numpy().T,
#                            color = 'white', s = 12, 
#                            marker = marker, zorder=2)
            
#             #   Plot years for reference
#             for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
#                 xy = pos_df.loc[pd.Timestamp(dt.datetime(year, 1, 1)), ['x', 'y']].to_list()
#                 ang = np.arctan2(xy[0], xy[1])
                
#                 indx = ((pos_df.index > dt.datetime(year, 1, 1) - dt.timedelta(days = 40)) &
#                         (pos_df.index < dt.datetime(year, 1, 1) + dt.timedelta(days = 40)))
                
#                 max_r = np.max(pos_df.iloc[indx].loc[:, 'r']) + 15
                
#                 ax.annotate(year, (max_r*np.cos(ang), max_r*np.sin(ang)), (0,0), 
#                             xycoords = 'data', textcoords = 'offset fontsize', 
#                             rotation = (-90 - (ang * 180/np.pi + 360)),
#                             ha = 'center', va = 'center', annotation_clip=True,
#                             fontsize='xx-small')
                
#             #   Plot Joy bounds (optional)
#             if joy:
#                 x_joy = JBC.find_JoyBoundaries(0.02, boundary = crossing_type.upper(), y=y_joy, z=0)
#                 ax.plot(y_joy, x_joy[1], label = r'$p_{{dyn}} = {:.2f} = 10^{{ {:.1f} }}$ nPa ($16^{{th}}$ %ile)'.format(0.03, np.log10(0.03)),
#                          color='C2', linewidth=1.5, linestyle=linestyle_joy)
                
#                 x_joy = JBC.find_JoyBoundaries(0.05, boundary = crossing_type.upper(), y=y_joy, z=0)
#                 ax.plot(y_joy, x_joy[1], label = r'$p_{{dyn}} = {:.2f} = 10^{{ {:.1f} }}$ nPa ($50^{{th}}$ %ile)'.format(0.08, np.log10(0.08)),
#                          color='C4', linewidth=1.5, linestyle=linestyle_joy)
                
#                 x_joy = JBC.find_JoyBoundaries(0.13, boundary = crossing_type.upper(), y=y_joy, z=0)
#                 ax.plot(y_joy, x_joy[1], label = r'$p_{{dyn}} = {:.2f} = 10^{{ {:.1f} }}$ nPa ($84^{{th}}$ %ile)'.format(0.13, np.log10(0.13)),
#                          color='C5', linewidth=1.5, linestyle=linestyle_joy)
                
#                 x_joy = JBC.find_JoyBoundaries(0.47, boundary = crossing_type.upper(), y=y_joy, z=0)
#                 ax.plot(y_joy, x_joy[1], label = r'$p_{{dyn}} = {:.2f} = 10^{{ {:.1f} }}$ nPa ($99^{{th}}$ %ile)'.format(0.47, np.log10(0.47)),
#                          color='C0', linewidth=1.5, linestyle=linestyle_joy)
                
#                 if crossing_type == 'bs': 
#                     ax.legend(loc='lower center', bbox_to_anchor=(1.04, 1.075), ncol=2, fontsize='small')
                
#             ax.set(xlim = [-150, 150], ylim = [150,-150],
#                    aspect = 1)
            
#         axs[0].set_title('Bow Shock Crossings (n = 117)', fontsize='small')
#         axs[1].set_title('Magnetopause Crossings (n = 454)', fontsize='small')
        
#         fig.supxlabel(r'$Y_{JSS}$ [$R_J$] (+ duskward)', x=0.55, y=0.175)
#         axs[0].set(ylabel = r'$X_{JSS}$ [$R_J$] (+ sunward)')
        
#         if mme:
#             ax2 = fig.add_axes([left, 0.125, right-left, 0.05])
            
#             cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2, orientation='horizontal')
#             #cb.ax.set_yticklabels([r'$10^{{{:.1f}}}$'.format(np.log10(num)) for num in cb.get_ticks()])
#             cb.ax.set_xticklabels([r'{:.1f}'.format(np.log10(num)) for num in bounds])
#             ax2.set(xlabel=r'$p_{dyn}$ [log(nPa)]')
        
#         plt.show()

# def plot_CrossingsAndTrajectories():
#     import spiceypy as spice
#     import numpy as np
    
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt

#     mp_crossing_data = make_CombinedCrossingsList(boundary = 'MP')
#     mp_crossing_data = make_HourlyCrossingList(mp_crossing_data)
    
#     #   Bow Shock Crossings
#     bs_crossing_data = make_CombinedCrossingsList(boundary = 'BS')
#     bs_crossing_data = make_HourlyCrossingList(bs_crossing_data)
    
#     #   Load SPICE kernels for positions of Juno
#     spice.furnsh(get_paths()['PlanetaryMetakernel'].as_posix())
#     spice.furnsh(get_paths()['JunoMetakernel'].as_posix())
    
#     R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
    
#     #   Get all hourly trajectory info
#     earliest_time = min(np.append(bs_crossing_data.index, mp_crossing_data.index))
#     latest_time = max(np.append(bs_crossing_data.index, mp_crossing_data.index))
    
#     datetimes = np.arange(pd.Timestamp(earliest_time).replace(minute=0, second=0, nanosecond=0),
#                           pd.Timestamp(latest_time).replace(minute=0, second=0, nanosecond=0) + dt.timedelta(hours=1),
#                           dt.timedelta(hours=1)).astype(dt.datetime)
    
#     ets = spice.datetime2et(datetimes)
    
#     pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
#     pos = pos.T / R_J
#     pos_df = pd.DataFrame({'x': pos[0], 'y': pos[1], 'z':pos[2]}, 
#                           index=datetimes)
    
#     spice.kclear()
    
#     sw_models = MMESH_reader.fromFile('/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv')
#     sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
#     for crossing_type in ['bs', 'mp']:
#         if crossing_type == 'bs':
#             crossing_data = bs_crossing_data
#             marker = 'x'
#         if crossing_type == 'mp':
#             crossing_data = mp_crossing_data
#             marker = 'o'
        
#         fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6,5))
#         plt.subplots_adjust(left=0.1, bottom=0.14, right=0.8, top=0.98, 
#                             wspace=0.05, hspace=0.05)
#         ax2 = fig.add_axes([0.82, 0.14, 0.04, 0.84])
        
#         axs[1,0].plot(pos_df['y'], pos_df['z'], 
#                       color='black', linewidth=0.5, zorder=1)
#         axs[1,0].annotate('Y-Z Plane', (0, 1), (1, -1), ha='left',
#                           xycoords='axes fraction', textcoords='offset fontsize')
        
#         axs[1,1].plot(pos_df['x'], pos_df['z'],
#                       color='black', linewidth=0.5, zorder=1)
#         axs[1,1].annotate('X-Z Plane', (0, 1), (1, -1), ha='left',
#                           xycoords='axes fraction', textcoords='offset fontsize')
        
#         axs[0,0].plot(pos_df['y'], pos_df['x'], 
#                       color='black', linewidth=0.5, zorder=1)
#         axs[0,0].annotate('Y-X Plane', (0, 1), (1, -1), ha='left',
#                           xycoords='axes fraction', textcoords='offset fontsize')
        
#         axs[0,1].set_axis_off()
#         # axs[0,1].annotate('3D View WIP', (0, 1), (1, -1), ha='left',
#         #                   xycoords='axes fraction', textcoords='offset fontsize')
        
#         #
#         bounds = np.logspace(-2, -1, 11)
#         cmap = plt.get_cmap('magma', 11)
#         norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
#         cs = axs[1,0].scatter(*pos_df.loc[crossing_data.index, ['y', 'z']].to_numpy().T,
#                               c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
#                               cmap = cmap, norm = norm,
#                               marker = marker, zorder=2)
#         axs[1,1].scatter(*pos_df.loc[crossing_data.index, ['x', 'z']].to_numpy().T,
#                               c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
#                               cmap = cmap, norm = norm,
#                               marker = marker, zorder=2)
#         axs[0,0].scatter(*pos_df.loc[crossing_data.index, ['y', 'x']].to_numpy().T,
#                               c = sw_mme.loc[crossing_data.index, 'p_dyn'].to_numpy(),
#                               cmap = cmap, norm = norm,
#                               marker = marker, zorder=2)
        
#         axs[1,0].set(xlabel = r'$Y_{JSS}$ (+ duskward) [$R_J$]', xlim = [-120, 80], 
#                      ylabel = r'$Z_{JSS}$ (+ north) [$R_J$]', ylim = [-120, 80])
#         axs[0,0].set(ylabel = r'$X_{JSS}$ (+ sunward) [$R_J$]', ylim = [40,-160])
#         axs[1,1].set(xlabel = r'$X_{JSS}$ (+ sunward) [$R_J$]', xlim = [40,-160])
        
#         cb = plt.colorbar(cs, cax=ax2)
#         #cb.ax.set_yticklabels([r'$10^{{{:.1f}}}$'.format(np.log10(num)) for num in cb.get_ticks()])
#         cb.ax.set_yticklabels([r'{:.1f}'.format(np.log10(num)) for num in bounds])
#         ax2.set(ylabel=r'Solar Wind Pressure [log(nPa)]')
        
#         plt.show()
    
#     breakpoint()




