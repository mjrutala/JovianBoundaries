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
from scipy.optimize import curve_fit
from scipy import odr
import pandas as pd


import sys
import JoyBoundaryCoords as JBC

sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

import boundaries

plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')

R_J = 71492.

# =============================================================================
# Section 1: Functions for reading in data, models, crossing lists
# =============================================================================

def read_SolarRadioFlux():
    sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
    import MMESH_context
    df = MMESH_context.read_SolarRadioFlux(dt.datetime(1970,1,1), dt.datetime(2024,3,1))
    return df

def nearest_hour(datetimes):
    result = [t.replace(minute=0, second=0, microsecond=0) + 
              dt.timedelta(hours=t.minute//30) for t in datetimes]
    return np.array(result)


# =============================================================================
# 
# =============================================================================
def compare_MMESHToData():
    import scipy
    import numpy as np
    
    import sys
    sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
    import spacecraftdata
    
    epochs = {}
    epochs['Pioneer11'] = {'spacecraft_name':'Pioneer 11',
                              'span':(dt.datetime(1977, 6, 3), dt.datetime(1977, 7, 29))}
    epochs['Voyager1'] = {'spacecraft_name':'Voyager 1',
                              'span':(dt.datetime(1979, 1, 3), dt.datetime(1979, 5, 5))}
    epochs['Voyager2'] = {'spacecraft_name':'Voyager 2',
                              'span':(dt.datetime(1979, 3, 30), dt.datetime(1979, 8, 20))}
    epochs['Ulysses_01']  = {'spacecraft_name':'Ulysses',
                              'span':(dt.datetime(1991,12, 8), dt.datetime(1992, 2, 2))}
    epochs['Ulysses_02']  = {'spacecraft_name':'Ulysses',
                              'span':(dt.datetime(1997, 8,14), dt.datetime(1998, 4,16))}
    epochs['Ulysses_03']  = {'spacecraft_name':'Ulysses',
                              'span':(dt.datetime(2003,10,24), dt.datetime(2004, 6,22))}
    epochs['Juno']     = {'spacecraft_name':'Juno',
                              'span':(dt.datetime(2016, 5,16), dt.datetime(2016, 6,25))}
    
    param = 'p_dyn'
    #  Load spacecraft data
    temp_dict = {'datetime':[], param:[]}
    for interval, info in epochs.items():
        spacecraft = spacecraftdata.SpacecraftData(info['spacecraft_name'])
        starttime, stoptime = info['span']
        spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
        
        if len(spacecraft.data) > 0:
            temp_dict['datetime'].extend(spacecraft.data.index.to_numpy())
            temp_dict[param].extend(spacecraft.data.loc[:, param].to_numpy()) 
    # temp_dict['datetime'] = np.array(concat['datetime'])
    # temp_dict[param] = np.array(concat[param])
    data_df = pd.DataFrame.from_dict(temp_dict)
    data_df = data_df.set_index('datetime')
    
    # Combine the data df with a solar radio flux df
    srf = read_SolarRadioFlux().rolling('9490h').mean().loc[:, 'adjusted_flux']
    data_df = data_df.merge(srf, how='outer', left_index=True, right_index=True)
    
# =============================================================================
#   Loading MME and Crossing Lists
# =============================================================================
    import JunoPreprocessingRoutines as PR 
    import JoyBoundaryCoords as JBC
    sw_mme_filepath = '/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv'
    
    mp_crossing_data = PR.read_Louis2023_CrossingList(mp=True)
    
    bs_crossing_data_Louis2023 = PR.read_Louis2023_CrossingList(bs=True)
    
    bs_crossing_data_Kurth = PR.read_Kurth_CrossingList(bs=True)
    bs_crossing_data_Kurth = bs_crossing_data_Kurth[bs_crossing_data_Kurth['direction'].notna()]
    
    bs_crossing_data = pd.concat([bs_crossing_data_Louis2023, bs_crossing_data_Kurth], axis=0, join="outer")
    
    #   Only pass the datetime index, the direction of the crossing, and any notes
    hourly_mp_crossing_data = PR.make_HourlyCrossingList(mp_crossing_data)
    hourly_mp_crossing_data['p_dyn'] = JBC.find_JoyPressures(*hourly_mp_crossing_data[['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T, 'MP')
    
    hourly_bs_crossing_data = PR.make_HourlyCrossingList(bs_crossing_data)
    hourly_bs_crossing_data['p_dyn'] = JBC.find_JoyPressures(*hourly_bs_crossing_data[['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T, 'BS')

    #   Load the solar wind data and select the ensesmble
    sw_models = MMESH_reader.fromFile(sw_mme_filepath)
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
    
# =============================================================================
#     Plotting stuff
# =============================================================================
    #   Setup some standardized bins

    
    # #   1: How do the measured and inferred p_dyn compare?
    # fig, ax = plt.subplots(figsize=(4,4))
    
    # data_hist, bins = np.histogram(data_df['p_dyn'], bins=pressure_bins, 
    #                                density=True)
    # inferred_hist, bins = np.histogram(inferred['p_dyn'], bins=pressure_bins, 
    #                                    density=True)
    
    # #   Find the probability of a single inferred value
    # bin_widths = bins[1:] - bins[:-1]
    # agreement_index = bins[:-1] > np.min(inferred['p_dyn'])
    # agreement_prob = np.sum((data_hist * bin_widths)[agreement_index])
    # ax.annotate('Agreement Probability (Single Inference): {:.2f}%'.format(agreement_prob*100),
    #             (0,1), (0,1), xycoords='axes fraction', textcoords='offset fontsize')
    
    # ax.stairs(data_hist, bins, 
    #           linewidth=2, 
    #           label='Spacecraft Measurements')
    # ax.stairs(inferred_hist, bins, 
    #           linewidth=2, 
    #           label='Inferred (Joy+ 2002, Louis+ 2023)')
    
    # ax.set(xlabel='Solar Wind Dynamic Pressure $(p_{dyn})$ [nPa]', xscale='log',
    #        ylabel='Data Density')
    # ax.legend()
    # plt.show()
    
    # breakpoint()
    
    # #   2: How do the measured and inferred p_dyn compare if we only look at Juno-era?
    # fig, ax = plt.subplots(figsize=(4,4))
    
    # temp_data_df = data_df[data_df.index > epochs['Juno']['span'][0]]
    # data_hist, bins = np.histogram(temp_data_df['p_dyn'], bins=pressure_bins, 
    #                                density=True)
    # inferred_hist, bins = np.histogram(inferred['p_dyn'], bins=pressure_bins, 
    #                                    density=True)
    
    # #   Find the probability of a single inferred value
    # bin_widths = bins[1:] - bins[:-1]
    # agreement_index = bins[:-1] > np.min(inferred['p_dyn'])
    # agreement_prob = np.sum((data_hist * bin_widths)[agreement_index])
    # ax.annotate('Agreement Probability (Single Inference): {:.2f}%'.format(agreement_prob*100),
    #             (0,1), (0,1), xycoords='axes fraction', textcoords='offset fontsize')
    
    # ax.stairs(data_hist, bins, 
    #           linewidth=2, 
    #           label='Juno Measurements')
    # ax.stairs(inferred_hist, bins, 
    #           linewidth=2, 
    #           label='Inferred (Joy+ 2002, Louis+ 2023)')
    
    # ax.set(xlabel='Solar Wind Dynamic Pressure $(p_{dyn})$ [nPa]', xscale='log',
    #        ylabel='Data Density')
    # ax.legend()
    # plt.show()
    
    # #   3: How does measured p_dyn vary with the solar cycle?
    # fig, ax = plt.subplots(figsize=(4,4))
    # hist, xedges, yedges = np.histogram2d(data_df.dropna(how='any')['p_dyn'], 
    #                                       data_df.dropna(how='any')['adjusted_flux'],
    #                                       bins=[pressure_bins, sfu_bins],
    #                                       density=True)
    
    # pcm = ax.pcolormesh(xedges, yedges, hist.T, norm='log')
    # ax.set(xlabel='Measured Solar Wind Dynamic Pressure ($p_{dyn}$) [nPa]', xscale='log',
    #        ylabel='13-month Mean Solar F10.7cm (Solar Cycle Phase) [SFU]')
    
    # fig.colorbar(pcm, ax=ax, orientation='vertical', label='Data Density')

    # plt.show()
    
    #   4: How does an MME stack up?
    pressure_bins = np.logspace(-3, 1, 41)
    sfu_bins = np.linspace(0,300,16)
    fig, ax = plt.subplots(figsize=(4,4))

    data_hist, _ = np.histogram(data_df['p_dyn'], pressure_bins, density=True)
    model_hist, _ = np.histogram(sw_mme['p_dyn'], pressure_bins, density=True)
    combined_inferred_p_dyn = np.concatenate([hourly_bs_crossing_data['p_dyn'].to_numpy(), 
                                        hourly_mp_crossing_data['p_dyn'].to_numpy()])
    inferred_hist, bins = np.histogram(combined_inferred_p_dyn, pressure_bins, density=True)
    
    ax.stairs(data_hist, bins, linewidth=2, label='Spacecraft Measurements', color='C0')
    ax.stairs(inferred_hist, bins, linewidth=2, label='Inferred w/ Joy+ (2002)', color='xkcd:lavender')
    ax.stairs(model_hist, bins, linewidth=2, label='MME', color='C3')
    ax.set(xlabel='Solar Wind Dynamic Pressure $(p_{dyn})$ [nPa]', xscale='log',
           ylabel='Data Density')

    ax.legend()
    plt.show()
    
    pressure_bins = np.linspace(-3, 1, 41)
    
    fig, ax = plt.subplots(figsize=(4,4))

    data_hist, _ = np.histogram(np.log10(data_df['p_dyn']), pressure_bins, density=True)
    
    combined_mme_p_dyn = np.concatenate([sw_mme['p_dyn']+sw_mme['p_dyn_pos_unc'], 
                                         sw_mme['p_dyn'],
                                         sw_mme['p_dyn']-sw_mme['p_dyn_neg_unc']])
    model_hist, _ = np.histogram(np.log10(combined_mme_p_dyn), pressure_bins, density=True)
   
    combined_inferred_p_dyn = np.concatenate([hourly_bs_crossing_data['p_dyn'].to_numpy(), 
                                        hourly_mp_crossing_data['p_dyn'].to_numpy()])
    inferred_hist, bins = np.histogram(np.log10(combined_inferred_p_dyn), pressure_bins, density=True)
    
    ax.stairs(data_hist, bins, linewidth=2, label='Spacecraft Measurements', color='C0')
    ax.stairs(inferred_hist, bins, linewidth=2, label='Inferred w/ Joy+ (2002)', color='xkcd:lavender')
    ax.stairs(model_hist, bins, linewidth=2, label='MME', color='C4')
    ax.set(xlabel='Log Solar Wind Dynamic Pressure $\mathrm{log}_{10}(p_{dyn}/nPa)$',
           ylabel='Data Density')

    ax.legend()
    plt.show()
    
    breakpoint()
    return

# def compare_MMESHToData_Overlapping():
#     import scipy
#     import numpy as np
    
#     import sys
#     sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
#     import spacecraftdata
    
#     epochs = {}
#     epochs['Pioneer11'] = {'spacecraft_name':'Pioneer 11',
#                               'span':(dt.datetime(1977, 6, 3), dt.datetime(1977, 7, 29))}
#     epochs['Voyager1'] = {'spacecraft_name':'Voyager 1',
#                               'span':(dt.datetime(1979, 1, 3), dt.datetime(1979, 5, 5))}
#     epochs['Voyager2'] = {'spacecraft_name':'Voyager 2',
#                               'span':(dt.datetime(1979, 3, 30), dt.datetime(1979, 8, 20))}
#     epochs['Ulysses_01']  = {'spacecraft_name':'Ulysses',
#                               'span':(dt.datetime(1991,12, 8), dt.datetime(1992, 2, 2))}
#     epochs['Ulysses_02']  = {'spacecraft_name':'Ulysses',
#                               'span':(dt.datetime(1997, 8,14), dt.datetime(1998, 4,16))}
#     epochs['Ulysses_03']  = {'spacecraft_name':'Ulysses',
#                               'span':(dt.datetime(2003,10,24), dt.datetime(2004, 6,22))}
#     epochs['Juno']     = {'spacecraft_name':'Juno',
#                               'span':(dt.datetime(2016, 5,16), dt.datetime(2016, 6,26))}
    
#     #  Load spacecraft data
#     for interval, info in epochs.items():
#         spacecraft = spacecraftdata.SpacecraftData(info['spacecraft_name'])
#         starttime, stoptime = info['span']
#         spacecraft.read_processeddata(starttime, stoptime, resolution='60Min')
        
#         breakpoint()
        
#         data = spacecraft.data['p_dyn'].dropna()
#         #model = sw_models.loc[spacecraft.data.index, ('ensemble', 'p_dyn')]
#         model = sw_models.loc[:, ('ensemble', 'p_dyn')]
#         inferred = bs_crossings_Louis2023['p_dyn']
#         breakpoint()
#         bins = np.logspace(-3,1,41)
#         data_hist, _ = np.histogram(data, bins, density=True)
#         model_hist, _ = np.histogram(model, bins, density=True)
#         inferred_hist, bins = np.histogram(inferred, bins, density=True)
        
#         ks = scipy.stats.kstest(data, model)
#         print(ks)
        
#         data_kde = scipy.stats.gaussian_kde(data.to_numpy('float64'))
        
#         model_kde = scipy.stats.gaussian_kde(sw_models.loc[spacecraft.data.index, ('ensemble', 'u_mag')].to_numpy('float64'))
        
#         fig, ax = plt.subplots()
        
#         ax.stairs(data_hist, bins, linewidth=2, label='Data (Juno)')
#         ax.stairs(model_hist, bins, linewidth=2, label='MME')
#         ax.stairs(inferred_hist, bins, linewidth=2, label='Louis+ 2023 (Inferred)')
#         ax.set(xscale='log')
        
#         # x = np.linspace(200,800,601)    
#         # ax.plot(x, data_kde.evaluate(x), linewidth=2, label='Data')
#         # ax.plot(x, model_kde.evaluate(x), linewidth=2, label='Model')
        
#         ax.legend()
#         plt.show()
        
#         breakpoint()


def compare_Pressures():
    import JunoPreprocessingRoutines as PR 
    import JoyBoundaryCoords as JBC
    sw_mme_filepath = '/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv'
    
    mp_crossing_data = PR.read_Louis2023_CrossingList(mp=True)
    
    bs_crossing_data_Louis2023 = PR.read_Louis2023_CrossingList(bs=True)
    
    bs_crossing_data_Kurth = PR.read_Kurth_CrossingList(bs=True)
    bs_crossing_data_Kurth = bs_crossing_data_Kurth[bs_crossing_data_Kurth['direction'].notna()]
    
    bs_crossing_data = pd.concat([bs_crossing_data_Louis2023, bs_crossing_data_Kurth], axis=0, join="outer")
    
    #   Only pass the datetime index, the direction of the crossing, and any notes
    hourly_mp_crossing_data = PR.make_HourlyCrossingList(mp_crossing_data)
    hourly_mp_crossing_data['p_dyn'] = JBC.find_JoyPressures(*hourly_mp_crossing_data[['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T, 'MP')
    
    hourly_bs_crossing_data = PR.make_HourlyCrossingList(bs_crossing_data)
    hourly_bs_crossing_data['p_dyn'] = JBC.find_JoyPressures(*hourly_bs_crossing_data[['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T, 'BS')

    #   Load the solar wind data and select the ensesmble
    sw_models = MMESH_reader.fromFile(sw_mme_filepath)
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)


    fig, ax = plt.subplots(figsize=(8, 3), nrows=1)
    plt.subplots_adjust(bottom=0.15, left=0.075, right=1-0.00625, top=0.85, 
                        wspace=0.025)
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2, which='major')
    ax.tick_params(width=1.5, which='minor')
    
    mp_color_dict = {'Louis+ (2023)': 'xkcd:raspberry'}
    colors = [mp_color_dict[origin[0]] for origin in hourly_mp_crossing_data['origin'].to_numpy()]
    ax.scatter(hourly_mp_crossing_data.index, hourly_mp_crossing_data['p_dyn'],
               marker='o', c=colors, s=18,
               label=r'Magnetopause Crossing (Louis+ (2023))')
    
    bs_color_dict = {'Louis+ (2023)': 'xkcd:lavender',
                     'Kurth, p.c.': 'xkcd:blue purple'}
    colors = [bs_color_dict[origin[0]] for origin in hourly_bs_crossing_data['origin'].to_numpy()]
    
    for key in bs_color_dict.keys():
        indx = (np.array(colors) == bs_color_dict[key])
        ax.scatter(hourly_bs_crossing_data.index.to_numpy()[indx], hourly_bs_crossing_data['p_dyn'].to_numpy()[indx], 
                   marker='x', c=np.array(colors)[indx], s=18,
                   label='Bow Shock Crossing ({})'.format(key))
    
    ax.plot(sw_mme.index, sw_mme['p_dyn'], 
            color='C4', linewidth=2, label='MME')
    ax.fill_between(sw_mme.index,
                    sw_mme['p_dyn'] - sw_mme['p_dyn_neg_unc'],
                    sw_mme['p_dyn'] + sw_mme['p_dyn_pos_unc'],
                    color='C4', alpha=0.5)
    
    ax.set(yscale='log', 
           xlabel = 'Year', xlim = [dt.datetime(2016, 7, 4), dt.datetime(2024, 3, 1)])
    
    ax.legend(ncols=4, loc='lower right', bbox_to_anchor = (1.0, 1.0), fancybox=True)
    ax.set_ylabel('Pressure [nPa]')
    
    plt.savefig('/Users/mrutala/projects/MMESH/testing/figures/poster07_JoyComparison.png',
                dpi=800, transparent=True)
    
    plt.show()


def runner(parameter_distributions=False):
    
    which_boundary = 'MP'
    sw_mme_filepath = '/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv'
    
    import JunoPreprocessingRoutines as PR 
    
    #   
    if which_boundary == 'MP':
        crossing_data = PR.read_Louis2023_CrossingList(mp=True)
    if which_boundary == 'BS':
        bs_crossing_data_Louis2023 = PR.read_Louis2023_CrossingList(bs=True)
        
        bs_crossing_data_Kurth = PR.read_Kurth_CrossingList(bs=True)
        bs_crossing_data_Kurth = bs_crossing_data_Kurth[bs_crossing_data_Kurth['direction'].notna()]
        
        crossing_data = pd.concat([bs_crossing_data_Louis2023, bs_crossing_data_Kurth], axis=0, join="outer")
    
    #   Only pass the datetime index, the direction of the crossing, and any notes
    hourly_crossing_data = PR.make_HourlyCrossingList(crossing_data.loc[:, ['direction', 'notes']])
    
    #   Load the solar wind data and select the ensemble
    sw_models = MMESH_reader.fromFile(sw_mme_filepath)
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    
    #   Select boundary model
    boundary_models = {'Shuelike': {'model': boundaries.Shuelike, 
                                    'params': ['r0', 'r1', 'a0', 'a1'], 
                                    'guess': [60, -0.15, 0.5, 1.0]},
                       'Shuelike_UniformPressureExponent': {'model': boundaries.Shuelike_UniformPressureExponent, 
                                                       'params': ['r0', 'r1', 'a0', 'a1'], 
                                                       'guess': [60, -0.15, 0.5, 1.0]},
                       'Shuelike_NonuniformPressureExponent': {'model': boundaries.Shuelike_NonuniformPressureExponent, 
                                                       'params': ['r0', 'r1', 'a0', 'a1', 'a2'], 
                                                       'guess': [60, -0.15, 0.5, 1.0, 1.0]},
                       'Shuelike_AsymmetryCase1': {'model': boundaries.Shuelike_AsymmetryCase1, 
                                                       'params': ['r0', 'r1', 'r2', 'r3', 'a0', 'a1'], 
                                                       'guess': [60, -0.15, 5, 5, 0.5, 1.0]},
                       'Shuelike_Asymmetric': {'model': boundaries.Shuelike_Asymmetric, 
                                                       'params': ['r0', 'r1', 'r2', 'a0', 'a1', 'a2'], 
                                                       'guess': [60, -0.15, 1, 0.5, 1.0, 1.0]},
                       'Joylike': {'model': boundaries.Joylike, 
                                   'params': ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1', 'F0', 'F1'], 
                                   'guess': [-0.134, 0.488, -0.581, -0.225, -0.186, -0.016, -0.014, 0.096, -0.814,  -0.811, -0.050, 0.168]}
                       }
    #boundary_model = boundary_models['Shuelike']
    boundary_model = boundary_models['Shuelike_AsymmetryCase1']
    
    # Here: add a loop for MC and a function which perturbs the variables
    n_mc = 1000
    stats = {'parameters': [], 
             'rmsd': [],
             'mae': [],
             'joy_rmsd': [],
             'joy_mae': []}
    for i in range(n_mc):
        
        p_x = perturb_Gaussian(*hourly_crossing_data.loc[:, ['x_JSS', 'x_unc_JSS']].to_numpy().T)
        p_y = perturb_Gaussian(*hourly_crossing_data.loc[:, ['y_JSS', 'y_unc_JSS']].to_numpy().T)
        p_z = perturb_Gaussian(*hourly_crossing_data.loc[:, ['z_JSS', 'z_unc_JSS']].to_numpy().T)
        
        r, t, p = boundaries.convert_CartesianToSphericalSolar(p_x, p_y, p_z)
        
        perturbed_hcd = hourly_crossing_data.loc[:, ['direction', 'notes']]
        perturbed_hcd = perturbed_hcd.assign(**{'x': p_x, 'y': p_y, 'z': p_z, 'abs_z': np.abs(p_z), 
                                                'r': r, 't': t, 'p': p})
        
        p_p_dyn = perturb_Placeholder(*sw_mme.loc[:, ['p_dyn', 'p_dyn_neg_unc', 'p_dyn_pos_unc']].to_numpy().T)
        perturbed_sw_mme = pd.DataFrame({'p_dyn': p_p_dyn}, index=sw_mme.index)
        
        res = fit_Boundary_withODR(perturbed_hcd, perturbed_sw_mme, boundary_model['model'], boundary_model['guess'])
        
        stats['parameters'].append(res['parameters'])
        
        #   Next, calculate the fit
        stats['rmsd'].append(res['rmsd'])
        stats['mae'].append(res['mae'])
        
        #   Use these perturbed values to compare to Joy+ 2002
        #   This bit might be removed if I ever get Joy+ working well
        joy_z = JBC.find_JoyBoundaries(perturbed_sw_mme.loc[hourly_crossing_data.index, 'p_dyn'].to_numpy(), 
                                       boundary=which_boundary, x=p_x, y=p_y)
        abs_z = np.abs(p_z)
        
        valid_joy_indx = ~np.isnan(joy_z[0])
        stats['joy_rmsd'].append(np.sqrt((1/len(abs_z[valid_joy_indx])) * np.sum((abs_z[valid_joy_indx] - joy_z[0][valid_joy_indx])**2)))
        stats['joy_mae'].append(np.mean(np.abs(abs_z[valid_joy_indx] - joy_z[0][valid_joy_indx])))
    
    parameters = np.array(stats['parameters'])
    
    plot_ParameterDistributions(parameters.T, names = boundary_model['params'])
    
    #   Quick plot of RMSD, MAE
    fig, axs = plt.subplots(ncols=2, figsize=(6,4))
    
    axs[0].hist(stats['rmsd'], bins=50, density=True, label='Shuelike')
    axs[0].hist(stats['joy_rmsd'], bins=50, density=True, label='Joy+ (2002)')
    axs[0].set(xlabel = r'Root-Mean-Square Error [$R_J$]')
    axs[0].legend(loc='upper right')
    
    axs[1].hist(stats['mae'], bins=50, density=True, label='Shuelike')
    axs[1].hist(stats['joy_mae'], bins=50, density=True, label='Joy+ (2002)')
    axs[1].set(xlabel = r'Mean-Absolute Error [$R_J$]')
    axs[1].legend(loc='upper right')
    
    fig.supylabel('Probability Density')
    
    plt.show()
    
    #breakpoint()
    #   Plotting pretty pictures
    #   It would be handy to have a dataframe with the coords in all the different frames
    #   Then we could query which ones we need, then plug them in...
    
    median_parameters = np.median(parameters.T, 1)
    
    #   Generate x, y, z coords, then convert to spherical, then plug in
    # x = np.linspace(-200, 200, 1000)
    # y = np.linspace(-200, 200, 1000)
    # z = np.zeros(1000)
    # p_dyn = np.zeros(1000) + 0.05
    
    # r, t, p = boundaries.convert_CartesianToSphericalSolar(x, y, z)
    
# =============================================================================
#     Write some functions which return the sets of coordinates needed to make
#   a folding plot-- i.e. y-x, y-z, x-z depending on inputs
#   e.g. get_SphericalSolarCoords()
#   should return (t, p), (t, p), (t, p)
# =============================================================================
    
    nsamples = 1000
    half_nsamples = int(nsamples/2)
    dtor = np.pi/180
    
    rtp_coords_dict = {}
    #   y-x plane: z=0
    t = np.append(np.linspace(180, 0, half_nsamples), np.linspace(0, 180, half_nsamples)) * dtor
    p = np.append(np.zeros(half_nsamples) + 90, np.zeros(half_nsamples) + 270) * dtor
    rtp_coords_dict['yx'] = (t, p)
    
    #   y-z plane: x=0
    t = (np.zeros(nsamples) + 90) * dtor
    p = np.linspace(0, 360, nsamples) * dtor
    rtp_coords_dict['yz'] = (t, p)
    
    #   x-z plane: y=0
    t = np.append(np.linspace(180, 0, half_nsamples), np.linspace(0, 180, half_nsamples)) * dtor
    p = np.append(np.zeros(half_nsamples) + 0, np.zeros(half_nsamples) + 180) * dtor
    rtp_coords_dict['xz'] = (t, p)
    
    test_pressure_cases = {'1': {'p_dyn': 0.005,
                                 'linestyle': ':',
                                 'color': 'black'},
                           '2': {'p_dyn': 0.03,
                                 'linestyle': '-',
                                 'color': 'black'},
                           '3': {'p_dyn': 0.3,
                                 'linestyle': '--',
                                 'color': 'black'}
                           }
    
    fig, axd = plt.subplot_mosaic(
        [['yx', '3d', 'cb'],
         ['yz', 'xz', 'cb']],
        figsize=(6,5), width_ratios = [1, 1, 0.1],
        per_subplot_kw = {'3d': {'projection': '3d'}})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.98, 
                        wspace=0.25, hspace=0.25)
    
    for ax_name in ['yx', 'yz', 'xz']:
        for case_name, case_info in test_pressure_cases.items():
            
            p_dyn = np.zeros(nsamples) + case_info['p_dyn']
            t, p = rtp_coords_dict[ax_name]
            r = boundary_model['model'](median_parameters, [t, p, p_dyn])
            x, y, z = boundaries.convert_SphericalSolarToCartesian(r, t, p)
            xyz_coords_dict = {'x':x, 'y':y, 'z':z}
            
            axd[ax_name].plot(xyz_coords_dict[ax_name[0]], 
                              xyz_coords_dict[ax_name[1]], 
                              linestyle = case_info['linestyle'],
                              color = case_info['color'],
                              linewidth = 1,
                              label=r'$p_{dyn}=$'+'{:.2f}'.format(case_info['p_dyn']))
        
        axd[ax_name].set(aspect='equal')
    
    #   Get residuals
    xyz_data = hourly_crossing_data[['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T
    rtp_data = boundaries.convert_CartesianToSphericalSolar(*xyz_data)
    p_dyn_data = sw_mme.loc[hourly_crossing_data.index]['p_dyn'].to_numpy()
    
    r_estimate = boundary_model['model'](median_parameters, 
                                         [rtp_data[1], rtp_data[2], p_dyn_data])
    residuals = rtp_data[0] - r_estimate
          
    
    
    import matplotlib as mpl
    
    bounds = np.linspace(-45,45,10)
    norm = mpl.colors.BoundaryNorm(bounds,10)
    base_cmap = plt.get_cmap('coolwarm_r')
    new_cmap_colors = []
    for i in np.linspace(0,1,10-1):
        new_cmap_colors.append(base_cmap(i))
        
    new_cmap = mpl.colors.ListedColormap(new_cmap_colors)
        
    #   Color according to 3rd dimension
    axd['yx'].scatter(hourly_crossing_data['y_JSS'], hourly_crossing_data['x_JSS'],
                      c = hourly_crossing_data['z_JSS'], cmap = new_cmap, norm = norm,
                      edgecolors = 'black', linewidth=0.5, s = 12)
    axd['yz'].scatter(hourly_crossing_data['y_JSS'], hourly_crossing_data['z_JSS'],
                      c = hourly_crossing_data['x_JSS'], cmap = new_cmap, norm = norm,
                      edgecolors = 'black', linewidth=0.5, s = 12)
    axd['xz'].scatter(hourly_crossing_data['x_JSS'], hourly_crossing_data['z_JSS'],
                      c = hourly_crossing_data['y_JSS'], cmap = new_cmap, norm = norm,
                      edgecolors = 'black', linewidth=0.5, s = 12)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = new_cmap), cax=axd['cb'])
    cb.ax.set_yticklabels([r'{:.1f}'.format(num) for num in bounds])
    axd['cb'].set_ylabel(r'Distance from plane [$R_J$]')
    
    
    # #   Color according to residuals
    # norm = norm = mpl.colors.CenteredNorm(halfrange=40)
    # axd['yx'].scatter(hourly_crossing_data['y_JSS'], hourly_crossing_data['x_JSS'],
    #                   c = residuals, cmap = 'coolwarm_r', norm = norm,
    #                   edgecolors = 'black', linewidth=0.5, s = 12)
    # axd['yz'].scatter(hourly_crossing_data['y_JSS'], hourly_crossing_data['z_JSS'],
    #                   c = residuals, cmap = 'coolwarm_r', norm = norm,
    #                   edgecolors = 'black', linewidth=0.5, s = 12)
    # axd['xz'].scatter(hourly_crossing_data['x_JSS'], hourly_crossing_data['z_JSS'],
    #                   c = residuals, cmap = 'coolwarm_r', norm = norm,
    #                   edgecolors = 'black', linewidth=0.5, s = 12)
    # cb = plt.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = 'coolwarm_r'), cax=axd['cb'])
    # axd['cb'].set_ylabel(r'Residuals in Radial Distance [$R_J$]')
    
    axd['yx'].set(xlim = (-125, 125), 
                  ylabel = r'$x_{JSS}/R_J$ (+ sunward)', ylim = (100, -150))
    axd['yz'].set(xlabel = r'$y_{JSS}/R_J$ (+ duskward)', xlim = (-125, 125), 
                  ylabel = r'$z_{JSS}/R_J$ (+ northward)', ylim = (-125, 125))
    axd['xz'].set(xlabel = r'$x_{JSS}/R_J$ (+ sunward)', xlim = (100, -150), 
                  ylim = (-125, 125))
    # axd['yx'].set(ylim = tuple(reversed(axd['xz'].get_xlim())))
    # axd['xz'].set(xlim = tuple(reversed(axd['xz'].get_xlim())))
    
    
    t = np.linspace(0, 180, 1000) * np.pi/180
    p = np.linspace(0, 360, 1000) * np.pi/180
    t_grid, p_grid = np.meshgrid(t, p)
    p_dyn_grid = np.zeros((1000,1000)) + test_pressure_cases['2']['p_dyn']
    r_grid = boundary_model['model'](median_parameters, [t_grid, p_grid, p_dyn_grid])
    x_grid, y_grid, z_grid = boundaries.convert_SphericalSolarToCartesian(r_grid, t_grid, p_grid)
    
    axd['3d'].view_init(elev=30, azim=225, roll=0)
    axd['3d'].set(xlabel = r'$x_{JSS}/R_J$ (+ sunward)', xlim = (100, -150), 
                  ylabel = r'$y_{JSS}/R_J$ (+ duskward)', ylim = (125, -125), 
                  zlabel = r'$z_{JSS}/R_J$ (+ northward)', zlim = (-125, 125))
    
    clip_indx = ((z_grid < 0) | (x_grid < -150))
    x_grid[clip_indx] = np.nan
    y_grid[clip_indx] = np.nan
    z_grid[clip_indx] = np.nan
    
    axd['3d'].plot_wireframe(x_grid, y_grid, z_grid, rstride=50, cstride=50,
                              linewidth=1.0, alpha=0.5, color='xkcd:indigo')
    # axd['3d'].plot_surface(x_grid, y_grid, z_grid,
    #                        linewidth=0.5, color='C3', alpha=0.5)
    
    # axd['3d'].scatter(hourly_crossing_data['x_JSS'], hourly_crossing_data['y_JSS'], hourly_crossing_data['z_JSS'],
    #                   color = 'C2',
    #                   edgecolors = 'black', linewidth=0.5, s = 12)
    
    axd['3d'].scatter(hourly_crossing_data['x_JSS'], hourly_crossing_data['y_JSS'], hourly_crossing_data['z_JSS'],
                      c = residuals, cmap='coolwarm_r', norm = norm,
                      edgecolors = 'black', linewidth=0.5, s = 12)
    
    
    plt.show()
    
    breakpoint()

    
    breakpoint()
    
    #   This is handy for 3D
    t = np.linspace(0, 180, 1000) * np.pi/180
    p = np.linspace(0, 360, 1000) * np.pi/180
    t_grid, p_grid = np.meshgrid(t, p)
    p_dyn_grid = np.zeros((1000,1000)) + 0.05
    
    
    dependent = boundary_model['model'](median_parameters, [t, p, p_dyn])
    
    
    return rmsd, mae

def plot_ParameterDistributions(parameters_arr, names = None):
    
    nplots = np.shape(parameters_arr)[0]
    
    nrows = int(np.floor(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots / nrows))
    
    fig, axs = plt.subplots(figsize = (6,4), nrows = nrows, ncols = ncols,
                            layout="constrained")
    
    for i, ax in enumerate(axs.flat):
        if i < nplots:
            ax.hist(parameters_arr[i], bins=50, density=True)
            
            ax.axvline(np.median(parameters_arr[i]), color='black', linestyle='--', linewidth=1.5)
            
            
            if names != None:
                ax.annotate('{} = {:.2f}'.format(names[i], np.median(parameters_arr[i])),
                            (0, 1), (1, -1), 
                            xycoords='axes fraction', textcoords='offset fontsize',
                            ha = 'left', va = 'top')
        else:
            ax.set_axis_off()
    
    fig.supylabel('Probability Density')
    plt.show()
    
    return

def perturb_Gaussian(variable, standard_deviation):
    rng = np.random.default_rng()
    perturbed_variable = rng.normal(variable, standard_deviation)
    return perturbed_variable

def perturb_Placeholder(variable, neg_unc, pos_unc):
    """
    This is a bit of NONPHYSICAL code to approximate the non-normal errors from the MME
    The MME *really* needs to changed to force a usable error distribution, even if its not perfect

    Parameters
    ----------
    variable : TYPE
        DESCRIPTION.
    neg_unc : TYPE
        DESCRIPTION.
    pos_unc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rng = np.random.default_rng()
    perturb = rng.normal(0, 0.3, len(variable))
    
    perturbed_variable = variable
    perturbed_variable[perturb < 0] += neg_unc[perturb < 0]
    perturbed_variable[perturb > 0] += pos_unc[perturb > 0]
    
    if (perturbed_variable < 0).any():
        #   This should NEVER hapen
        breakpoint()
    
    return perturbed_variable

def fit_Boundary(crossings, solarwind_model, function, initial_parameters):
    
    #   Add the model parameters as variables, 
    #   then take only times with both model and crossing
    variables = pd.concat([crossings, solarwind_model], axis=1, join="inner")
    
    #   Fetch the expected variables from the function being fit
    independent_variables, dependent_variables = function(variables=True)
    
    independents = variables.loc[:, independent_variables].to_numpy().T
    dependents = variables.loc[:, dependent_variables].to_numpy().T
    
    #   Begin the ODR
    model = odr.Model(function)
    
    data = odr.Data(independents, dependents) #, indies_sigma, dies_sigma)
    
    odr_fit = odr.ODR(data, model, beta0 = initial_parameters, maxit=100)
    
    odr_result = odr_fit.run()
    
    #
    parameters = odr_result.beta, odr_result.sd_beta
    
    return parameters

def fit_Boundary_withODR(crossings, solarwind_model, function, initial_parameters):
    
    #   Add the model parameters as variables, 
    #   then take only times with both model and crossing
    variables = pd.concat([crossings, solarwind_model], axis=1, join="inner")
    
    #   Fetch the expected variables from the function being fit
    independent_variables, dependent_variables = function(variables=True)
    
    independents = variables.loc[:, independent_variables].to_numpy().T
    dependents = variables.loc[:, dependent_variables].to_numpy().T
    
    #   Begin the ODR
    model = odr.Model(function)
    
    data = odr.Data(independents, dependents) #, indies_sigma, dies_sigma)
    
    odr_fit = odr.ODR(data, model, 
                      beta0 = initial_parameters,
                      maxit=100)
    
    odr_result = odr_fit.run()
    
    #   Get errors
    estimated_dependents = function(odr_result.beta, independents)
    rmsd = np.sqrt((1/len(dependents)) * np.sum((dependents - estimated_dependents)**2))
    mae = np.mean(np.abs(dependents - estimated_dependents))
    
    result = {'parameters': odr_result.beta,
              'parameters_stddev': odr_result.sd_beta,
              'rmsd': rmsd,
              'mae': mae}
   
    return result


    # fig, ax = plt.subplots()

    # test_x = np.linspace(-200,200,1000)*1/120

    # for x_value, y_value, p_dyn_value in indies.T:
        
    #     test_y = np.zeros(1000) + (y_value)
    #     test_p_dyn = np.zeros(1000) + p_dyn_value
        
    #     test_z = boundary_joy2002_4D(myoutput.beta, np.array([test_x, test_y, test_p_dyn]))

    #     ax.plot(test_x*120, test_z*120, color='xkcd:brick')
    #     ax.plot(test_x*120, -test_z*120, color='xkcd:brick') # , label='Juno-MMESH Fit Bow Shock'

    #     original_z = boundary_joy2002_4D(boundary_joy2002_initial, np.array([test_x, test_y, test_p_dyn]))

    #     ax.plot(test_x*120, original_z*120, color='black', linestyle=':')
    #     ax.plot(test_x*120, -original_z*120, color='black', linestyle=':')#, label='Joy+ 2002 Bow Shock')

    # ax.set(xlabel=r'$X_{JSS}$ (+ toward Sun)', ylabel=r'$Z_{JSS}$ (+ toward Jovian Rotational North')

    # ax.scatter(indies[0]*120, dies*120, marker='o', s=24, label='Juno Crossings', color='xkcd:cornflower')
    # ax.annotate(r'$Y_{JSS} = -107 R_J$ Plane', (0,1), (1,-1), 
    #             xycoords='axes fraction', textcoords='offset fontsize')

    # ax.legend()

    # plt.show()
    
    

# #   The first bowshock encounter in this list is during cruise, which is not currently present in the MMESH outputs
# sw_models = read_MMESH_Outputs()
# bowshock_encounters = read_Louis2023_CrossingList()

# #   Assign perijove labels to each bowshock
# #   PJ00 starts at PJ00 and ends at PJ01
# perijove_dict = {'PJ00': (dt.datetime(2016, 7, 5, 2,47,32), dt.datetime(2016, 8,27,12,50,44)),
#                  'PJ01': (dt.datetime(2016, 8,27,12,50,44), dt.datetime(2016,10,19,18,10,54)),
#                  'PJ02': (dt.datetime(2016,10,19,18,10,54), dt.datetime(2016,12,11,17, 3,41)),
#                  'PJ03': (dt.datetime(2016,12,11,17, 3,41), dt.datetime(2017, 2, 2,12,57, 9)),
#                  'PJ04': (dt.datetime(2017, 2, 2,12,57, 9), dt.datetime(2017, 3,27, 8,51,52))}
# bowshock_encounters['perijove'] = 'none'
# for perijove_label, dates in perijove_dict.items():
#     bowshock_encounters.loc[dates[0]:dates[1], 'perijove'] = perijove_label

# #   Loop over perijoves
# for perijove in set(bowshock_encounters['perijove'].values):
    
#     bs_subset = bowshock_encounters.loc[bowshock_encounters['perijove'] == perijove]
#     direction_colors = np.array(['xkcd:aqua' if row['direction'] == 'out' else 'xkcd:salmon' 
#                                  for index, row in bs_subset.iterrows()])
#     direction_markers = np.array(['x' if row['direction'] == 'out' else 'o'
#                                   for index, row in bs_subset.iterrows()])
    
#     #   Quick function to round datetimes to the nearest hour
#     def nearest_hour(datetimes):
#         result = [t.replace(minute=0, second=0, microsecond=0) + 
#                   dt.timedelta(hours=t.minute//30) for t in datetimes]
#         return np.array(result)
    
#     padding = dt.timedelta(hours=24*4)
#     datetime_range = nearest_hour(bs_subset.index[[0,-1]].to_pydatetime())
#     datetime_range = datetime_range + np.array([-padding, padding])
    
#     extended_datetime_range = datetime_range + np.array([-4*padding, 4*padding])
    
#     datetimes = np.arange(*datetime_range, dt.timedelta(hours=1)).astype(dt.datetime)
#     extended_datetimes = np.arange(*extended_datetime_range, dt.timedelta(hours=1)).astype(dt.datetime)
    
#     fig, axd = plt.subplot_mosaic(
#         """
#         abc
#         ddd
#         eee
#         """,
#         layout="constrained", figsize=(6,6))
    
#     #   Juno global positiong from SPICE
#     with spice.KernelPool('/Users/mrutala/SPICE/juno/metakernel_juno.txt'):
#         ets = spice.datetime2et(extended_datetimes)
#         extended_juno_pos, lts = spice.spkpos('Juno', ets, 
#                                      'Juno_JSS', 'LT', 'Jupiter')
#         extended_juno_pos = extended_juno_pos.T / R_J
#         juno_pos = extended_juno_pos[:, (extended_datetimes >= datetime_range[0]) & (extended_datetimes <= datetime_range[1])]
        
#         crossing_pos, lts = spice.spkpos('Juno', spice.datetime2et(bs_subset.index.to_pydatetime()), 
#                                          'Juno_JSS', 'LT', 'Jupiter')
#         crossing_pos = crossing_pos.T / R_J
    
#     #   Start plotting
#     #   X-Y plane 
#     for marker in set(direction_markers):
#         mask = np.array(direction_markers) == marker
#         axd['a'].scatter(crossing_pos[0,mask], crossing_pos[1,mask],
#                          marker=marker, c=direction_colors[mask], s=32, linewidth=1)
#     axd['a'].scatter(0,0,marker='*', color='xkcd:peach')
    
#     axd['a'].plot(extended_juno_pos[0], extended_juno_pos[1], 
#                   color='black', linewidth=0.5)
#     axd['a'].plot(juno_pos[0], juno_pos[1], 
#                   color='xkcd:blue', linewidth=1)
    
#     x_bs = np.arange(-150,150.1,.1)
#     y_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=x_bs, z=np.mean(bs_subset['z_JSS']))
#     axd['a'].plot(x_bs, y_bs[0], color='gray', linewidth=1)
#     axd['a'].plot(x_bs, y_bs[1], color='gray', linewidth=1)
#     axd['a'].annotate('z = {0:.1f}'.format(np.mean(bs_subset['z_JSS'])), 
#                       (1,1), (-1,-1), 
#                       xycoords='axes fraction', textcoords='offset fontsize',
#                       ha='right', va='center')
    
#     axd['a'].set_aspect(1)
#     axd['a'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Y_{JSS}$')
    
#     #   X-Z plane
#     axd['b'].scatter(crossing_pos[0],crossing_pos[2],
#                      marker='x', c=direction_colors, s=32, linewidth=0.5)
#     axd['b'].scatter(0,0,marker='*', color='xkcd:peach')
#     xlim, ylim = axd['b'].get_xlim(), axd['b'].get_ylim()
    
#     axd['b'].plot(extended_juno_pos[0], extended_juno_pos[2], 
#                   color='black', linewidth=0.5)
#     axd['b'].plot(juno_pos[0], juno_pos[2], 
#                  color='xkcd:blue', linewidth=1)
    
#     x_bs = np.arange(-150,150.1,.1)
#     z_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=x_bs, y=np.mean(bs_subset['y_JSS']))
#     axd['b'].plot(x_bs, z_bs[0], color='gray', linewidth=1)
#     axd['b'].plot(x_bs, z_bs[1], color='gray', linewidth=1)
#     axd['b'].annotate('y = {0:.1f}'.format(np.mean(bs_subset['y_JSS'])), 
#                       (1,1), (-1,-1), 
#                       xycoords='axes fraction', textcoords='offset fontsize',
#                       ha='right', va='center')
    
#     #axd['b'].set(xlim=xlim, ylim=ylim)
#     axd['b'].set_aspect(1)
#     axd['b'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Z_{JSS}$')
    
#     #   Y-Z Plane
#     axd['c'].scatter(crossing_pos[1],crossing_pos[2],
#                      marker='x', c=direction_colors, s=32, linewidth=0.5)
#     axd['c'].scatter(0,0,marker='*', color='xkcd:peach')
    
#     axd['c'].plot(extended_juno_pos[1], extended_juno_pos[2], 
#                   color='black', linewidth=1)
#     axd['c'].plot(juno_pos[1], juno_pos[2], 
#                   color='xkcd:blue', linewidth=2)
    
#     y_bs = np.arange(-150,150.1,.1)
#     z_bs = JBC.find_JoyBowShock(np.mean(bs_subset['p_dyn']), x=np.mean(bs_subset['x_JSS']), y=y_bs)
#     axd['c'].plot(y_bs, z_bs[0], color='gray', linewidth=1)
#     axd['c'].plot(y_bs, z_bs[1], color='gray', linewidth=1)
#     axd['c'].annotate('x = {0:.1f}'.format(np.mean(bs_subset['x_JSS'])), 
#                       (1,1), (-1,-1), 
#                       xycoords='axes fraction', textcoords='offset fontsize',
#                       ha='right', va='center')
    
#     axd['c'].set_aspect(1)
#     axd['c'].set(xlabel=r'$Y_{JSS}$', ylabel=r'$Z_{JSS}$')
    
#     #   MMESH
#     colors = ['xkcd:baby blue', 'xkcd:kelly green', 'xkcd:lavender', 'xkcd:navy']
#     sw_model_names = list(set(sw_models.columns.get_level_values(0)))
    
#     try:
#         for num, sw_model in enumerate(sw_model_names):
            
#             y = sw_models.loc[datetimes, (sw_model, 'p_dyn')]
#             y_upper = sw_models.loc[datetimes, (sw_model, 'p_dyn_pos_unc')]
#             y_lower = sw_models.loc[datetimes, (sw_model, 'p_dyn_neg_unc')]               
            
#             axd['d'].plot(y.index, y,
#                           color=colors[num], label=sw_model)
#             axd['d'].fill_between(y.index, y-y_lower, y+y_upper,
#                                   alpha=0.5, color=colors[num])
#             axd['d'].set_yscale('log')
            
            
#         axd['d'].legend()

#     except:
#         axd['d'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
#                           xycoords='axes fraction', textcoords='offset fontsize')
    
#     for marker in set(direction_markers):
#         mask = np.array(direction_markers) == marker
#         axd['d'].scatter(bs_subset.index[mask], bs_subset['p_dyn'].values[mask],
#                          marker=marker, c=direction_colors[mask], s=32, linewidth=1)
#     axd['d'].set(ylabel='Pressure [nPa]', xlabel='Date')
        
#     try:
#         for num, sw_model in enumerate(sw_model_names):
            
#             y = sw_models.loc[datetimes, (sw_model, 'u_mag')]
#             y_upper = sw_models.loc[datetimes, (sw_model, 'u_mag_pos_unc')]
#             y_lower = sw_models.loc[datetimes, (sw_model, 'u_mag_neg_unc')]               
            
#             axd['e'].plot(y.index, y,
#                           color=colors[num], label=sw_model)
#             axd['e'].fill_between(y.index, y-y_lower, y+y_upper,
#                                   alpha=0.5, color=colors[num])
#             axd['e'].set_yscale('log')
            
            
#         axd['e'].legend()

#     except:
#         axd['e'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
#                           xycoords='axes fraction', textcoords='offset fontsize')
#     axd['e'].set(ylabel='Flow Speed [km/s]', xlabel='Date')
    
#     plt.show()
#     print(datetimes[0], datetimes[-1])
    
# hourly_index = [t.to_datetime64() for t in nearest_hour(bowshock_encounters.index)]
# hourly_index = hourly_index[1:]  #  !!!! TEMPORARILY drop the first element, which we don't currently have pressure for
# bowshock_p_dyn = [sw_models.loc[indx, ('ensemble', 'p_dyn')] for indx in hourly_index]
# indies = np.array([bowshock_encounters['x_JSS'][1:]*1/120, 
#                    bowshock_encounters['y_JSS'][1:]*1/120, 
#                    bowshock_p_dyn])
# dies = bowshock_encounters['z_JSS'][1:]*1/120

# def boundary_joy2002(X, a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1):
#     x, y, p_dyn = X
#     a = a0 + a1*p_dyn**(-1/4)
#     b = b0 + b1*p_dyn**(-1/4)
#     c = c0 + c1*p_dyn**(-1/4)
#     d = d0 + d1*p_dyn
#     e = e0 + e1*p_dyn
#     f = f0 + f1*p_dyn
#     z = np.sqrt(a + b*x + c*x**2 + d*y + e*y**2 + f*x*y)
#     return z

# boundary_joy2002_initial = [-1.107, 1.591, -0.566,  -0.812, +0.048, -0.059, +0.077, -0.038, -0.874, -0.299, -0.055, 0.124] 

# popt, pcov = curve_fit(boundary_joy2002, indies, dies, p0=boundary_joy2002_initial)
# fig, axs = plt.subplots(nrows=2)
# p_dyn = np.logspace(-3, -0.3, 100)
# axs[0].plot(p_dyn**(-1/4), popt[0] + popt[1]*p_dyn**(-1/4))
# axs[0].plot(p_dyn**(-1/4), popt[2] + popt[3]*p_dyn**(-1/4))
# axs[0].plot(p_dyn**(-1/4), popt[4] + popt[5]*p_dyn**(-1/4))
# axs[1].plot(p_dyn, popt[6] + popt[7]*p_dyn)
# axs[1].plot(p_dyn, popt[8] + popt[9]*p_dyn)
# axs[1].plot(p_dyn, popt[10] + popt[11]*p_dyn)
# plt.show()

# fig, ax = plt.subplots()
# test_x = np.linspace(-100,100,1000)*1/120.
# test_y = np.zeros(1000)*1/120.
# test_p_dyn = np.zeros(1000) + 0.04

# ax.plot(test_x*120, boundary_joy2002(np.array([test_x, test_y, test_p_dyn]), *popt)*120, color='red')
# ax.plot(test_x*120, boundary_joy2002(np.array([test_x, test_y, test_p_dyn]), *boundary_joy2002_initial)*120, color='black')
# ax.plot(test_x*120, JBC.find_JoyBowShock(0.04, x=test_x*120, y=0)[1], color='blue', linestyle=':')
# ax.scatter(indies[0]*120, dies*120, color='xkcd:cyan', marker='x')
# plt.show()

# fig, ax = plt.subplots()
# test_x = np.zeros(1000) - 15*1/120. 
# test_y = np.linspace(-150,150,1000)*1/120.
# test_p_dyn = np.zeros(1000) + 0.045
# test_z = boundary_joy2002(np.array([test_x, test_y, test_p_dyn]), *popt)

# ax.plot(test_y*120, test_z*120, color='red')
# ax.plot(test_y*120, boundary_joy2002(np.array([test_x, test_y, test_p_dyn]), *boundary_joy2002_initial)*120, color='black')
# ax.plot(test_y*120, JBC.find_JoyBowShock(0.04, y=test_y*120, x=0)[1], color='blue', linestyle=':')
# ax.scatter(indies[1]*120, dies*120, color='xkcd:cyan', marker='x')
# plt.show()


# test_x = np.linspace(-500,500,1000)*1/120. 
# test_y = np.linspace(-500,500,1000)*1/120.
# test_p_dyn = np.zeros(1000) + 0.045

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, Y = np.meshgrid(test_x, test_y)

# Z = np.zeros((len(test_x), len(test_y)))
# for i in range(len(test_x)):
#     Z[:,i] = boundary_joy2002(np.array([X[:,i], Y[:,i], test_p_dyn]), *boundary_joy2002_initial)
# ax.plot_surface(X*120, Y*120, Z*120, 
#                 alpha=0.5)

# Z = np.zeros((len(test_x), len(test_y)))
# for i in range(len(test_x)):
#     Z[:,i] = boundary_joy2002(np.array([X[:,i], Y[:,i], test_p_dyn]), *popt)
# ax.plot_surface(X*120, Y*120, Z*120, 
#                 alpha=0.5)    
    
# ax.set(xlabel=r'$X_{JSS}$', ylabel=r'$Y_{JSS}$', zlabel=r'$Z_{JSS}$')

# plt.show()


# def boundary_joy2002_3D(X, a, b, c, d, e, f):
#     x, y, p_dyn = X
#     z = np.sqrt(a + b*x + c*x**2 + d*y + e*y**2 + f*x*y)
#     return z

# #   Try binning up the pressures:
# p_dyn_bin_edges = np.arange(0.01, 0.12, 0.01)
# p_dyn_labels = np.digitize(indies[2], p_dyn_bin_edges)
# pressure, coeffs = [], []
# for p_dyn_label, p_dyn_label_count in zip(*np.unique(p_dyn_labels, return_counts=True)):
#     print(p_dyn_label_count)
#     if p_dyn_label_count >= 6:
#         target_p_dyn = p_dyn_bin_edges[p_dyn_label]
#         pressure.append(target_p_dyn + 0.01/2)
        
#         indies_subset = indies.T[p_dyn_labels == p_dyn_label,:].T
#         dies_subset = dies[p_dyn_labels == p_dyn_label]
        
#         popt, pcov = curve_fit(boundary_joy2002_3D, indies_subset, dies_subset)
#         coeffs.append(popt)
        
# pressure = np.array(pressure)
# coeffs = np.array(coeffs).T

# fig, axs = plt.subplots(nrows=2)
# axs[0].scatter(pressure**(-1/4), coeffs[0], label='a')
# axs[0].scatter(pressure**(-1/4), coeffs[1], label='b')
# axs[0].scatter(pressure**(-1/4), coeffs[2], label='c')
# axs[0].set(ylim=[-30,30])
# axs[0].legend()

# axs[1].scatter(pressure, coeffs[3], label='d')
# axs[1].scatter(pressure, coeffs[4], label='e')
# axs[1].scatter(pressure, coeffs[5], label='f')
# axs[1].set(ylim=[-1,5])
# axs[1].legend()

# plt.show()
    

   
# #   Try binning up the pressures in a way that accounts for errors? KDE? 
   
#     #breakpoint()
    
    
# #   ODR
# def boundary_joy2002_4D(p, X):
#     x, y, p_dyn = X
#     a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1 = p
#     a = a0 + a1*p_dyn**(-1/4)
#     b = b0 + b1*p_dyn**(-1/4)
#     c = c0 + c1*p_dyn**(-1/4)
#     d = d0 + d1*p_dyn
#     e = e0 + e1*p_dyn
#     f = f0 + f1*p_dyn
#     z = np.sqrt(a + b*x + c*x**2 + d*y + e*y**2 + f*x*y)
#     return z
# model = odr.Model(boundary_joy2002_4D)

# hourly_index = [t.to_datetime64() for t in nearest_hour(bowshock_encounters.index)]
# hourly_index = hourly_index[1:]  #  !!!! TEMPORARILY drop the first element, which we don't currently have pressure for
# bowshock_p_dyn = [sw_models.loc[indx, ('Tao', 'p_dyn')] for indx in hourly_index]
# indies = np.array([bowshock_encounters['x_JSS'][1:]*1/120, 
#                    bowshock_encounters['y_JSS'][1:]*1/120, 
#                    bowshock_p_dyn])
# assumed_spice_error = 0.01*1/120.  #  R_J, based on rounding
# bowshock_p_dyn_sigma = [(sw_models.loc[indx, ('Tao', 'p_dyn_pos_unc')] + 
#                          sw_models.loc[indx, ('Tao', 'p_dyn_neg_unc')])/2. 
#                         for indx in hourly_index]
# indies_sigma = np.array([np.zeros(len(indies[0])) + assumed_spice_error,
#                          np.zeros(len(indies[1])) + assumed_spice_error,
#                          bowshock_p_dyn_sigma])

# dies = bowshock_encounters['z_JSS'][1:]*1/120
# dies_sigma = np.zeros(len(dies)) + assumed_spice_error

# data = odr.RealData(indies, dies, indies_sigma, dies_sigma)
# myodr = odr.ODR(data, model, beta0 = boundary_joy2002_initial)
# myoutput = myodr.run()
# #myoutput.pprint()


# fig, ax = plt.subplots()

# test_x = np.linspace(-200,200,1000)*1/120

# for x_value, y_value, p_dyn_value in indies.T:
    
#     test_y = np.zeros(1000) + (y_value)
#     test_p_dyn = np.zeros(1000) + p_dyn_value
    
#     test_z = boundary_joy2002_4D(myoutput.beta, np.array([test_x, test_y, test_p_dyn]))

#     ax.plot(test_x*120, test_z*120, color='xkcd:brick')
#     ax.plot(test_x*120, -test_z*120, color='xkcd:brick') # , label='Juno-MMESH Fit Bow Shock'

#     original_z = boundary_joy2002_4D(boundary_joy2002_initial, np.array([test_x, test_y, test_p_dyn]))

#     ax.plot(test_x*120, original_z*120, color='black', linestyle=':')
#     ax.plot(test_x*120, -original_z*120, color='black', linestyle=':')#, label='Joy+ 2002 Bow Shock')

# ax.set(xlabel=r'$X_{JSS}$ (+ toward Sun)', ylabel=r'$Z_{JSS}$ (+ toward Jovian Rotational North')

# ax.scatter(indies[0]*120, dies*120, marker='o', s=24, label='Juno Crossings', color='xkcd:cornflower')
# ax.annotate(r'$Y_{JSS} = -107 R_J$ Plane', (0,1), (1,-1), 
#             xycoords='axes fraction', textcoords='offset fontsize')

# ax.legend()

# plt.show()

# zs_odr, zs_orig = [], []
# zs_odr_off, zs_orig_off = [], []
# for x, y, p, z in np.array([*indies, dies]).T:
#     z_odr = boundary_joy2002_4D(myoutput.beta, np.array([x,y,p]))
#     z_orig = boundary_joy2002_4D(boundary_joy2002_initial, np.array([x,y,p]))
    
#     zs_odr.append(z_odr)
#     zs_orig.append(z_orig)
    
#     zs_odr_off.append(z_odr - abs(z))
#     zs_orig_off.append(z_orig - abs(z))

# z_off_odr = np.array(zs_odr - dies.values)
# z_off_orig = np.array(zs_orig - dies.values)

# fig, ax = plt.subplots(nrows=4, sharey=True)
# ax[0].scatter(indies[0]*120, z_off_odr*120, color='xkcd:red', label='New Fit')
# ax[0].scatter(indies[0]*120, z_off_orig*120, color='black', label='Joy+ 2002')

# ax[1].scatter(indies[1]*120, z_off_odr*120, color='xkcd:red', label='New Fit')
# ax[1].scatter(indies[1]*120, z_off_orig*120, color='black', label='Joy+ 2002')

# ax[2].scatter(indies[2], z_off_odr*120, color='xkcd:red', label='New Fit')
# ax[2].scatter(indies[2], z_off_orig*120, color='black', label='Joy+ 2002')

# ax[3].scatter(dies*120, z_off_odr*120, color='xkcd:red', label='New Fit')
# ax[3].scatter(dies*120, z_off_orig*120, color='black', label='Joy+ 2002')
    
# # for index, row in bowshock_encounters.iterrows():
    
# #     nearest_hour = row['datetime'].replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=row['datetime'].minute//30)
    
# #     delta = dt.timedelta(hours=24*5)
    
# #     datetimes = np.arange(row.index-delta, row.index+delta, dt.timedelta(hours=1)).astype(dt.datetime)
# #     datetimes_hourly = np.arange(nearest_hour-delta, nearest_hour+delta, dt.timedelta(hours=1)).astype(dt.datetime)
    
# #     fig, axd = plt.subplot_mosaic(
# #         """
# #         abc
# #         ddd
# #         """,
# #         layout="constrained",)
    
# #     with spice.KernelPool('/Users/mrutala/SPICE/juno/metakernel_juno.txt'):
        
# #         ets = spice.datetime2et(datetimes)
        
# #         pos, lts = spice.spkpos('Juno', ets, 'Juno_JSS', 'LT', 'Jupiter')
# #         pos = pos.T / RJ
        
# #         crossing, crossing_lt = spice.spkpos('Juno', spice.datetime2et(row['datetime']), 'Juno_JSS', 'LT', 'Jupiter')
# #         crossing = crossing.T / RJ
        
    
        
# #     #   X-Y plane 
# #     axd['a'].plot(pos[0], pos[1])
# #     axd['a'].scatter(crossing[0],crossing[1],marker='x', color='black')
# #     axd['a'].scatter(0,0,marker='*', color='xkcd:peach')
# #     axd['a'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Y_{JSS}$')
    
# #     x_bs = np.arange(-100,100.1,.1)
# #     y_bs = JBC.find_JoyBowShock(row['p_dyn'], x=x_bs, z=row['z_JSS'])
# #     axd['a'].plot(x_bs, y_bs[0], color='gray', linewidth=1)
# #     axd['a'].plot(x_bs, y_bs[1], color='gray', linewidth=1)
    
# #     #   X-Z plane
# #     axd['b'].plot(pos[0], pos[2])
# #     axd['b'].scatter(crossing[0],crossing[2],marker='x', color='black')
# #     axd['b'].scatter(0,0,marker='*', color='xkcd:peach')
# #     axd['b'].set(xlabel=r'$X_{JSS}$', ylabel=r'$Z_{JSS}$')
    
# #     x_bs = np.arange(-100,100.1,.1)
# #     z_bs = JBC.find_JoyBowShock(row['p_dyn'], x=x_bs, y=row['y_JSS'])
# #     axd['b'].plot(x_bs, z_bs[0], color='gray', linewidth=1)
# #     axd['b'].plot(x_bs, z_bs[1], color='gray', linewidth=1)
    
# #     #   Y-Z Plane
# #     axd['c'].plot(pos[1], pos[2])
# #     axd['c'].scatter(crossing[1],crossing[2],marker='x', color='black')
# #     axd['c'].scatter(0,0,marker='*', color='xkcd:peach')
# #     axd['c'].set(xlabel=r'$Y_{JSS}$', ylabel=r'$Z_{JSS}$')
    
# #     y_bs = np.arange(-100,100.1,.1)
# #     z_bs = JBC.find_JoyBowShock(row['p_dyn'], x=row['x_JSS'], y=y_bs)
# #     axd['c'].plot(y_bs, z_bs[0], color='gray', linewidth=1)
# #     axd['c'].plot(y_bs, z_bs[1], color='gray', linewidth=1)
    
# #     #   MMESH
# #     try:
# #         y = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn')]
# #         y_upper = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn_pos_unc')]
# #         y_lower = sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn_neg_unc')]               
        
# #         axd['d'].plot(sw_models.loc[datetimes_hourly, ('ensemble', 'p_dyn')])
# #         axd['d'].fill_between(y.index, y-y_lower, y+y_upper,
# #                               alpha=0.5)
# #         axd['d'].set_yscale('log')
# #         axd['d'].axvline(row['datetime'], color='red')
# #         axd['d'].axhline(row['p_dyn'], color='red', linestyle='--')
# #         axd['d'].set(ylabel='Pressure [nPa]', xlabel='Date')
# #     except:
# #         axd['d'].annotate('No MMESH output currently available!', (0,1), (1,-1), 
# #                           xycoords='axes fraction', textcoords='offset fontsize')
    
# #     #axd['e'].plot(sw_models.loc[datetimes_hourly, ('ensemble', 'u_mag')],)
    
# #     plt.show()
        
        

    