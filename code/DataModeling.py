#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:48:26 2024

@author: mrutala
"""

def analyze_RhoDistributions():
    import spiceypy as spice
    import numpy as np
    import matplotlib.pyplot as plt
    
    crossings_df = JPR.make_CombinedCrossingsList(boundary = 'MP')
    crossings_df = crossings_df.drop(['p_dyn', 'r_mp', 'r_bs'], axis='columns')
    
    rpl_crossings = boundaries.convert_CartesianToCylindricalSolar(*crossings_df.loc[:, ['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T)
    crossings_df[['rho', 'phi', 'ell']] = rpl_crossings.T
    
    #   Load the solar wind data, select the ensemble, and add it to the df
    sw_models = MMESH_reader.fromFile('/Users/mrutala/projects/JupiterBoundaries/mmesh_run/MMESH_atJupiter_20160301-20240301_withConstituentModels.csv')
    sw_mme = sw_models.xs('ensemble', axis='columns', level=0)
    hourly_crossings = JPR.get_HourlyFromDatetimes(crossings_df.index)
    crossings_df.loc[:, ['p_dyn_mmesh', 'p_dyn_nu_mmesh', 'p_dyn_pu_mmesh']] = sw_mme.loc[hourly_crossings, ['p_dyn', 'p_dyn_neg_unc', 'p_dyn_pos_unc']].to_numpy()
    
    #   Load SPICE kernels for positions of Juno
    spice.furnsh(JPR.get_paths()['PlanetaryMetakernel'].as_posix())
    spice.furnsh(JPR.get_paths()['JunoMetakernel'].as_posix())
    
    R_J = spice.bodvcd(599, 'RADII', 3)[1][1]
    
    #   Get all hourly trajectory info
    earliest_time = min(crossings_df.index)
    latest_time = max(crossings_df.index)
     
    datetimes = np.arange(pd.Timestamp(earliest_time).replace(minute=0, second=0, nanosecond=0),
                          pd.Timestamp(latest_time).replace(minute=0, second=0, nanosecond=0) + dt.timedelta(hours=1),
                          dt.timedelta(hours=1)).astype(dt.datetime)
     
    ets = spice.datetime2et(datetimes)   
    
    pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
    xyz = pos.T / R_J   
    
    spice.kclear()
    
    rpl_Juno = boundaries.convert_CartesianToCylindricalSolar(*xyz)
    Juno_df = pd.DataFrame(data = rpl_Juno.T, columns=['rho', 'phi', 'ell'], index = datetimes)
    
    #   Format the dataframe
    
    #   Boundaries in the elevation (z cylindrical axis + toward Sun)
    ell_bounds = np.array([np.linspace(-20, -120, 6), np.linspace(0, -100, 6)]).T
    
    for ell_bound in ell_bounds:
        
        #   Binary truth indices for bounds, dawn/dusk
        #   In this coordinate system, dawn is 0-pi and dusk is pi-2pi
        crossings_dawn_query = '({0} <= ell < {1}) & (phi > 0)'.format(*ell_bound)
        crossings_dusk_query = '({0} <= ell < {1}) & (phi < 0)'.format(*ell_bound)
        
        rho_bin_edges = np.arange(50, 130, 5)
        p_dyn_bin_edges = np.logspace(-2.5, -0.5, 21)
        
        #   Plot rho vs p_dyn for dawn and dusk hemisphere
        fig, axd = plt.subplot_mosaic([['dawn_scatter', 'dawn_p_dyn', '.', 'dusk_scatter', 'dusk_p_dyn'],
                                       ['dawn_rho', '.', '.', 'dusk_rho', '.']], 
                                        width_ratios = [3, 1, 0.5, 3, 1], height_ratios=[3, 1],
                                        figsize=[6,3])
        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.95, 
                            hspace=0, wspace=0)
        
        def errorscatter_rho_p_dyn(ax, df):
            ax.errorbar(df['rho'],
                        df['p_dyn_mmesh'],
                        yerr = df[['p_dyn_nu_mmesh', 'p_dyn_pu_mmesh']].to_numpy().T,
                        linestyle = 'None', marker = 'o', markersize = 4,
                        elinewidth = 1.0)   
        
        errorscatter_rho_p_dyn(axd['dawn_scatter'], crossings_df.query(crossings_dawn_query))
        errorscatter_rho_p_dyn(axd['dusk_scatter'], crossings_df.query(crossings_dusk_query))
        
        def histogram_rho(ax, df, **kwargs):
            histo, _ = np.histogram(df['rho'], 
                                    bins=rho_bin_edges,
                                    density=False)
            ax.stairs(histo, rho_bin_edges, 
                      linewidth=2, **kwargs)
            
        histogram_rho(axd['dawn_rho'], crossings_df.query(crossings_dawn_query), color='C1')
        #histogram_rho(axd['dawn_rho'], Juno_df.query(crossings_dawn_query), color='C2')
        
        histogram_rho(axd['dusk_rho'], crossings_df.query(crossings_dusk_query), color='C1')
        #histogram_rho(axd['dusk_rho'], Juno_df.query(crossings_dusk_query), color='C2')
        
        def histogram_p_dyn(ax, df):
            histo, _ = np.histogram(df['p_dyn_mmesh'],
                                    bins = p_dyn_bin_edges,
                                    density=True)
            ax.stairs(histo, p_dyn_bin_edges,
                      linewidth=2, color='C3',
                      orientation='horizontal')
            
        histogram_p_dyn(axd['dawn_p_dyn'], crossings_df.query(crossings_dawn_query))
        histogram_p_dyn(axd['dusk_p_dyn'], crossings_df.query(crossings_dusk_query))
        
        for ax_name in ['dawn_scatter', 'dawn_rho', 'dusk_scatter', 'dusk_rho']:
            axd[ax_name].set(xlim = (rho_bin_edges[0], rho_bin_edges[-1]))
        for ax_name in ['dawn_scatter', 'dawn_p_dyn', 'dusk_scatter', 'dusk_p_dyn']:
            axd[ax_name].set(ylim = (p_dyn_bin_edges[0], p_dyn_bin_edges[-1]), 
                             yscale = 'log')
        for ax_name in ['dawn_p_dyn', 'dusk_p_dyn']:
            axd[ax_name].set(yticklabels='')
            
        plt.show()
        #breakpoint()
        # hist_crossings, bin_edges = np.histogram(crossings_dawn_query.loc[:, ['rho']], 
        #                                          density=True, 
        #                                          bins=rho_bin_edges)
        # # hist_Juno, _ = np.histogram(rpl_Juno[0, indx_Juno],
        # #                          density=True, 
        # #                          bins=np.arange(30, 140, 10))
        
        # # hist_joint = hist_crossings/hist_Juno
        
        
        
        # fig, axs = plt.subplots(nrows=2, sharex=True)
        # axs[0].stairs(hist_crossings, bin_edges, linewidth=4, label='Crossings')
        # #axs[0].stairs(hist_Juno, bin_edges, linewidth=4, label='Juno')
        # axs[0].legend()
        
        # #axs[1].stairs(hist_joint, bin_edges)
        
        # plt.show()
        
        
        
        # # res = np.histogram2d(rpl_crossings[0, indx_crossings], sw_mme_sample['p_dyn'],
        # #                      [np.arange(30, 140, 10), np.arange(0.01, 0.11, 0.01)])
        
        # # plt.imshow(res[0], origin='lower', extent=[30, 130, 0.01, 0.1], aspect=1000)
        # # plt.show()
        
        # plt.scatter(rpl_crossings[0, indx_crossings], sw_mme_sample['p_dyn'])
        # plt.show()
    
    breakpoint()