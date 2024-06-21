#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:48:26 2024

@author: mrutala
"""
import numpy as np
import sys
import datetime
import pandas as pd
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

import pymc as pm
import arviz as az
from sklearn.neighbors import KernelDensity

import BoundaryModels as BM
import JunoPreprocessingRoutines as JPR

sys.path.append('/Users/mrutala/projects/MMESH/mmesh/')
import MMESH_reader

def analyze_RhoDistributions():

    
    crossings_df = JPR.make_CombinedCrossingsList(boundary = 'MP')
    crossings_df = crossings_df.drop(['p_dyn', 'r_mp', 'r_bs'], axis='columns')
    
    rpl_crossings = BM.convert_CartesianToCylindricalSolar(*crossings_df.loc[:, ['x_JSS', 'y_JSS', 'z_JSS']].to_numpy().T)
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
    
    #   Get hourly-resolution Juno ephemerides
    #   We'll need higher resolution later, but no sense storing all of that in memory
    earliest_time = min(crossings_df.index)
    latest_time = max(crossings_df.index)
     
    datetimes = np.arange(pd.Timestamp(earliest_time).replace(minute=0, second=0, nanosecond=0),
                          pd.Timestamp(latest_time).replace(minute=0, second=0, nanosecond=0) + datetime.timedelta(hours=1),
                          datetime.timedelta(hours=1)).astype(datetime.datetime)
     
    ets = spice.datetime2et(datetimes)   
    
    pos, lt = spice.spkpos('Juno', ets, 'Juno_JSS', 'None', 'Jupiter')
    xyz = pos.T / R_J   
    
    
    
    rpl_Juno = BM.convert_CartesianToCylindricalSolar(*xyz)
    Juno_df = pd.DataFrame(data = rpl_Juno.T, columns=['rho', 'phi', 'ell'], index = datetimes)
    
    #   Visualize the crossings in ell-rho space
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(*crossings_df.query('phi > 0').loc[:, ['ell', 'rho']].to_numpy().T,
               color='xkcd:gold',
               label = 'Dawn Crossings')
    ax.scatter(*crossings_df.query('phi < 0').loc[:, ['ell', 'rho']].to_numpy().T,
               color='xkcd:blue',
               label = 'Dusk Crossings')
    
    ax.legend()
    
    ax.add_patch(plt.Circle((0,0), 1.0, color='xkcd:peach'))
    ax.add_patch(patch.Wedge((0,0), 1.0, 90, 270, color='black'))
    ax.set(xlim = (-100, 40), xlabel = r'$x_{JSS}$ [$R_J$] (+sunward)',
           ylim = (-10, 120), ylabel = r'$\rho_{JSS}$ [$R_J$] (+outward)')
    plt.show()
    
    #   Boundaries in the elevation (z cylindrical axis, + toward Sun)
    ell_left, ell_right, ell_step = -120, 40, 20
    n_ell_bins = (ell_right - ell_left)/ell_step
    
    ell_bounds = np.array([np.arange(ell_left, ell_right, ell_step),
                           np.arange(ell_left, ell_right, ell_step) + ell_step]).T
    
    
    results = [{}] * int(n_ell_bins)
    for step, ell_bound in enumerate(ell_bounds):
        
        #   Set the bounds
        results[step]['ell_bound'] = ell_bound
        
        #   Add Juno ephemerides
        results[step]['Juno_eph'] = Juno_df.query("{0} < ell < {1}".format(*ell_bound))
        
        #   Perform a deeper search for maximum rho in Juno ephemerides
        
        breakpoint()
        
        
    ell_refs = np.arange(ell_left, ell_right, ell_step) + 0.5*ell_step

    
    breakpoint()
    results_dict = {}
    
    
    spice.kclear()
    
    d = {}
    for ell_ref, ell_bound in zip(ell_refs, ell_bounds):
        
        #   Binary truth indices for bounds, dawn/dusk
        #   In this coordinate system, dawn is 0-pi and dusk is pi-2pi
        crossings_dawn_query = '({0} <= ell < {1}) & (phi > 0)'.format(*ell_bound)
        crossings_dusk_query = '({0} <= ell < {1}) & (phi < 0)'.format(*ell_bound)
        
        dawn_df = crossings_df.query(crossings_dawn_query)
        dusk_df = crossings_df.query(crossings_dusk_query)
        
        rho_bin_edges = np.arange(50, 150, 5)
        p_dyn_bin_edges = np.logspace(-2.5, -0.5, 21)
        
        # #   Plot rho vs p_dyn for dawn and dusk hemisphere
        # fig, axd = plt.subplot_mosaic([['dawn_scatter', 'dawn_p_dyn', '.', 'dusk_scatter', 'dusk_p_dyn'],
        #                                ['dawn_rho', '.', '.', 'dusk_rho', '.']], 
        #                                 width_ratios = [3, 1, 0.5, 3, 1], height_ratios=[3, 1],
        #                                 figsize=[6,3])
        # plt.subplots_adjust(bottom=0.1, left=0.1, top=0.95, right=0.95, 
        #                     hspace=0, wspace=0)
        
        def errorscatter_rho_p_dyn(ax, df):
            ax.errorbar(df['rho'],
                        df['p_dyn_mmesh'],
                        yerr = df[['p_dyn_nu_mmesh', 'p_dyn_pu_mmesh']].to_numpy().T,
                        linestyle = 'None', marker = 'o', markersize = 4,
                        elinewidth = 1.0)   
        
        # errorscatter_rho_p_dyn(axd['dawn_scatter'], crossings_df.query(crossings_dawn_query))
        # errorscatter_rho_p_dyn(axd['dusk_scatter'], crossings_df.query(crossings_dusk_query))
        
        def histogram_rho(ax, df, density=True, **kwargs):
            histo, _ = np.histogram(df['rho'], 
                                    bins=rho_bin_edges,
                                    density=density)
            ax.stairs(histo, rho_bin_edges, 
                      linewidth=2, **kwargs)
            
        # histogram_rho(axd['dawn_rho'], crossings_df.query(crossings_dawn_query), color='C1')
        # #histogram_rho(axd['dawn_rho'], Juno_df.query(crossings_dawn_query), color='C2')
        
        # histogram_rho(axd['dusk_rho'], crossings_df.query(crossings_dusk_query), color='C1')
        # #histogram_rho(axd['dusk_rho'], Juno_df.query(crossings_dusk_query), color='C2')
        
        def histogram_p_dyn(ax, df):
            histo, _ = np.histogram(df['p_dyn_mmesh'],
                                    bins = p_dyn_bin_edges,
                                    density=True)
            ax.stairs(histo, p_dyn_bin_edges,
                      linewidth=2, color='C3',
                      orientation='horizontal')
            
        # histogram_p_dyn(axd['dawn_p_dyn'], crossings_df.query(crossings_dawn_query))
        # histogram_p_dyn(axd['dusk_p_dyn'], crossings_df.query(crossings_dusk_query))
        
        # for ax_name in ['dawn_scatter', 'dawn_rho', 'dusk_scatter', 'dusk_rho']:
        #     axd[ax_name].set(xlim = (rho_bin_edges[0], rho_bin_edges[-1]))
        # for ax_name in ['dawn_scatter', 'dawn_p_dyn', 'dusk_scatter', 'dusk_p_dyn']:
        #     axd[ax_name].set(ylim = (p_dyn_bin_edges[0], p_dyn_bin_edges[-1]), 
        #                      yscale = 'log')
        # for ax_name in ['dawn_p_dyn', 'dusk_p_dyn']:
        #     axd[ax_name].set(yticklabels='')
            
        # fig.suptitle(ell_bound)
        # plt.show()
    
        #   Attempt to fit a truncated pdf to the data
        #   Truncated regression model
        def truncated_regression(x, y, bounds):
            with pm.Model() as model:
                slope = 0.0 # pm.Uniform('slope', -0.5, 0.5)
                #intercept = pm.Normal('intercept', 80, 16)
                intercept = pm.Uniform('intercept', 30, 180)
                #sigma = pm.HalfNormal('sigma', sigma=30)
                sigma = pm.Uniform('sigma', 0, 50)
                
                #   Assume the obs are normally distributed about a line defined by mu
                normal_dist = pm.Normal.dist(mu=slope * x + intercept, sigma=sigma)
                pm.Truncated("obs", normal_dist, lower=bounds[0], upper=bounds[1], observed=y)
                
            return model
        
        if len(dawn_df.index) > 0:
            #   Get the maximum rho of June between the earliest measured 
            #   crossing and the latest (with some slight padding)
            earliest_crossing = np.min(dawn_df.index) - datetime.timedelta(hours=2)
            latest_crossing = np.max(dawn_df.index) + datetime.timedelta(hours=2)
            max_rho = np.max(Juno_df.loc[(Juno_df.index > earliest_crossing) 
                                         & (Juno_df.index < latest_crossing), 
                                         'rho'])
            
            dawn_rho_bounds = [1, max_rho]
            dawn_truncated_model = truncated_regression(*dawn_df.loc[:, ['ell', 'rho']].to_numpy().T, dawn_rho_bounds)
            with dawn_truncated_model:
                dawn_truncated_fit = pm.sample()
            dawn_fit_df = dawn_truncated_fit.to_dataframe(include_coords=False, 
                                                          groups='posterior')
            
        if len(dusk_df.index) > 0:
            
            #   Get the maximum rho of June between the earliest measured 
            #   crossing and the latest (with some slight padding)
            earliest_crossing = np.min(dusk_df.index) - datetime.timedelta(hours=2)
            latest_crossing = np.max(dusk_df.index) + datetime.timedelta(hours=2)
            max_rho = np.max(Juno_df.loc[(Juno_df.index > earliest_crossing) 
                                         & (Juno_df.index < latest_crossing), 
                                         'rho'])
        
            dusk_rho_bounds = [1, max_rho]
            dusk_truncated_model = truncated_regression(*dusk_df.loc[:, ['ell', 'rho']].to_numpy().T, dusk_rho_bounds)
            with dusk_truncated_model:
                try:
                    dusk_truncated_fit = pm.sample()
                except:
                    breakpoint()
            dusk_fit_df = dusk_truncated_fit.to_dataframe(include_coords=False, 
                                                          groups='posterior')
        
        fig, axs = plt.subplots(figsize=(6,4), ncols=2)
        r_abcissa = np.linspace(rho_bin_edges[0], rho_bin_edges[-1], 100)
        
        #   for export
        temp_d = {'ell': np.zeros(100) + ell_ref, 'rho': r_abcissa}
        
        if len(dawn_df.index) > 0:
            histogram_rho(axs[0], dawn_df, density=True, 
                          color='black', label = 'Crossings')
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(dawn_fit_df['intercept'].to_numpy().reshape(-1, 1))
            log_kde_pdf = kde.score_samples(r_abcissa.reshape(-1,1))
            kde_pdf = np.exp(log_kde_pdf)
            axs[0].plot(r_abcissa, kde_pdf, color='C4', label = r'Fit $\mu$')
            
            axs[0].axvline(dawn_rho_bounds[1], linestyle='--', color='C0', label=r'Max Apojove $\rho$')
            
            axs[0].legend()
            
            temp_d['dawn'] = kde_pdf
        else:
            temp_d['dawn'] = np.zeros(100)
        
        if len(dusk_df.index) > 0:
            histogram_rho(axs[1], dusk_df, density=True, 
                          color='black', label = 'Crossings')
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(dusk_fit_df['intercept'].to_numpy().reshape(-1, 1))
            log_kde_pdf = kde.score_samples(r_abcissa.reshape(-1,1))
            kde_pdf = np.exp(log_kde_pdf)
            axs[1].plot(r_abcissa, kde_pdf, color='C4', label = r'Fit $\mu$')
            
            axs[1].axvline(dusk_rho_bounds[1], linestyle='--', color='C0', label=r'Max Apojove $\rho$')
            
            axs[1].legend()
            
            temp_d['dusk'] = kde_pdf
        else:
            temp_d['dusk'] = np.zeros(100)
        
        fig.suptitle(r'{} $R_J$ < $x_{{JSS}}$ < {} $R_J$'.format(*ell_bound))
        axs[0].set(title='Dawn Flank',
                   xlim = (rho_bin_edges[0], rho_bin_edges[-1]),
                   xlabel = r'$\rho$ [$R_J$] (+ outward)',
                   ylabel = 'Probability Density')
        axs[1].set(title = 'Dusk Flank',
                   xlim = (rho_bin_edges[0], rho_bin_edges[-1]),
                   xlabel = r'$\rho$ [$R_J$] (+ outward)')
    
        d[str(ell_ref)] = pd.DataFrame(temp_d)
        
        # fig, axs = plt.subplots(figsize=(10, 5), ncols=3)
        
        # az.plot_posterior(truncated_fit, var_names=["intercept"], ax=axs[1])
        # axs[1].set(title="Truncated regression\n(truncated data)", xlabel="intercept")
        # az.plot_posterior(truncated_fit, var_names=["sigma"], ax=axs[2])
        # axs[2].set(title="Truncated regression\n(truncated data)", xlabel="sigma")

        plt.show()
        
    
    #   Visualize the crossings in ell-rho space
    fig, ax = plt.subplots(figsize=(6,4))
    
    for _, df in d.items():
        ax.scatter(df['ell'], df['rho'], c = df['dawn'], marker='_', s=1000)
        
        #ax.scatter(df['ell']+5, df['rho'], c = df['dusk'], marker='_', s=100, cmap='magma')
    
    ax.scatter(*crossings_df.query('phi > 0').loc[:, ['ell', 'rho']].to_numpy().T,
               color='xkcd:gold',
               label = 'Dawn Crossings')
    # ax.scatter(*crossings_df.query('phi < 0').loc[:, ['ell', 'rho']].to_numpy().T,
    #            color='xkcd:blue',
    #            label = 'Dusk Crossings')
    
    ax.legend()
    
    ax.add_patch(plt.Circle((0,0), 1.0, color='xkcd:peach'))
    ax.add_patch(patch.Wedge((0,0), 1.0, 90, 270, color='black'))
    ax.set(xlim = (-100, 40), xlabel = r'$x_{JSS}$ [$R_J$] (+sunward)',
           ylim = (-10, 120), ylabel = r'$\rho_{JSS}$ [$R_J$] (+outward)')
    plt.show()
    
    breakpoint()
    
    breakpoint()