#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:53 2024

@author: mrutala
"""
import numpy as np
import pymc as pm

# =============================================================================
# Section 1: Define a useful coordinate system for the boundaries
# =============================================================================
def convert_CartesianToCylindricalSolar(x, y, z):
    """
    Defines a cylindrical coordinate system with:
        Longitudinal axis pointing toward the Sun
        Polar axis pointing toward the north rotational pole
        Angles measured positive counterclockwise from the polar axis

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    ell : TYPE
        DESCRIPTION.

    """
    
    rho = np.sqrt(y**2 + z**2)
    phi = np.arctan2(-y, z)
    ell = x
    
    return np.array([rho, phi, ell])

def convert_SphericalSolarToCartesian(r, t, p):
    x = r * np.cos(t)
    y = - r * np.sin(t) * np.sin(p)
    z = r * np.sin(t) * np.cos(p)
    
    return np.array([x, y, z])

def convert_CylindricalSolarToSphericalSolar(rho, phi, ell):
    
    r = np.sqrt(rho**2 + ell**2)
    t = np.arctan2(rho, ell)
    p = phi
    
    return np.array([r, t, p])

def convert_CartesianToSphericalSolar(x, y, z):
    rho, phi, ell = convert_CartesianToCylindricalSolar(x, y, z)
    r, t, p = convert_CylindricalSolarToSphericalSolar(rho, phi, ell)
    
    return np.array([r, t, p])

def convert_SphericalSolarToCylindricalSolar(r, t, p):
    
    rho = r * np.sin(t)
    phi = p
    ell = r * np.cos(t)
    
    return np.array([rho, phi, ell])


# =============================================================================
# Model lookup and initialization utilities
# =============================================================================
def lookup(model_number):
    bm = {'001': 'Shuelike',
          '002': 'Shuelike_Asymmetric',
          '003': 'Shuelike_AsymmetricAlpha'}
    
    return bm['{:03d}'.format(model_number)]

def init(model_name):
    #   Select boundary model
    bm = {'Shuelike': 
              {'model': Shuelike, 
               'model_number': 1,
               'param_dict': {
                   'r0': [30, 40, 50, 60, 70, 80],
                   'r1': [-0.1, -0.2, -0.3, -0.4],
                   'a0': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   'a1': [0.01, 0.1, 1.0, 10.0]},
               'param_distributions': {
                   'r0': pm.InverseGamma,
                   'r1': pm.Normal,
                   'a0': pm.InverseGamma,
                   'a1': pm.Uniform},
               'param_descriptions': {
                   'r0': {'mu': 60, 'sigma': 30},
                   'r1': {'mu': -0.2, 'sigma': 0.05},
                   'a0': {'mu': 1, 'sigma': 0.5},
                   # 'a1': {'lower': "-1 * param_dict['a0']/p_dyn", 'upper': "2", 'EVAL_NEEDED':True}}
                   # 'a1': {'lower': "-1 * param_dict['a0']", 'upper': "1", 'EVAL_NEEDED':True}}
                   'a1': {'lower': -10, 'upper': 10}}
               },
          'Shuelike_Asymmetric':
              {'model': Shuelike_Asymmetric, 
               'model_number': 2,
               'param_dict': {
                   'r0': [30, 40, 50, 60, 70, 80],
                   'r1': [-0.1, -0.2, -0.3, -0.4],
                   'a0': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   'a1': [0.01, 0.1, 1.0, 10.0]},
               'param_distributions': {
                   'r0': pm.InverseGamma,
                   'r1': pm.Normal,
                   'r2': pm.Normal,
                   'r3': pm.Normal,
                   'a0': pm.InverseGamma,
                   'a1': pm.Uniform},
               'param_descriptions': {
                   'r0': {'mu': 60, 'sigma': 30},
                   'r1': {'mu': -0.2, 'sigma': 0.05},
                   'r2': {'mu': 0, 'sigma': 10},
                   'r3': {'mu': 0, 'sigma': 10},
                   'a0': {'mu': 1.0, 'sigma': 0.5},
                   'a1': {'lower': -10, 'upper': 10}}
               }
          }
    return bm[model_name]
          

# =============================================================================
# Section 1.5: 3D Functional Forms for the boundaries
# =============================================================================
def Shuelike_Static(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t'), 'r'
    
    t = coordinates
    r0, a0 = parameters

    
    r = r0 * (2/(1 + np.cos(t)))**a0

    return r
# =============================================================================
# Section 2: 4D Functional forms for the boundaries
# =============================================================================

def Shuelike(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    #rho, phi, ell = coordinates
    #r, t, p = convert_CylindricalSolarToSphericalSolar(*coordinates)
    t, p, p_dyn = coordinates
    r0, r1, a0, a1 = parameters
    
    # if (p_dyn < 0).any():
    #     breakpoint()
    r_0 = r0*((p_dyn)**r1)
    
    a_0 = a0 + a1 * p_dyn
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r

def Shuelike_UniformPressureExponent(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    #rho, phi, ell = coordinates
    #r, t, p = convert_CylindricalSolarToSphericalSolar(*coordinates)
    t, p, p_dyn = coordinates
    r0, r1, a0, a1 = parameters
    
    # if (p_dyn < 0).any():
    #     breakpoint()
    r_0 = r0*((p_dyn)**r1)
    
    a_0 = a0 + a1 * p_dyn**r1
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r

def Shuelike_NonuniformPressureExponent(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    #rho, phi, ell = coordinates
    #r, t, p = convert_CylindricalSolarToSphericalSolar(*coordinates)
    t, p, p_dyn = coordinates
    r0, r1, a0, a1, a2 = parameters
    
    # if (p_dyn < 0).any():
    #     breakpoint()
    r_0 = r0*((p_dyn)**r1)
    
    a_0 = a0 + a1 * p_dyn**a2
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r

def Shuelike_AsymmetryCase1(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    #rho, phi, ell = coordinates
    #r, t, p = convert_CylindricalSolarToSphericalSolar(*coordinates)
    t, p, p_dyn = coordinates
    r0, r1, r2, r3, r4, a0, a1 = parameters
    
    # if (p_dyn < 0).any():
    #     breakpoint()
    r_0 = r0*((p_dyn)**r1) + r2*np.sin(p)**2 + r3*np.sin(t + r4)*np.sin(p)
    
    a_0 = a0 + a1 * p_dyn**r1
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0 
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r
def Shuelike_Asymmetric(parameters=[], coordinates=[], variables=False):
    """
    r = r_0 (2/(1 + cos(theta)))^alpha

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    #rho, phi, ell = coordinates
    #r, t, p = convert_CylindricalSolarToSphericalSolar(*coordinates)
    t, p, p_dyn = coordinates
    r0, r1, r2, r3, a0, a1 = parameters
    
    # with np.errstate(invalid='raise'):
    #     try: 
    #         r_0 = r0*(p_dyn**r1)
    #     except RuntimeError():
    #         print('You caught the error!')
    r_0 = (r0 + r2*np.cos(p)**2 + r3*np.sin(p)*np.sin(t))*((p_dyn)**r1)
    
    a_0 = (a0 + a1 * p_dyn)# * (1 + a3*np.sin(p) + a4*np.sin(-p) + a5*np.cos(p)**2)
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r

def Shuelike_AsymmetryCase2(parameters=[], coordinates=[], variables=False):
    """
    

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    
    t, p, p_dyn = coordinates
    r0, r1, r2, r3, r4, a0, a1, a2, a3, a4 = parameters
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    r_n = (r0 * p_dyn**r1) * (r2*np.cos(p)**2 + r3*sg_pos*np.sin(p)*np.cos(t) + r4*sg_neg*np.sin(p)*np.cos(t))
    
    a_f = (a0 + p_dyn*a1) * (a2*np.cos(p)**2 + a3*sg_pos*np.sin(p)*np.cos(t) + a4*sg_neg*np.sin(p)*np.cos(t))
    
    r = r_n * (2/(1 + np.cos(t)))**a_f

    return r

def Joylike(parameters=[], coordinates=[], variables=False):
    """
    

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    z : TYPE
        DESCRIPTION.

    """
    if variables:
        return ('x', 'y', 'p_dyn'), 'abs_z'
    
    x, y, p_dyn = coordinates
    a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1 = parameters
    
    x = x * 1/120
    y = y * 1/120
    
    a = a0 + a1*p_dyn**(-1/4)
    b = b0 + b1*p_dyn**(-1/4)
    c = c0 + c1*p_dyn**(-1/4)
    d = d0 + d1*p_dyn
    e = e0 + e1*p_dyn
    f = f0 + f1*p_dyn
    z = np.sqrt(a + b*x + c*x**2 + d*y + e*y**2 + f*x*y)
    return z * 120
    
    return


def boundary_Winslowlike():
    
    return

def boundary_Caternary():
    
    return

# =============================================================================
# 
# =============================================================================
def explore_Variation(model_name):
    
    model_dict = init(model_name)
    
    
    breakpoint()
    # Draw samples to give a sense of the model spread:
    posterior_params_samples = az.extract(posterior, num_samples=100)
    
    posterior_params_mean = []
    posterior_params_vals = []
    for param_name in model_dict['param_distributions'].keys():
        
        # Get mean values for each parameter
        posterior_params_mean.append(np.mean(posterior[param_name].values))
        
        # And record the sample values
        posterior_params_vals.append(posterior_params_samples[param_name].values)
    
    # Transpose so we get a list of params in proper order
    posterior_params_vals = np.array(posterior_params_vals).T
        
    # Plotting coords
    n_coords = int(1e4)
    mean_p_dyn = np.mean(positions_df['p_dyn'])
    
    t_coord = np.linspace(0, 0.99*np.pi, n_coords)
    
    p_coords = {'North': np.full(n_coords, 0),
                'South': np.full(n_coords, +np.pi),
                'Dawn': np.full(n_coords, +np.pi/2.),
                'Dusk': np.full(n_coords, -np.pi/2.)
                }
    
    p_dyn_coords = {'16': np.full(n_coords, np.percentile(positions_df['p_dyn'], 16)),
                    '50': np.full(n_coords, np.percentile(positions_df['p_dyn'], 50)),
                    '84': np.full(n_coords, np.percentile(positions_df['p_dyn'], 84))
                    }
    
    fig, axs = plt.subplots(nrows = 3, sharex = True,
                            figsize = (6.5, 5))
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.7, top=0.98,
                        hspace=0.08)
    
    # Set up each set of axes
    # x_label_centered_x = (axs[0].get_position()._points[0,0] + axs[0].get_position()._points[1,0])/2.
    # x_label_centered_y = (0 + axs[1].get_position()._points[0,1])/2.
    # fig.supxlabel(r'$x_{JSS}$ [$R_J$] (+ toward Sun)', 
    #               position = (x_label_centered_x, x_label_centered_y),
    #               ha = 'center', va = 'top')
    # y_label_centered_x = (0 + axs[0].get_position()._points[0,0])/2.
    # y_label_centered_y = (axs[0].get_position()._points[1,1] + axs[1].get_position()._points[0,1])/2.
    # fig.supylabel(r'$\rho_{JSS} = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]', 
    #               position = (y_label_centered_x, y_label_centered_y),
    #               ha = 'right', va = 'center')
    
    axs[0].set(xlim = (300, -600),
               ylim = (-200, 200),
               aspect = 1)
    axs[1].set(ylim = (-200, 200),
               aspect = 1)
    axs[2].set(ylim = (0, 400),
               aspect = 1)
    
    axs[0].annotate('(a)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    axs[1].annotate('(b)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    axs[2].annotate('(c)', (0,1), (0.5,-1.5), 'axes fraction', 'offset fontsize')
    
    direction_colors = {'North': 'C0',
                        'South': 'C1',
                        'Dawn': 'C3',
                        'Dusk': 'C5'}
    p_dyn_linestyles = {'16': ':',
                        '50': '-',
                        '84': '--'}

    # Top axes: Side-view, dusk on bottom
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['Dusk'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dusk'])
    axs[0].plot(xyz[0], xyz[1], color='black')
    
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
    axs[0].plot(xyz[0], xyz[1], color='black')
    
    for params in posterior_params_vals:
        r_coord = model_dict['model'](params, [t_coord, p_coords['Dusk'], p_dyn_coords['50']])
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dusk'])
        axs[0].plot(xyz[0], xyz[1], color='black', alpha=0.05, zorder=-10)
        
        r_coord = model_dict['model'](params, [t_coord, p_coords['Dawn'], p_dyn_coords['50']])
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['Dawn'])
        axs[0].plot(xyz[0], xyz[1], color='black', alpha=0.05, zorder=-10)
    
    # Middle axes: Top-down view
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['North'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['North'])
    axs[1].plot(xyz[0], xyz[2], color='black')
    
    r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords['South'], p_dyn_coords['50']])
    xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['South'])
    axs[1].plot(xyz[0], xyz[2], color='black')
    
    for params in posterior_params_vals:
        r_coord = model_dict['model'](params, [t_coord, p_coords['North'], p_dyn_coords['50']])
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['North'])
        axs[1].plot(xyz[0], xyz[2], color='black', alpha=0.05, zorder=-10)
        
        r_coord = model_dict['model'](params, [t_coord, p_coords['South'], p_dyn_coords['50']])
        xyz = BM.convert_SphericalSolarToCartesian(r_coord, t_coord, p_coords['South'])
        axs[1].plot(xyz[0], xyz[2], color='black', alpha=0.05, zorder=-10)
    
    # Bottom axes: plot for different pressures, superimposed
    for p_dyn_value in p_dyn_coords.keys():
        for direction in ['North', 'South', 'Dawn', 'Dusk']:
            
            r_coord = model_dict['model'](posterior_params_mean, [t_coord, p_coords[direction], p_dyn_coords[p_dyn_value]])
            rpl = BM.convert_SphericalSolarToCylindricalSolar(r_coord, t_coord, p_coords[direction])
            axs[2].plot(rpl[2], rpl[0],
                        color = direction_colors[direction], ls = p_dyn_linestyles[p_dyn_value],
                        label = r'{}, $p_{{dyn}} = {}^{{th}} \%ile$'.format(direction, p_dyn_value))
            
            axs[2].legend()
    
    return


def explore_Variation_ShueLike_Static():
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=2, figsize=(8,8), sharex=True)
    plt.subplots_adjust(left=0.15, bottom=0.1, top=0.975, right=0.975, hspace=0.025)
    theta = np.linspace(-180, 180, 1000) * np.pi/180
    
    for r0 in [40, 50, 60, 70, 80]:
        r = Shuelike_Static([r0, 0.6], theta)
        axs[1].plot(r*np.cos(theta), r*np.sin(theta), label = r'$r_0$ = {:n}'.format(r0))
        
    for alpha in [0.3, 0.5, 0.7, 0.9, 1.1]:
        r = Shuelike_Static([60, alpha], theta)
        axs[0].plot(r*np.cos(theta), r*np.sin(theta), label = r'$\alpha$ = {:.1f}'.format(alpha))
    
    axs[0].plot(Shuelike_Static([60, 0.7], theta)*np.cos(theta), Shuelike_Static([60, 0.7], theta)*np.sin(theta), color='white', linewidth=6, zorder=-1)
        
    for ax in axs:
        ax.legend(loc = 'lower left')
        ax.set(xlim=(-300, 100), ylim=(0,200))
        ax.set(aspect = 1)
        
    axs[1].set(xlabel=r'$x_{JSS}$ [$R_J$] (+ sunward)')
    fig.supylabel(r'$\rho = \sqrt{y_{JSS}^2 + z_{JSS}^2}$ [$R_J$]')
    
    axs[0].annotate(r'For $r_0 = 60$', (1,1), (-1,-1), xycoords='axes fraction', textcoords='offset fontsize', ha='right', va='top')
    axs[1].annotate(r'$\alpha = 0.6 = $const.', (1,1), (-1,-1), xycoords='axes fraction', textcoords='offset fontsize', ha='right', va='top')

    plt.show()
