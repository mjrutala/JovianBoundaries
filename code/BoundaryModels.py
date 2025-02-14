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

def plot_CoordinateSystemsDiagram():
    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    # Load custom plotting style
    try:
        plt.style.use('/Users/mrutala/code/python/mjr.mplstyle')
    except:
        pass

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
    
            return np.min(zs)


    fig = plt.figure(figsize=(3.5,3.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    ax.set(xlim = (0,1.2), xticks=np.arange(0,1.2,0.1), xticklabels = [], 
           ylim = (0,1.2), yticks=np.arange(0,1.2,0.1), yticklabels = [],
           zlim = (0,1.2), zticks=np.arange(0,1.2,0.1), zticklabels = [])
    ax.minorticks_off()
    ax.get_xaxis().set_visible(False)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=35, azim=45)
    ax.plot((0, 0), (1.2, 1.2), (0, 1.2), color='black', lw=1, zorder=-16)
    ax.plot((0, 0), (0, 1.2), (1.2, 1.2), color='black', lw=1, zorder=-16)
    ax.plot((0, 1.2), (0, 0), (1.2, 1.2), color='black', lw=1, zorder=-16)
    ax.plot((1.2, 1.2), (0, 0), (1.2, 0), color='black', lw=1, zorder=-16)
    ax.plot((1.2, 1.2), (0, 1.2), (0, 0), color='black', lw=1, zorder=-16)
    ax.plot((1.2, 0), (1.2, 1.2), (0, 0), color='black', lw=1, zorder=-16)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Plot grid lines
    for val in np.arange(0, 1.2, 0.1):
        ax.plot([0, 0], [val, val], [0, 1.2], lw=0.5, color='#aaaaaa', zorder=-64)
        ax.plot([0, 0], [0, 1.2], [val, val], lw=0.5, color='#aaaaaa', zorder=-64)
        ax.plot([val, val], [0, 0], [0, 1.2], lw=0.5, color='#aaaaaa', zorder=-64)
        ax.plot([0, 1.2], [0, 0], [val, val], lw=0.5, color='#aaaaaa', zorder=-64)
        ax.plot([val, val], [0, 1.2], [0, 0], lw=0.5, color='#aaaaaa', zorder=-64)
        ax.plot([0, 1.2], [val, val], [0, 0], lw=0.5, color='#aaaaaa', zorder=-64)
        
    
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    
    # JSS x, y, z axes
    x = Arrow3D([0,1], [0,0], [0,0], **arrow_prop_dict)
    ax.text(0.9, 0, 0.1, r'$\hat{x}_{JSS}$', [1, 0, 0], ha='center', va='center', size='large')
    y = Arrow3D([0,0], [0,1], [0,0], **arrow_prop_dict)
    ax.text(0.1, 0.9, 0, r'$\hat{y}_{JSS}$', [0, 1, 0], ha='center', va='center', size='large')
    z = Arrow3D([0,0], [0,0], [0,1], **arrow_prop_dict)
    ax.text(0, 0.1, 0.9, r'$\hat{z}_{JSS}$', [0, 0, 1], ha='center', va='center', size='large')
    for artist in [x, y, z]:
        ax.add_artist(artist)
        
    ax.text(0.9, 0, 0.2, r'$\leftarrow$ to Sun', [1, 0, 0], ha='center', va='center', size='large')
     
    # "Point of interest"
    # Solar Spherical r, theta, phi
    poi = (1.1, 60, -60) # point of interest
    poi_radians = (poi[0], np.radians(poi[1]), np.radians(poi[2]))
    
    x, y, z = convert_SphericalSolarToCartesian(np.array(poi_radians[0]), 
                                                np.array(poi_radians[1]), 
                                                np.array(poi_radians[2]))
    
    red = '#e50000'
    blue = '#75bbfd'
    purple = '#bf77f6'
    
    # Plot Jupiter
    ax.plot([0], [0], [0], marker='o', markersize=8, color='xkcd:peach', zorder=128)
    
    # Plot the vector
    ax.plot([0, x], [0, y], [0, z], color=purple, lw=2, zorder=4)
    ax.scatter([x], [y], [z], color=purple, s=24, edgecolors='black', lw=1, zorder=16)
    ax.text(x+0.05, y-0.05, z, r'$r$', zorder=10, size='large', ha='center', va='center')
    
    # Plot rho component of the vector
    ax.plot([0, 0], [0, y], [0, z], color=red, ls=':', lw=2, zorder=8)
    ax.text(0, y/2-0.075, z/2+0.025, r'$\rho$', size='large', ha='center', va='center')
    
    # Plot x component of the vector
    ax.plot([0, x], [y, y], [z, z], color=blue, ls=':', lw=2, zorder=8)
    # No text
    
    # theta, in 3D
    r, t, p = np.linspace(0.5, 0.5, 100), np.linspace(0, poi_radians[1], 100), np.zeros(100) + poi_radians[2]
    x, y, z = convert_SphericalSolarToCartesian(r, t, p)
    ax.plot(x, y, z,'k-', lw=1, zorder=2)
    ax.text(x[50]+0.025, y[50]+0.1, z[50], r'$\theta$', size='large', ha='center', va='center')
    
    # phi, in 3D
    r, t, p = np.linspace(0.5, 0.5, 100), np.linspace(np.pi/2, np.pi/2, 100), np.linspace(0, poi_radians[2], 100)
    x, y, z = convert_SphericalSolarToCartesian(r, t, p)
    ax.plot(x, y, z,'k-', lw=1, zorder=2)
    ax.text(x[50], y[50]+0.05, z[50]+0.05, r'$-\phi$', size='large', ha='center', va='center')
    
    # r = Arrow3D([0, xyz_r[0]], [0, xyz_r[1]], [0, xyz_r[2]], **arrow_prop_dict)
    # ax.add_artist(r)
        
    plt.show()


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
    
    bm = {}
    
    bm['Shuelike'] = {'model': Shuelike,
                      'param_dict': {},
                      'param_distributions': {'r0': pm.Gamma,
                                              'r1': pm.Normal,
                                              'a0': pm.Gamma,
                                              'a1': pm.Normal},
                      'param_descriptions': {'r0': {'mu': 40, 'sigma': 20},
                                             'r1': {'mu': -0.25, 'sigma': 0.05},
                                             'a0': {'mu': 1, 'sigma': 0.5},
                                             'a1': {'mu': 0, 'sigma': 1}}
                      }
    # bm['Shuelike_log'] = {'model': Shuelike_log,
    #                   'param_dict': {},
    #                   'param_distributions': {'r0': pm.Normal,
    #                                           'r1': pm.Normal,
    #                                           'a0': pm.LogNormal,
    #                                           'a1': pm.Normal},
    #                   'param_descriptions': {'r0': {'mu': 3.5, 'sigma': 0.5},
    #                                          'r1': {'mu': -0.25, 'sigma': 0.05},
    #                                          'a0': {'mu': 0, 'sigma': 0.25},
    #                                          'a1': {'mu': 0, 'sigma': 1}}
    #                   }
    
    bm['Shuelike_r1fixed'] = {'model': Shuelike_r1fixed,
                              'param_dict': {},
                              'param_distributions': {'r0': pm.Gamma,
                                                      'a0': pm.Gamma,
                                                      'a1': pm.Normal},
                              'param_descriptions': {'r0': {'mu': 30, 'sigma': 15},
                                                     'a0': {'mu': 0.8, 'sigma': 0.4},
                                                     'a1': {'mu': 0, 'sigma': 1}}
                              }
    bm['Joylike'] = {'model': Joylike,
                     'param_dict': {},
                     'param_distributions': {'r1': pm.Normal,
                                             'A0': pm.Normal,
                                             'A1': pm.Normal,
                                             'B0': pm.Normal,
                                             'B1': pm.Normal,
                                             'C0': pm.Normal,
                                             'C1': pm.Normal,
                                             'D0': pm.Normal,
                                             'D1': pm.Normal,
                                             'E0': pm.Normal,
                                             'E1': pm.Normal,
                                             'F0': pm.Normal,
                                             'F1': pm.Normal},
                     'param_descriptions': {'r1': {'mu': -0.25, 'sigma': 0.05},
                                            'A0': {'mu':-0.62, 'sigma':0.49},
                                            'A1': {'mu':+1.04, 'sigma':0.55},
                                            'B0': {'mu':-0.57, 'sigma':0.01},
                                            'B1': {'mu':-0.52, 'sigma':0.30},
                                            'C0': {'mu':-0.07, 'sigma':0.12},
                                            'C1': {'mu':+0.04, 'sigma':0.02},
                                            'D0': {'mu':+0.03, 'sigma':0.05},
                                            'D1': {'mu':+0.03, 'sigma':0.07},
                                            'E0': {'mu':-0.84, 'sigma':0.03},
                                            'E1': {'mu':-0.55, 'sigma':0.85},
                                            'F0': {'mu':-0.05, 'sigma':0.10},
                                            'F1': {'mu':+0.15, 'sigma':0.22}}
                     }
    bm['Joylike_r1fixed'] = {'model': Joylike_r1fixed,
                             'param_dict': {},
                             'param_distributions': {'A0': pm.Normal,
                                                     'A1': pm.Normal,
                                                     'B0': pm.Normal,
                                                     'B1': pm.Normal,
                                                     'C0': pm.Normal,
                                                     'C1': pm.Normal,
                                                     'D0': pm.Normal,
                                                     'D1': pm.Normal,
                                                     'E0': pm.Normal,
                                                     'E1': pm.Normal,
                                                     'F0': pm.Normal,
                                                     'F1': pm.Normal},
                             'param_descriptions': {'A0': {'mu':-0.62, 'sigma':0.49},
                                                    'A1': {'mu':+1.04, 'sigma':0.55},
                                                    'B0': {'mu':-0.57, 'sigma':0.01},
                                                    'B1': {'mu':-0.52, 'sigma':0.30},
                                                    'C0': {'mu':-0.07, 'sigma':0.12},
                                                    'C1': {'mu':+0.04, 'sigma':0.02},
                                                    'D0': {'mu':+0.03, 'sigma':0.05},
                                                    'D1': {'mu':+0.03, 'sigma':0.07},
                                                    'E0': {'mu':-0.84, 'sigma':0.03},
                                                    'E1': {'mu':-0.55, 'sigma':0.85},
                                                    'F0': {'mu':-0.05, 'sigma':0.10},
                                                    'F1': {'mu':+0.15, 'sigma':0.22}}
                     }
    bm['Shuelike_rasymmetric'] = {'model': Shuelike_rasymmetric, 
                                  'model_number': 3,
                                  'param_dict': {},
                                  'param_distributions': {'r0': pm.InverseGamma,
                                                          'r1': pm.Normal,
                                                          'r2': pm.Normal,
                                                          'r3': pm.Normal,
                                                          'r4': pm.Normal,
                                                          'a0': pm.InverseGamma,
                                                          'a1': pm.Normal},
                                  'param_descriptions': {'r0': {'mu': 50, 'sigma': 10},
                                                         'r1': {'mu': -0.2, 'sigma': 0.05},
                                                         'r2': {'mu': -5, 'sigma': 3},
                                                         'r3': {'mu': 0, 'sigma': 10},
                                                         'r4': {'mu': 0, 'sigma': 10},
                                                         'a0': {'mu': 0.75, 'sigma': 0.25},
                                                         'a1': {'mu': -1.0, 'sigma': 2.0}}
                                  }
    
    bm['Shuelike_rasymmetric_r1fixed'] = {'model': Shuelike_rasymmetric_r1fixed, 
                                          'model_number': 3,
                                          'param_dict': {},
                                          'param_distributions': {'r0': pm.InverseGamma,
                                                                  'r2': pm.Normal,
                                                                  'r2_scale': pm.Gamma,
                                                                  'r3': pm.Normal,
                                                                  'r4': pm.Normal,
                                                                  'a0': pm.InverseGamma,
                                                                  'a1': pm.Normal},
                                          'param_descriptions': {'r0': {'mu': 50, 'sigma': 30},
                                                                 'r2': {'mu': -5, 'sigma': 3},
                                                                 'r2_scale': {'mu': 0.1, 'sigma':0.1},
                                                                 'r3': {'mu': 10, 'sigma': 10},
                                                                 'r4': {'mu': 10, 'sigma': 10},
                                                                 'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                 'a1': {'mu': -1.0, 'sigma': 1.0}}
                                  }
    bm['Shuelike_rasymmetric_simple'] = {'model': Shuelike_rasymmetric_simple, 
                                          'model_number': 3,
                                          'param_dict': {},
                                          'param_distributions': {'r0': pm.InverseGamma,
                                                                  'r3': pm.Normal,
                                                                  'r4': pm.Normal,
                                                                  'a0': pm.InverseGamma,
                                                                  'a1': pm.Normal},
                                          'param_descriptions': {'r0': {'mu': 30, 'sigma': 30},
                                                                 'r3': {'mu': 10, 'sigma': 10},
                                                                 'r4': {'mu': 10, 'sigma': 10},
                                                                 'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                 'a1': {'mu': -1.0, 'sigma': 1.0}}
                                  }
    bm['Shuelike_aasymmetric_r1fixed'] = {'model': Shuelike_aasymmetric_r1fixed, 
                                          'model_number': 3,
                                          'param_dict': {},
                                          'param_distributions': {'r0': pm.InverseGamma,
                                                                  'a0': pm.InverseGamma,
                                                                  'a1': pm.Normal,
                                                                  'a2': pm.Beta,
                                                                  'a3': pm.Beta,
                                                                  'a4': pm.Beta},
                                          'param_descriptions': {'r0': {'mu': 50, 'sigma': 30},
                                                                 'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                 'a1': {'mu': -1.0, 'sigma': 1.0},
                                                                 'a2': {'alpha': 2.0, 'beta': 5.0},
                                                                 'a3': {'alpha': 2.0, 'beta': 5.0},
                                                                 'a4': {'alpha': 2.0, 'beta': 5.0},}
                                  }
    
    
    
    
    
    bm['ShuelikeAsymmetric'] = {'model': ShuelikeAsymmetric, 
                                'model_number': 3,
                                'param_dict': {},
                                'param_distributions': {'r0': pm.Gamma,
                                                        'r1': pm.Normal,
                                                        'r2': pm.Gamma,
                                                        'r3': pm.Gamma,
                                                        'a0': pm.Gamma,
                                                        'a1': pm.Normal},
                                'param_descriptions': {'r0': {'mu': 60, 'sigma': 30},
                                                       'r1': {'mu':-0.25, 'sigma':0.03},
                                                       'r2': {'mu': 10, 'sigma': 10},
                                                       'r3': {'mu': 10, 'sigma': 10},
                                                       'a0': {'mu': 1.0, 'sigma': 0.5},
                                                       'a1': {'mu': -1.0, 'sigma': 1.0}}
                                }
    bm['ShuelikeAsymmetric_r1fixed'] = {'model': ShuelikeAsymmetric_r1fixed, 
                                              'model_number': 3,
                                              'param_dict': {},
                                              'param_distributions': {'r0': pm.Gamma,
                                                                      'r2': pm.Gamma,
                                                                      'r3': pm.Gamma,
                                                                      'a0': pm.Gamma,
                                                                      'a1': pm.Normal},
                                              'param_descriptions': {'r0': {'mu': 60, 'sigma': 30},
                                                                     'r2': {'mu': 10, 'sigma': 10},
                                                                     'r3': {'mu': 10, 'sigma': 10},
                                                                     'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                     'a1': {'mu': -1.0, 'sigma': 1.0}}
                                              }
    bm['ShuelikeAsymmetric_AsPerturbation'] = {'model': ShuelikeAsymmetric_AsPerturbation, 
                                               'model_number': 3,
                                               'param_dict': {},
                                               'param_distributions': {'r0': pm.Gamma,
                                                                       'r1': pm.Normal,
                                                                       'r2': pm.Gamma,
                                                                       'r3': pm.Gamma,
                                                                       'a0': pm.Gamma,
                                                                       'a1': pm.Normal},
                                               'param_descriptions': {'r0': {'mu': 60, 'sigma': 30},
                                                                      'r1': {'mu':-0.25, 'sigma':0.03},
                                                                      'r2': {'mu': 10, 'sigma': 10},
                                                                      'r3': {'mu': 10, 'sigma': 10},
                                                                      'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                      'a1': {'mu': -1.0, 'sigma': 1.0}}
                                               }
    # bm['ShuelikeAsymmetric_AsPerturbation_r1fixed'] = {'model': ShuelikeAsymmetric_AsPerturbation_r1fixed, 
    #                                                    'model_number': 3,
    #                                                    'param_dict': {},
    #                                                    'param_distributions': {'r0': pm.Gamma,
    #                                                                            'r2': pm.Gamma,
    #                                                                            'r3': pm.Gamma,
    #                                                                            'a0': pm.Gamma,
    #                                                                            'a1': pm.Normal},
    #                                                    'param_descriptions': {'r0': {'mu': 40, 'sigma': 15},
    #                                                                           'r2': {'mu': 10, 'sigma': 10},
    #                                                                           'r3': {'mu': 10, 'sigma': 10},
    #                                                                           'a0': {'mu': 1.0, 'sigma': 0.5},
    #                                                                           'a1': {'mu': -1.0, 'sigma': 1.0}}
    #                                                    }
    bm['ShuelikeAsymmetric_AsPerturbation_2'] = {'model': ShuelikeAsymmetric_AsPerturbation_2, 
                                                 'model_number': 3,
                                                 'param_dict': {},
                                                 'param_distributions': {'r0': pm.Gamma,
                                                                         'r1': pm.Normal,
                                                                         'r2': pm.Gamma,
                                                                         'r3': pm.Gamma,
                                                                         'a0': pm.Gamma,
                                                                         'a1': pm.Normal},
                                                 'param_descriptions': {'r0': {'mu': 40, 'sigma': 20},
                                                                        'r1': {'mu': -0.25, 'sigma':0.03},
                                                                        'r2': {'mu': 10, 'sigma': 10},
                                                                        'r3': {'mu': 10, 'sigma': 10},
                                                                        'a0': {'mu': 1.0, 'sigma': 0.5},
                                                                        'a1': {'mu': 0, 'sigma': 1.0}}
                                                 }
    bm['ShuelikeAsymmetric_AsPerturbation_r1fixed_2'] = {'model': ShuelikeAsymmetric_AsPerturbation_r1fixed_2, 
                                                       'model_number': 3,
                                                       'param_dict': {},
                                                       'param_distributions': {'r0': pm.Gamma,
                                                                               'r2': pm.Gamma,
                                                                               'r3': pm.Gamma,
                                                                               'a0': pm.Gamma,
                                                                               'a1': pm.Normal},
                                                       'param_descriptions': {'r0': {'mu': 30, 'sigma': 10},
                                                                              'r2': {'mu': 10, 'sigma': 10},
                                                                              'r3': {'mu': 10, 'sigma': 10},
                                                                              'a0': {'mu': 1.0, 'sigma': 0.2},
                                                                              'a1': {'mu': 0, 'sigma': 2.0}}
                                                       }
       
    return bm[model_name]
          
# =============================================================================
# Section 1.5: 3D Functional Forms for the boundaries
# =============================================================================
# def Shuelike_Static(parameters=[], coordinates=[], variables=False):
#     """
#     r = r_0 (2/(1 + cos(theta)))^alpha

#     Parameters
#     ----------
#     parameters : TYPE
#         DESCRIPTION.
#     coordinates : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     r : TYPE
#         DESCRIPTION.

#     """
#     if variables:
#         return ('t'), 'r'
    
#     t = coordinates
#     r0, a0 = parameters

    
#     r = r0 * (2/(1 + np.cos(t)))**a0

#     return r
# =============================================================================
# Section 2: 4D Functional forms for the boundaries
# =============================================================================

def Shuelike(parameters=[], coordinates=[], variables=False,
             return_r_ss:bool=False, return_a_f:bool=False):
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
    # Optionally, return variables
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    
    # Unpack coordinates
    t, p, p_dyn = coordinates
    r0, r1, a0, a1 = parameters
    
    # Calculate r_b & a_f, returning one if requested
    r_ss = r0*((p_dyn)**(r1))
    a_f = a0 + a1 * p_dyn
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f
    
    return r

    # # Attempt to handle multiple parameters simultaneously-- pymc didn't like it :(
    # # Optionally, return variables
    # if variables:
    #     return ('t', 'p', 'p_dyn'), 'r'
    
    # # Unpack coordinates
    # t, p, p_dyn = coordinates
    
    # # Unpack parameters, checking that they are iterable
    # if len(np.shape(parameters)) == 1:
    #     parameters = [[p] for p in parameters]
    # r0arr, r1arr, a0arr, a1arr = parameters
    
    # r_arr, r_b_arr, a_f_arr = [], [], []
    # for r0, r1, a0, a1 in zip(r0arr, r1arr, a0arr, a1arr):
    #     # Calculate r_b & a_f, returning one if requested
    #     r_b_arr.append(r0*((p_dyn)**(r1)))
    #     a_f_arr.append(a0 + a1 * p_dyn)
        
    #     # Calculate r
    #     r_arr.append(r_b_arr[-1] * (2/(1 + np.cos(t)))**a_f_arr[-1])
        
    
    # if return_r_b:
    #     return np.squeeze(np.array(r_b_arr))
    # if return_a_f:
    #     return np.squeeze(np.array(a_f_arr))
    # return np.squeeze(np.array(r_arr))

def Shuelike_log(parameters=[], coordinates=[], variables=False,
             return_r_ss:bool=False, return_a_f:bool=False):
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
    # Optionally, return variables
    if variables:
        return ('t', 'p', 'log_p_dyn'), 'log_r'
    
    # Unpack coordinates
    t, p, log_p_dyn = coordinates
    log_r0, r1, a0, a1 = parameters
    
    # Calculate r_b & a_f, returning one if requested
    p_dyn = np.exp(log_p_dyn)
    r0 = np.exp(log_r0)
    
    r_ss = r0*((p_dyn)**(r1))
    a_f = a0 + a1 * p_dyn
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f
    log_r = np.log(r)
    
    return log_r


def Shuelike_r1fixed(parameters=[], coordinates=[], variables=False,
             return_r_b:bool=False, return_a_f:bool=False):
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
    # Optionally, return variables
    if variables:
        return ('t', 'p', 'p_dyn'), 'r'
    
    # Unpack coordinates
    t, p, p_dyn = coordinates
    r0, a0, a1 = parameters
    r1 = -2.5
    
    # Calculate r_b & a_f, returning one if requested
    r_b = r0*((p_dyn)**(r1/10))
    a_f = a0 + a1 * p_dyn
    
    if return_r_b:
        return r_b
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_b * (2/(1 + np.cos(t)))**a_f
    
    return r

def Shuelike_rasymmetric(parameters=[], coordinates=[], variables=False,
                         return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r1, r2, r3, r4, a0, a1 = parameters
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    r2_term = r2*np.cos(p)**2
    r3_term = r3*sg_pos*np.sin(p)**2
    r4_term = r4*sg_neg*np.sin(p)**2
    r_ss = (r0 + (np.sin(t/2)**2)*(r2_term + r3_term + r4_term)) * ((p_dyn)**r1)
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f
        
    return r

def Shuelike_rasymmetric_r1fixed(parameters=[], coordinates=[], variables=False,
                                 return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r2, r2_scale, r3, r4, a0, a1 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r3_term = r3*sg_pos*np.sin(p)
    r4_term = r4*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)) * (r2_term + r3_term + r4_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r

def Shuelike_rasymmetric_simple(parameters=[], coordinates=[], variables=False,
                                return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r3, r4, a0, a1 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r3_term = r3*sg_pos*np.sin(p)
    r4_term = r4*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)**2) * (r3_term + r4_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r

def ShuelikeAsymmetric(parameters=[], coordinates=[], variables=False,
                       return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r1, r2, r3, a0, a1 = parameters
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos*np.sin(p)
    r3_term = r3*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) + (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f

    return r

def ShuelikeAsymmetric_r1fixed(parameters=[], coordinates=[], variables=False,
                       return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r2, r3, a0, a1 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos*np.sin(p)
    r3_term = r3*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) + (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f

    return r

def ShuelikeAsymmetric_AsPerturbation(parameters=[], coordinates=[], variables=False,
                                      return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r1, r2, r3, a0, a1 = parameters
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos*np.sin(p)
    r3_term = r3*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r

def ShuelikeAsymmetric_AsPerturbation_r1fixed(parameters=[], coordinates=[], variables=False,
                                      return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r2, r3, a0, a1 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos*np.sin(p)
    r3_term = r3*sg_neg*np.sin(p)
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r

def ShuelikeAsymmetric_AsPerturbation_2(parameters=[], coordinates=[], variables=False,
                                        return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r1, r2, r3, a0, a1 = parameters
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = -(np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos * np.sin(p)**2
    r3_term = r3*sg_neg * np.sin(p)**2
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r


def ShuelikeAsymmetric_AsPerturbation_r1fixed_2(parameters=[], coordinates=[], variables=False,
                                      return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, r2, r3, a0, a1 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = -(np.sign(np.sin(p)) - 1)/2
    
    # r2_term = r2*np.cos(p)**2 * (r2_scale) + (1 - r2_scale)
    r2_term = r2*sg_pos * np.sin(p)**2
    r3_term = r3*sg_neg * np.sin(p)**2
    r_ss = r0 * ((p_dyn)**r1) 
    r_perturb = (np.sin(t/2)**2) * (r2_term + r3_term) * ((p_dyn)**r1) 
    
    a_f =  (a0 + a1 * p_dyn)
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f + r_perturb

    return r


def Shuelike_aasymmetric_r1fixed(parameters=[], coordinates=[], variables=False,
                                 return_r_ss:bool=False, return_a_f:bool=False):
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
    r0, a0, a1, a2, a3, a4 = parameters
    r1 = -0.25
    
    sg_pos = (np.sign(np.sin(p)) + 1)/2
    sg_neg = (np.sign(np.sin(p)) - 1)/2
    
    r_ss = r0 * ((p_dyn)**r1) 
    
    a2_term = 1 - (a2 * np.cos(p)**2)
    a3_term = 1 - (sg_pos * a3 * np.sin(p)**2)
    a4_term = 1 - (sg_neg * a4 * np.sin(p)**2)
    a_f =  (a0 + a1 * p_dyn) * a2_term * a3_term * a4_term
    
    if return_r_ss:
        return r_ss
    if return_a_f:
        return a_f
    
    # Calculate r
    r = r_ss * (2/(1 + np.cos(t)))**a_f

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
        return ('t', 'p', 'p_dyn'), 'r'
    
    t, p, p_dyn = coordinates
    r1, a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1 = parameters
    
    # x = x * 1/120
    # y = y * 1/120
    
    a = a0 + a1*p_dyn**(r1)
    b = b0 + b1*p_dyn**(r1)
    c = c0 + c1*p_dyn**(r1)
    d = d0 + d1*p_dyn
    e = e0 + e1*p_dyn
    f = f0 + f1*p_dyn
    
    A = c*np.cos(t)**2 + e*np.sin(t)**2*np.sin(p)**2 - f*np.sin(t)*np.cos(t)*np.sin(p) - np.sin(t)**2*np.cos(p)**2
    B = b*np.cos(t) - d*np.sin(t)*np.sin(p)
    C = a
    
    r = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    
    return np.array(r) * 120


def Joylike_r1fixed(parameters=[], coordinates=[], variables=False):
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
        return ('t', 'p', 'p_dyn'), 'r'
    
    t, p, p_dyn = coordinates
    a0, a1, b0, b1, c0, c1, d0, d1, e0, e1, f0, f1 = parameters
    r1 = -1/4
    
    # x = x * 1/120
    # y = y * 1/120
    
    a = a0 + a1*p_dyn**(r1)
    b = b0 + b1*p_dyn**(r1)
    c = c0 + c1*p_dyn**(r1)
    d = d0 + d1*p_dyn
    e = e0 + e1*p_dyn
    f = f0 + f1*p_dyn
    
    A = c*np.cos(t)**2 + e*np.sin(t)**2*np.sin(p)**2 - f*np.sin(t)*np.cos(t)*np.sin(p) - np.sin(t)**2*np.cos(p)**2
    B = b*np.cos(t) - d*np.sin(t)*np.sin(p)
    C = a
    
    r = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    
    return np.array(r) * 120


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
