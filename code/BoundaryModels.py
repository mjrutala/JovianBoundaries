#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:53 2024

@author: mrutala
"""
import numpy as np

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
    y = r * np.sin(t) * np.sin(p)
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
    r0, r1, r2, a0, a1, a2 = parameters
    
    # with np.errstate(invalid='raise'):
    #     try: 
    #         r_0 = r0*(p_dyn**r1)
    #     except RuntimeError():
    #         print('You caught the error!')
    r_0 = r0*((p_dyn)**r1) + r2*np.sin(p)**2
    
    a_0 = (a0 + a1 * p_dyn**a2)# * (1 + a3*np.sin(p) + a4*np.sin(-p) + a5*np.cos(p)**2)
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
    return r

def Shuelike_Square(parameters=[], coordinates=[], variables=False):
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
    r0, r1, a0, a1, a2, a3, a4 = parameters
    
    # with np.errstate(invalid='raise'):
    #     try: 
    #         r_0 = r0*(p_dyn**r1)
    #     except RuntimeError():
    #         print('You caught the error!')
    r_0 = r0*((p_dyn/0.003)**r1)
    
    a_0 = (a0 + a1 * p_dyn**a2) * (1 + a3*np.sign(np.sin(p)+1)*np.sin(p) + a4*np.sign(np.sin(-p)+1)*np.sin(-p))
    
    r = r_0 * (2/(1 + np.cos(t)))**a_0
    #rho, phi, ell = convert_SphericalSolarToCylindricalSolar(r, t, p)
    
    #rho = np.interp(coordinates[2], ell, rho, left=np.nan, right=np.nan)
    
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
def explore_Variation(form, ranges):
    import matplotlib.pyplot as plt
    
    naxs = len(ranges)
    nrows = int(np.floor(np.sqrt(naxs)))
    ncols = int(np.ceil(naxs / nrows))
    
    fig, axs = plt.subplots(figsize=(6,4), nrows=nrows, ncols=ncols)
    
    t = np.linspace(0, 180, 1000) * np.pi/180.
    p = np.zeros(1000)
    p_dyn = np.zeros(1000) + 0.01
    
    for i in range(naxs):
        
        test_param_values = np.linspace(ranges[i], 5)
        for test_param_value in test_param_values:
            r = form([test_param_value, ])
            
        axs[i].plot
    
    
    breakpoint()
    
    return

