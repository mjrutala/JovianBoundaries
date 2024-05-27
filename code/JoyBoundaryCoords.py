#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:52:22 2024

@author: mrutala
"""
def find_JoyBowShock(p_dyn, x=False, y=False, z=False):
    return find_JoyBoundaries(p_dyn, boundary='BS', x=x, y=y, z=z)
    
def find_JoyMagnetopause(p_dyn, x=False, y=False, z=False):
    return find_JoyBoundaries(p_dyn, boundary='MP', x=x, y=y, z=z)


def get_JoyCoefficients(p=0, boundary='BS', function=False):
    
    #   Joy+ 2002 coefficients
    match boundary.lower():
        case ('bs' | 'bowshock' | 'bow shock'):
            A0, A1 = -1.107, +1.591
            B0, B1 = -0.566, -0.812
            C0, C1 = +0.048, -0.059
            D0, D1 = +0.077, -0.038
            E0, E1 = -0.874, -0.299
            F0, F1 = -0.055, +0.124
        case ('mp' | 'magnetopause'):
            A0, A1 = -0.134, +0.488
            B0, B1 = -0.581, -0.225
            C0, C1 = -0.186, -0.016
            D0, D1 = -0.014, +0.096
            E0, E1 = -0.814, -0.811
            F0, F1 = -0.050, +0.168
        case _:
            return
        
    A = lambda p: (A0 + A1*p**(-1/4))
    B = lambda p: (B0 + B1*p**(-1/4))
    C = lambda p: (C0 + C1*p**(-1/4))
    D = lambda p: (D0 + D1*p)
    E = lambda p: (E0 + E1*p)
    F = lambda p: (F0 + F1*p)
    
    if function:
        result = {'A':A, 'B':B, 'C':C, 'D':D, 'E':E, 'F':F}
    else:
        result = {'A':A(p), 'B':B(p), 'C':C(p), 'D':D(p), 'E':E(p), 'F':F(p)}
    return result

def find_JoyBoundaries(p_dyn, boundary='BS', x=False, y=False, z=False):
    import numpy as np
    import warnings
    
    def fxn():
        warnings.warn("runtime", RuntimeWarning)
        
    #   Joy+ 2002 coefficients
    A, B, C, D, E, F = get_JoyCoefficients(p_dyn, boundary=boundary).values()
    scale_factor = 1/120.
    
    #   z**2 = A + B*x + C*x**2 + D*y + E*y**2 + F*x*y
    match x, y, z:    
        case x, y, bool():
            #   0 = 1 z^2 + 0z + (A + Bx + Cx^2 + Dy + Ey^2 + Fxy)
            if type(x) in [int, float]:    
                plane, abcissa = 'yz', 'y'
            else:
                plane, abcissa = 'xz', 'x'
            xs, ys = x * scale_factor, y * scale_factor
            aq = -1
            bq = 0
            cq = A + B*xs + C*xs**2 + D*ys + E*ys**2 + F*xs*ys
            
        case x, bool(), z:
            #   0 = E y^2 + (D + Fx) y + (A + Bx + Cx^2 - z^2)
            if type(x) in [int, float]:
                plane, abcissa = 'yz', 'z' 
            else:
                plane, abcissa = 'xy', 'x' 
            xs, zs = x * scale_factor, z * scale_factor
            aq = E
            bq = D + F*xs
            cq = A + B*xs + C*xs**2 - zs**2
        
        case bool(), y, z:
            #   0 = C x^2 + (B + Fy)x + (A + Dy + Ey^2 - z^2)
            if type(y) in [int, float]:
                plane, abcissa = 'xz', 'z'
            else:
                plane, abcissa = 'xy', 'y'
            ys, zs = y * scale_factor, z * scale_factor
            aq = C
            bq = B + F*ys
            cq = A + D*ys + E*ys**2 - zs**2
    
    #   Ignore math warnings
    with warnings.catch_warnings(action="ignore"):
        fxn()
        
        result_plus = (-bq + np.sqrt(bq**2 - 4*aq*cq))/(2*aq)
        result_minus = (-bq - np.sqrt(bq**2 - 4*aq*cq))/(2*aq)
        
    return np.array([result_plus, result_minus]) / scale_factor

def find_JoyPressures(x, y, z, boundary='BS'):
    import numpy as np
    from scipy.optimize import minimize_scalar
    
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    A, B, C, D, E, F = get_JoyCoefficients(boundary=boundary, function=True).values()
    
    scale_factor = 1/120.
    
    resulting_pressures = []
    for xi, yi, zi in zip(x, y, z):
        xi = xi*scale_factor
        yi = yi*scale_factor
        zi = zi*scale_factor
        
        def Joy2002(p):
            function = A(p) + B(p)*xi + C(p)*xi**2 + D(p)*yi + E(p)*yi**2 + F(p)*xi*yi - zi**2
            return abs(function)
    
        res = minimize_scalar(Joy2002, bounds=(1e-5, 1e1), method='bounded')
        
        resulting_pressures.append(res.x)
    breakpoint()
        
    