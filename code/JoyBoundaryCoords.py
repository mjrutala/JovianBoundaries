#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:52:22 2024

@author: mrutala
"""

def find_JoyBowShock(p_dyn, x=False, y=False, z=False):
    import numpy as np
    import warnings
    
    def fxn():
        warnings.warn("runtime", RuntimeWarning)
        

        
    #   Joy+ 2002 coefficients
    A = -1.107 + 1.591 * p_dyn**(-1/4)
    B = -0.566 - 0.812 * p_dyn**(-1/4)
    C = 0.048 - 0.059*p_dyn**(-1/4)
    
    D = 0.077 - 0.038 * p_dyn
    E = -0.874 - 0.299 * p_dyn
    F = -0.055 + 0.124 * p_dyn
    
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
        
    