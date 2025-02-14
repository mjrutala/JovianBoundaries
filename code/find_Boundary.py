def find_Boundary(boundary='MP', x=False, y=False, z=False, p_sw=False, 
                  guess=1):
    """
    This function takes 3 of (x, y, z, p_sw) and estimates the 4th,
    beginning the search at 'guess'

    Parameters
    ----------
    boundary : str
        A string defining the boundary of interest. 
        By default, the magnetopause.
    x : floatlike or bool, optional
        JSS x-coordinate of the point of interest.
        Leave out or set to TRUE to estimate x.
    y : floatlike or bool, optional
        JSS y-coordinate of the point of interest.
        Leave out or set to TRUE to estimate y.
    z : floatlike or bool, optional
        JSS z-coordinate of the point of interest.
        Leave out or set to TRUE to estimate z.
    p_sw : floatlike or bool, optional
        Solar wind (dynamic) pressure at the point of interest.
        Leave out or set to TRUE to estimate p_sw.
    guess : floatlike, optional
        Initial guess for estimating the result. The boundary functions are
        typically double-valued; the value closest to initial guess will be 
        returned.
        The default is 1.

    Returns
    -------
    list
        Returns a list containing the estimated boundary coordinate.
        Spatial coordinates are normally distributed, and will return:
            [mean, std]
        Pressure coordinates are lognormally distributed, and will return:
            [mean, -std, +std]

    """
    import numpy as np
    from scipy.optimize import fsolve
    
    # If the function was called incorrectly, print the docstring
    bool_count = 0
    for v in [x, y, z, p_sw]:
        if type(v) is bool:
            bool_count += 1
    if bool_count > 1:
        print(find_Boundary.__doc__)
        return
    
    # The uncertainties are found by Monte Carlo, 
    # with random seed (rng), size (n), and samples (gauss_factor)
    rng = np.random.default_rng()
    n = 500 
    gauss_factor = rng.normal(0, 1, n)
    
    # Choose the appropriate model parameter vector (MP or BS)
    if boundary.lower() in ['mp', 'magnetopause']:
        r0 = rng.normal(33.5, 0.0, n)
        r1 = np.full(n, -0.25)
        r2 = rng.normal(12.9, 0.0, n)
        r3 = rng.normal(25.8, 0.0, n)
        a0 = rng.normal(0.19, 0.00, n)
        a1 = rng.normal(1.27, 0.00, n)
        
        sigma_b = rng.normal(16.8, 0.8, n)
        sigma_m = rng.normal(0.15, 0.01, n)
        
        C = [r0, r1, r2, r3, a0, a1]
        sigma = [sigma_b, sigma_m]
    elif boundary.lower() in ['bs', 'bow shock', 'bow_shock', 'bowshock']:
        r0 = rng.normal(36.4, 0.0, n)
        r1 = np.full(n, -0.25)
        r2 = rng.normal(0.0, 0.0, n)
        r3 = rng.normal(9.9, 0.0, n)
        a0 = rng.normal(0.89, 0.00, n)
        a1 = rng.normal(0.88, 0.00, n)
        
        sigma_b = rng.normal(13.3, 0.8, n)
        sigma_m = rng.normal(0.20, 0.01, n)
        
        C = [r0, r1, r2, r3, a0, a1]
        sigma = [sigma_b, sigma_m]
    
    # Determine which coordinate to solve for
    if type(x) is bool:
        fn_r = lambda x: np.sqrt(x**2 + y**2 + z**2)
        fn_theta = lambda x: np.arctan2(np.sqrt(y**2 + z**2), x)
        fn_phi = lambda x: np.arctan2(-y, z)
        fn_log_p_sw = lambda x: np.log10(p_sw)
        
    elif type(y) is bool:
        fn_r = lambda y: np.sqrt(x**2 + y**2 + z**2)
        fn_theta = lambda y: np.arctan2(np.sqrt(y**2 + z**2), x)
        fn_phi = lambda y: np.arctan2(-y, z)
        # dependent = lambda: y
        fn_log_p_sw = lambda y: np.log10(p_sw)
        
    elif type(z) is bool:
        fn_r = lambda z: np.sqrt(x**2 + y**2 + z**2)
        fn_theta = lambda z: np.arctan2(np.sqrt(y**2 + z**2), x)
        fn_phi = lambda z: np.arctan2(-y, z)
        fn_log_p_sw = lambda z: np.log10(p_sw)
        
    elif type(p_sw) is bool:
        fn_r = lambda p_sw: np.sqrt(x**2 + y**2 + z**2)
        fn_theta = lambda p_sw: np.arctan2(np.sqrt(y**2 + z**2), x)
        fn_phi = lambda p_sw: np.arctan2(-y, z)
        fn_log_p_sw = lambda log_p_sw: log_p_sw
        
        guess = np.log10(guess)
    
    # This inner function returns the distance to the boundary (r_b)
    def r_b(dependent):
        
        X = [fn_theta(dependent), 
             fn_phi(dependent), 
             10.**fn_log_p_sw(dependent)]
        
        r_ss = C[0] * X[2]**C[1]
        alpha_f = C[4] + C[5] * X[2]
        
        
        pass_positive_phi = 0.5*np.sign(X[1])+0.5
        pass_negative_phi = -0.5*np.sign(X[1])+0.5
        
        r_b_prime = \
            pass_positive_phi * (np.sin(X[0]/2)**2) * (C[2] * np.sin(X[1])**2) * X[2]**C[1] + \
            pass_negative_phi * (np.sin(X[0]/2)**2) * (C[3] * np.sin(X[1])**2) * X[2]**C[1]
            
        r_b = r_ss * (2/(1 + np.cos(X[0])))**alpha_f + r_b_prime
        
        r_b = r_b + (sigma[0] + sigma[1]*r_b)*gauss_factor
        
        return r_b
    
    # This inner function returns the difference between r_b and the point
    def minimization(dependent):
        return r_b(dependent) - np.full(n, fn_r(dependent))
    
    # The best solution should minimize 'minimization' (i.e., ~0)
    solutions = fsolve(minimization, np.full(n, guess))
    # Check that solutions yield near-zero residuals
    check = np.isclose(minimization(solutions), 0, atol=1e-1, rtol=0)
    
    # Unless y == z == 0, in which case r_b is independent of x
    # then fsolve is unable to solve, and the trivial solution is
    if y == z == 0.0:
        solutions = r_b(0)
        check = np.full(n, True)
    
    # For pressure, return lognormal uncertainties in linear space;
    # otherwise, return normal uncertainties
    if type(p_sw) is bool:
        result = [10.**np.mean(solutions),
                  10.**np.mean(solutions) - 10.**(np.mean(solutions) - np.std(solutions)),
                  10.**(np.mean(solutions) + np.std(solutions)) - 10.**np.mean(solutions)]
    else:
        result = (np.mean(solutions), np.std(solutions))
    
    # If minimization has not been successfully minimized, return NaNs
    if check.all() == False:
        result = [np.nan for element in result]

    return result