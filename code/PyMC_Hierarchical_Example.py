#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:59:45 2024

@author: mrutala
"""

import warnings

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
#import theano.tensor as tt

from pymc import Model, Normal, Slice, sample, Uniform
from pymc.distributions import Interpolated
from scipy import stats
#from theano import as_op

#plt.style.use("seaborn-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")

warnings.filterwarnings("ignore")

# =============================================================================
# Generate Data
# =============================================================================
# Initialize random number generator
np.random.seed(93457)

# True parameter values
r0_true = 100
alpha0_true = 0.5
#log_p_dyn_true = -2

# Size of dataset
size = 100

# Predictor variable
theta = (np.linspace(90, 110, size) + np.random.randn(size)) * np.pi/180
#log_p_dyn = np.random.randn(size) * 0.2

# Simulate outcome variable
#Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

# Simulate heavily biased observations
R = r0_true * (2 / (1 + np.cos(theta)))**(alpha0_true) - np.abs(5*np.random.randn(size)) 

# =============================================================================
# Model specification
#   Our initial beliefs about the parameters are quite uninformative
# =============================================================================

basic_model = Model()

with basic_model:
    # Priors for unknown model parameters
    r0 = Uniform('r0', 30, 200) # Normal("alpha", mu=0, sigma=1)
    alpha0 = Uniform('alpha0', 0, 1) # Normal("beta0", mu=12, sigma=1)
    log_p_dyn = Uniform('log_p_dyn', -5, 1) # Normal("beta1", mu=18, sigma=1)

    # Expected value of outcome
    # alpha + beta0 * X1 + beta1 * X2
    mu = r0 * (2 / (1 + np.cos(theta)))**(alpha0) + np.random.randn(size)

    # Likelihood (sampling distribution) of observations
    # Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)
    R_obs = Normal('R_obs', mu = mu, sigma = 1, observed = R)

    # draw 1000 posterior samples
    trace = sample(1000)

az.plot_trace(trace)
plt.show()

breakpoint()

# Get Prior from Posterior
def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

#
traces = [trace]

#
for _ in range(10):
    # generate more data
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

    model = Model()
    with model:
        # Priors are posteriors from previous iteration
        posterior_from_trace = az.extract(trace, group='posterior')
        alpha = from_posterior("alpha", posterior_from_trace["alpha"].values)
        beta0 = from_posterior("beta0", posterior_from_trace["beta0"].values)
        beta1 = from_posterior("beta1", posterior_from_trace["beta1"].values)

        # Expected value of outcome
        mu = alpha + beta0 * X1 + beta1 * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

        # draw 10000 posterior samples
        trace = sample(1000)
        traces.append(trace)
        
# 
print("Posterior distributions after " + str(len(traces)) + " iterations.")
cmap = mpl.cm.autumn
for param in ["alpha", "beta0", "beta1"]:
    fig, ax = plt.subplots(figsize=(8, 2))
    for update_i, trace in enumerate(traces):
        from_trace = az.extract(trace, group='posterior')
        samples = from_trace[param].values
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        ax.plot(x, y, color=cmap(1 - update_i / len(traces)),
                label=str(update_i))
    ax.axvline({"alpha": alpha_true, "beta0": beta0_true, "beta1": beta1_true}[param], c="k")
    ax.set_ylabel("Frequency")
    ax.set_title(param)
    ax.legend()
    plt.show()

