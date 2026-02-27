"""
Prior distributions for Phase C MCMC.

Uniform (uninformative) priors within physically motivated bounds.
"""

import numpy as np


def log_prior_B_c15(theta):
    """
    Prior for B_c15: theta = [a, b]
    - a in [0.01, 0.15]   (slope: centred on ~0.045)
    - b in [4.0, 15.0]    (cutoff: centred on ~7.7)
    """
    a, b = theta
    if not (0.01 < a < 0.15):
        return -np.inf
    if not (4.0 < b < 15.0):
        return -np.inf
    return 0.0


def log_prior_B_c17(theta):
    """
    Prior for B_c17: theta = [a, b, c]
    - a in [0.01, 0.10]   (slope: centred on ~0.036)
    - b in [4.0, 15.0]    (cutoff: centred on ~8.0)
    - c in [-2.0, 5.0]    (shift: centred on ~0.74)
    """
    a, b, c = theta
    if not (0.01 < a < 0.10):
        return -np.inf
    if not (4.0 < b < 15.0):
        return -np.inf
    if not (-2.0 < c < 5.0):
        return -np.inf
    return 0.0


PRIORS = {
    'B_c15': log_prior_B_c15,
    'B_c17': log_prior_B_c17,
}
