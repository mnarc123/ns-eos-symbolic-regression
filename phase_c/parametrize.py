"""
Parametrized SR formulas for Phase C MCMC.

The tree structure (operators, nesting) is fixed from Phase B.
Only the numerical constants are free parameters for MCMC sampling.
"""

import numpy as np


def cs2_B_c15(x, theta):
    """
    c_s²(x) = tanh(x · tanh(a·x)) · tanh(exp(-tanh(x - b)))

    Parameters
    ----------
    x : array_like
        Normalized baryon density x = n_b / n_0.
    theta : array_like
        [a, b] where:
        - a: controls slope of rise (~0.04469)
        - b: position of high-density cutoff (~7.691)

    Returns
    -------
    cs2 : ndarray
    """
    a, b = theta
    x = np.asarray(x, dtype=np.float64)
    factor1 = np.tanh(x * np.tanh(a * x))
    factor2 = np.tanh(np.exp(-np.tanh(x - b)))
    return factor1 * factor2


def cs2_B_c17(x, theta):
    """
    c_s²(x) = tanh(a · x · (x + c)) · tanh(exp(-tanh(x - b)))

    Parameters
    ----------
    x : array_like
        Normalized baryon density x = n_b / n_0.
    theta : array_like
        [a, b, c] where:
        - a: slope parameter (~0.03631)
        - b: cutoff position (~8.0)
        - c: linear shift (~0.7405)

    Returns
    -------
    cs2 : ndarray
    """
    a, b, c = theta
    x = np.asarray(x, dtype=np.float64)
    factor1 = np.tanh(a * x * (x + c))
    factor2 = np.tanh(np.exp(-np.tanh(x - b)))
    return factor1 * factor2


# Best-fit parameter values from Phase B symbolic regression
THETA_0 = {
    'B_c15': np.array([0.044691067, 7.691]),
    'B_c17': np.array([0.036304515, 8.0, 0.7405348]),
}

FORMULAS = {
    'B_c15': cs2_B_c15,
    'B_c17': cs2_B_c17,
}

PARAM_NAMES = {
    'B_c15': [r'$a$', r'$b$'],
    'B_c17': [r'$a$', r'$b$', r'$c$'],
}
