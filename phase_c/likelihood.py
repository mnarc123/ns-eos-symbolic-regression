"""
Composite observational likelihood for Phase C MCMC.

L(theta) = L_NICER(theta) + L_GW(theta) + L_Mmax(theta)

For each parameter vector theta:
  theta -> cs2(x) -> mu(n) -> eps(n), P(n) -> crust join -> TOV -> M(R), Lambda(M)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from physics.units import N_0, M_NUCLEON, MEV_FM3_TO_GEO, M_SUN_KM
from physics.tov_solver import find_radius_at_mass
from phase_c.tidal import solve_tov_love, combined_tidal_deformability


# =========================================================================
# EoS reconstruction from parametrized cs2
# =========================================================================

def eos_from_theta(cs2_func, theta, crust):
    """
    Reconstruct full EoS (crust + core) from parameter vector theta.

    Parameters
    ----------
    cs2_func : callable
        cs2(x, theta) where x = n_b / n_0.
    theta : array_like
        Parameter vector.
    crust : dict
        Crust data from phase_c.crust.load_crust_from_sly4().

    Returns
    -------
    eos : dict with 'n_b', 'P', 'epsilon' (arrays, MeV/fm^3 units)
        or None if EoS is unphysical.
    """
    # Matching point at end of crust
    n_match = crust['n_b'][-1]
    mu_match = crust['mu'][-1]
    eps_match = crust['epsilon'][-1]

    # Core density grid: from matching point to 8 * n_0
    n_core = np.geomspace(n_match, 8.0 * N_0, 500)
    x_core = n_core / N_0

    # Evaluate cs2 on core grid
    try:
        cs2 = cs2_func(x_core, theta)
    except Exception:
        return None

    cs2 = np.asarray(cs2, dtype=np.float64)

    # Physics check: stability and causality
    if np.any(cs2 < -0.01) or np.any(cs2 > 1.01):
        return None
    # Clip tiny violations from numerical noise
    cs2 = np.clip(cs2, 1e-6, 1.0)

    # mu(n) = mu_match * exp(integral cs2/n dn from n_match)
    integrand = cs2 / n_core
    integral = cumulative_trapezoid(integrand, n_core, initial=0.0)
    mu_core = mu_match * np.exp(integral)

    # epsilon(n) = eps_match + integral mu(n') dn' from n_match
    eps_core = eps_match + cumulative_trapezoid(mu_core, n_core, initial=0.0)

    # P(n) = n * mu - epsilon
    P_core = n_core * mu_core - eps_core

    # Check: P must be monotonically increasing
    if np.any(np.diff(P_core) < 0):
        return None

    # Check: P must be positive in the core
    if np.any(P_core[10:] <= 0):
        return None

    # Join crust + core (skip first core point to avoid duplication)
    n_full = np.concatenate([crust['n_b'], n_core[1:]])
    P_full = np.concatenate([crust['P'], P_core[1:]])
    eps_full = np.concatenate([crust['epsilon'], eps_core[1:]])

    # Ensure pressure continuity at junction
    # (small shift if needed)
    P_crust_end = crust['P'][-1]
    P_core_start = P_core[1]
    if P_core_start < P_crust_end:
        # Shift core P up to be continuous
        dP = P_crust_end - P_core_start + 1e-6
        idx_core_start = len(crust['P'])
        P_full[idx_core_start:] += dP

    return {
        'n_b': n_full,
        'P': P_full,
        'epsilon': eps_full,
    }


def compute_observables(eos):
    """
    Compute M-R-Lambda sequence and key observables in a single TOV+Love pass.

    Returns
    -------
    obs : dict or None
        'mr_curve'      : ndarray (N, 3) [M, R, Lambda]
        'M_max'         : float [M_sun]
        'R_1.4'         : float [km] or None
        'R_2.0'         : float [km] or None
        'Lambda_of_M'   : callable, Lambda(M) interpolator on stable branch
    """
    if eos is None:
        return None

    n_b = eos['n_b']
    P = eos['P']
    eps = eos['epsilon']

    # Central pressure grid
    mask = (n_b > 0) & (P > 0)
    P_of_n = interp1d(np.log(n_b[mask]), np.log(P[mask]),
                      kind='cubic', fill_value='extrapolate')
    n_central = np.geomspace(1.0 * N_0, 8.0 * N_0, 50)
    Pc_arr = np.exp(P_of_n(np.log(n_central)))

    results = []
    for Pc in Pc_arr:
        M, R, k2, Lam = solve_tov_love(P, eps, Pc)
        if M is not None and 0.3 < M < 3.0 and R is not None:
            results.append([M, R, Lam if Lam is not None else 0.0])

    if len(results) < 5:
        return None

    mr = np.array(results)  # columns: M, R, Lambda
    M_max = mr[:, 0].max()

    # Stable branch: up to M_max
    idx_max = np.argmax(mr[:, 0])
    M_stable = mr[:idx_max + 1, 0]
    R_stable = mr[:idx_max + 1, 1]
    Lam_stable = mr[:idx_max + 1, 2]

    R_14 = None
    R_20 = None
    Lambda_of_M = None

    if len(M_stable) > 3:
        try:
            R_of_M = interp1d(M_stable, R_stable, kind='linear',
                              bounds_error=False, fill_value=np.nan)
            Lam_of_M = interp1d(M_stable, Lam_stable, kind='linear',
                                bounds_error=False, fill_value=np.nan)
            if M_stable.min() <= 1.4 <= M_stable.max():
                R_14 = float(R_of_M(1.4))
            if M_stable.min() <= 2.0 <= M_stable.max():
                R_20 = float(R_of_M(2.0))
            Lambda_of_M = Lam_of_M
        except Exception:
            pass

    return {
        'mr_curve': mr,
        'M_max': M_max,
        'R_1.4': R_14,
        'R_2.0': R_20,
        'Lambda_of_M': Lambda_of_M,
    }


# =========================================================================
# NICER likelihood via Gaussian approximation
# =========================================================================

# Published NICER measurements (Gaussian approximation of posteriors)
NICER_PULSARS = {
    'J0030': {
        'M': 1.44, 'sigma_M': 0.15,
        'R': 12.45, 'sigma_R': 0.65,
        'ref': 'Vinciguerra+2024',
    },
    'J0740': {
        'M': 2.073, 'sigma_M': 0.069,
        'R': 12.49, 'sigma_R': 1.08,
        'ref': 'Salmi+2024',
    },
    'J0437': {
        'M': 1.418, 'sigma_M': 0.044,
        'R': 11.36, 'sigma_R': 0.95,
        'ref': 'Choudhury+2024',
    },
}


def log_likelihood_nicer(obs):
    """
    NICER log-likelihood: Gaussian in R at the measured mass.

    For each pulsar, compare the predicted R(M_pulsar) from the EoS
    with the NICER measurement.
    """
    if obs is None:
        return -np.inf

    mr = obs['mr_curve']  # columns: M, R, Lambda
    ll = 0.0

    # Build R(M) interpolator on stable branch
    idx_max = np.argmax(mr[:, 0])
    M_stable = mr[:idx_max + 1, 0]
    R_stable = mr[:idx_max + 1, 1]

    if len(M_stable) < 3:
        return -np.inf

    try:
        R_of_M = interp1d(M_stable, R_stable, kind='linear',
                          bounds_error=False, fill_value=np.nan)
    except Exception:
        return -np.inf

    for name, data in NICER_PULSARS.items():
        M_psr = data['M']
        R_pred = R_of_M(M_psr)

        if np.isnan(R_pred):
            return -np.inf

        # Gaussian likelihood in R
        R_obs = data['R']
        sigma_R = data['sigma_R']
        ll += -0.5 * ((R_pred - R_obs) / sigma_R) ** 2

    return ll


# =========================================================================
# GW170817 likelihood
# =========================================================================

GW170817 = {
    'M_chirp': 1.186,  # M_sun
    'q_median': 0.87,  # mass ratio (median)
    'Lambda_tilde_mean': 300.0,
    'Lambda_tilde_sigma': 190.0,  # Gaussian approximation
}


def log_likelihood_gw(obs):
    """
    GW170817 tidal deformability likelihood.

    Uses the Lambda(M) interpolator already computed from the TOV+Love scan.
    """
    if obs is None:
        return -np.inf

    Lambda_of_M = obs.get('Lambda_of_M')
    if Lambda_of_M is None:
        return 0.0

    q = GW170817['q_median']
    M_chirp = GW170817['M_chirp']
    M_tot = M_chirp / (q / (1 + q)**2)**(3.0 / 5.0)
    M1 = M_tot / (1 + q)
    M2 = M_tot * q / (1 + q)

    try:
        Lambda1 = float(Lambda_of_M(M1))
        Lambda2 = float(Lambda_of_M(M2))
    except (ValueError, IndexError):
        return 0.0

    if np.isnan(Lambda1) or np.isnan(Lambda2):
        return 0.0

    Lambda_tilde = combined_tidal_deformability(Lambda1, Lambda2, q)

    if Lambda_tilde < 0 or Lambda_tilde > 5000:
        return -np.inf

    mean = GW170817['Lambda_tilde_mean']
    sigma = GW170817['Lambda_tilde_sigma']
    return -0.5 * ((Lambda_tilde - mean) / sigma) ** 2


# =========================================================================
# Maximum mass likelihood
# =========================================================================

def log_likelihood_mmax(obs):
    """
    Maximum mass constraint from radio timing.

    Hard cut: M_max >= 1.97 M_sun (conservative, from PSR J0348+0432)
    Soft penalty: Gaussian around 2.08 M_sun (PSR J0740+6620)
    """
    if obs is None:
        return -np.inf

    M_max = obs['M_max']

    # Hard cut
    if M_max < 1.97:
        return -np.inf

    # Soft Gaussian penalty: M_max ~ 2.08 +/- 0.07
    return -0.5 * max(0.0, (2.08 - M_max) / 0.07) ** 2


# =========================================================================
# Total log-likelihood
# =========================================================================

def log_likelihood_total(theta, cs2_func, crust):
    """
    Total observational log-likelihood.

    Parameters
    ----------
    theta : array_like
        Parameter vector for the SR formula.
    cs2_func : callable
        cs2(x, theta).
    crust : dict
        Crust EoS data.

    Returns
    -------
    ll : float
        Total log-likelihood (sum of NICER + GW + M_max components).
    """
    # 1. Reconstruct EoS
    eos = eos_from_theta(cs2_func, theta, crust)
    if eos is None:
        return -np.inf

    # 2. Compute M-R observables
    obs = compute_observables(eos)
    if obs is None:
        return -np.inf

    # 3. M_max constraint (cheapest, check first)
    ll_mmax = log_likelihood_mmax(obs)
    if not np.isfinite(ll_mmax):
        return -np.inf

    # 4. NICER constraint
    ll_nicer = log_likelihood_nicer(obs)
    if not np.isfinite(ll_nicer):
        return -np.inf

    # 5. GW170817 constraint (uses Lambda from same TOV+Love scan)
    ll_gw = log_likelihood_gw(obs)

    total = ll_mmax + ll_nicer + ll_gw

    return total if np.isfinite(total) else -np.inf
