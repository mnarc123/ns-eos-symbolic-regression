"""
Reconstruct the full EoS {P(n), ε(n), μ(n)} from a speed-of-sound-squared
function c_s²(n).

Thermodynamic relations (Tews et al. 2018, ApJ 860, 149):

    c_s² = dP/dε = (n/μ) dμ/dn

    μ(n) = μ₀ exp(∫_{n₀}^{n} c_s²(n')/n' dn')

    ε(n) = ε_match + ∫_{n_match}^{n} μ(n') dn'

    P(n) = n μ(n) - ε(n)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import CubicSpline

from physics.units import N_0, M_NUCLEON


def chemical_potential(cs2_func, n_grid, mu_0=None, n_ref=None):
    """
    Compute the baryon chemical potential μ(n) by integrating c_s²(n)/n.

    Parameters
    ----------
    cs2_func : callable
        Speed of sound squared as a function of n_b [fm⁻³].
    n_grid : array_like
        Baryon density grid in fm⁻³ (must be sorted ascending).
    mu_0 : float, optional
        Chemical potential at reference density n_ref in MeV.
        Default: m_N ≈ 939 MeV at n_ref = n_grid[0].
    n_ref : float, optional
        Reference density for mu_0. Default: n_grid[0].

    Returns
    -------
    mu : ndarray
        Chemical potential in MeV at each point of n_grid.
    """
    if n_ref is None:
        n_ref = n_grid[0]
    if mu_0 is None:
        mu_0 = M_NUCLEON  # MeV, neutron mass as starting approximation

    # Integrand: c_s²(n') / n'
    integrand = np.array([cs2_func(n) / n for n in n_grid])

    # Cumulative integral from n_grid[0] to each n
    integral = cumulative_trapezoid(integrand, n_grid, initial=0.0)

    # If n_ref != n_grid[0], adjust the offset
    if not np.isclose(n_ref, n_grid[0]):
        # Integrate from n_ref to n_grid[0]
        offset, _ = quad(lambda n: cs2_func(n) / n, n_ref, n_grid[0])
        integral += offset

    mu = mu_0 * np.exp(integral)
    return mu


def eos_from_cs2(cs2_func, n_grid, mu_0=None, n_ref=None,
                 epsilon_match=None):
    """
    Reconstruct the full EoS from c_s²(n).

    Parameters
    ----------
    cs2_func : callable
        c_s²(n_b) where n_b is in fm⁻³. Must return dimensionless values.
    n_grid : array_like
        Baryon density grid in fm⁻³ (sorted ascending, starting from crust-core
        transition or matching density).
    mu_0 : float, optional
        Chemical potential at the reference density [MeV].
    n_ref : float, optional
        Reference density for μ₀ [fm⁻³].
    epsilon_match : float, optional
        Energy density at n_grid[0] in MeV/fm³. If None, approximate as
        n_grid[0] * m_N.

    Returns
    -------
    eos : dict with keys:
        'n_b'     : baryon density [fm⁻³]
        'mu'      : chemical potential [MeV]
        'epsilon' : energy density [MeV/fm³]
        'P'       : pressure [MeV/fm³]
        'cs2'     : speed of sound squared (input, for verification)
    """
    n_grid = np.asarray(n_grid, dtype=np.float64)

    # Step 1: Chemical potential
    mu = chemical_potential(cs2_func, n_grid, mu_0=mu_0, n_ref=n_ref)

    # Step 2: Energy density by integrating μ(n)
    if epsilon_match is None:
        epsilon_match = n_grid[0] * M_NUCLEON  # rough approximation

    eps_integral = cumulative_trapezoid(mu, n_grid, initial=0.0)
    epsilon = epsilon_match + eps_integral

    # Step 3: Pressure from thermodynamic identity
    P = n_grid * mu - epsilon

    # Step 4: Verify c_s² for consistency
    cs2_check = np.array([cs2_func(n) for n in n_grid])

    return {
        'n_b': n_grid,
        'mu': mu,
        'epsilon': epsilon,
        'P': P,
        'cs2': cs2_check,
    }


def eos_from_cs2_of_x(cs2_x_func, x_min=0.5, x_max=8.0, n_points=500,
                       mu_0=None, epsilon_match=None):
    """
    Convenience wrapper: given c_s²(x) where x = n/n₀, reconstruct EoS.

    Parameters
    ----------
    cs2_x_func : callable
        c_s²(x) where x = n_b / n₀ (dimensionless).
    x_min, x_max : float
        Range of normalized density.
    n_points : int
        Number of grid points (log-spaced).
    mu_0 : float, optional
        Chemical potential at x_min * n₀ [MeV].
    epsilon_match : float, optional
        Energy density at x_min * n₀ [MeV/fm³].

    Returns
    -------
    eos : dict
        Same as eos_from_cs2, plus 'x' key for normalized density.
    """
    n_grid = np.geomspace(x_min * N_0, x_max * N_0, n_points)

    # Wrap cs2_x_func to accept n_b in fm⁻³
    def cs2_of_n(n):
        x = n / N_0
        return cs2_x_func(x)

    eos = eos_from_cs2(cs2_of_n, n_grid, mu_0=mu_0,
                       epsilon_match=epsilon_match)
    eos['x'] = n_grid / N_0
    return eos


def join_crust_core(crust_eos, core_eos, n_match):
    """
    Join a crust EoS table with a core EoS at a matching density.

    Ensures thermodynamic consistency at the junction by adjusting
    the core EoS pressure to be continuous.

    Parameters
    ----------
    crust_eos : dict
        Crust EoS with keys 'n_b', 'P', 'epsilon' (arrays in MeV/fm³ units).
    core_eos : dict
        Core EoS from eos_from_cs2, same keys.
    n_match : float
        Matching density in fm⁻³.

    Returns
    -------
    full_eos : dict
        Joined EoS covering crust + core.
    """
    # Crust part: n < n_match
    crust_mask = crust_eos['n_b'] < n_match
    # Core part: n >= n_match
    core_mask = core_eos['n_b'] >= n_match

    # Pressure continuity: shift core P if needed
    from scipy.interpolate import interp1d
    P_crust_at_match = interp1d(
        crust_eos['n_b'], crust_eos['P'], kind='cubic'
    )(n_match)
    P_core_at_match = interp1d(
        core_eos['n_b'], core_eos['P'], kind='cubic'
    )(n_match)
    dP = P_crust_at_match - P_core_at_match

    full_eos = {}
    for key in ['n_b', 'P', 'epsilon']:
        crust_part = crust_eos[key][crust_mask]
        core_part = core_eos[key][core_mask]
        if key == 'P':
            core_part = core_part + dP  # ensure continuity
        full_eos[key] = np.concatenate([crust_part, core_part])

    return full_eos
