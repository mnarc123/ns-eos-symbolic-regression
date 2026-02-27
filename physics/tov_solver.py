"""
Tolman-Oppenheimer-Volkoff (TOV) equation solver.

Solves the structure equations for a spherically symmetric, static
neutron star in general relativity:

    dm/dr = 4π r² ε(r)
    dP/dr = -(ε + P)(m + 4π r³ P) / [r(r - 2m)]

with the boundary conditions m(0) = 0, P(0) = P_central.
Integration proceeds outward until P = 0 (stellar surface).

All internal calculations use geometric units (G = c = 1):
    - lengths in km
    - masses in km (M_☉ = 1.4766 km)
    - P, ε in km⁻²

Input EoS tables are expected in MeV/fm³ and converted internally.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from physics.units import MEV_FM3_TO_GEO, M_SUN_KM


def _build_eos_interpolator(P_mev, eps_mev):
    """
    Build ε(P) interpolator in geometric units using log-log spline.

    Parameters
    ----------
    P_mev : array_like
        Pressure in MeV/fm³ (must be monotonically increasing).
    eps_mev : array_like
        Energy density in MeV/fm³.

    Returns
    -------
    eps_of_P : callable
        Function ε(P) in geometric units (km⁻²).
    P_min, P_max : float
        Valid pressure range in geometric units.
    """
    P_geo = np.asarray(P_mev, dtype=np.float64) * MEV_FM3_TO_GEO
    eps_geo = np.asarray(eps_mev, dtype=np.float64) * MEV_FM3_TO_GEO

    # Filter to positive values only
    mask = (P_geo > 0) & (eps_geo > 0)
    P_geo = P_geo[mask]
    eps_geo = eps_geo[mask]

    # Ensure monotonicity in P
    sort_idx = np.argsort(P_geo)
    P_geo = P_geo[sort_idx]
    eps_geo = eps_geo[sort_idx]

    # Remove duplicates
    _, unique_idx = np.unique(P_geo, return_index=True)
    P_geo = P_geo[unique_idx]
    eps_geo = eps_geo[unique_idx]

    log_P = np.log(P_geo)
    log_eps = np.log(eps_geo)

    interp_log = interp1d(log_P, log_eps, kind='cubic',
                          fill_value='extrapolate')

    def eps_of_P(P):
        if P <= 0:
            return 0.0
        try:
            return np.exp(interp_log(np.log(P)))
        except (ValueError, RuntimeWarning):
            return 0.0

    return eps_of_P, P_geo.min(), P_geo.max()


def _tov_rhs(r, y, eps_of_P):
    """
    Right-hand side of the TOV equations.

    Parameters
    ----------
    r : float
        Radial coordinate [km].
    y : array_like
        [m, P] - enclosed mass [km] and pressure [km⁻²].
    eps_of_P : callable
        Energy density as function of pressure (geometric units).
    """
    m, P = y

    if P <= 0:
        return [0.0, 0.0]

    eps = eps_of_P(P)

    if r < 1e-10:
        return [0.0, 0.0]

    denom = r * (r - 2.0 * m)
    if denom <= 0:
        return [0.0, 0.0]

    dm_dr = 4.0 * np.pi * r**2 * eps
    dP_dr = -(eps + P) * (m + 4.0 * np.pi * r**3 * P) / denom

    return [dm_dr, dP_dr]


def solve_tov(P_mev, eps_mev, P_central_mev, r_max=50.0):
    """
    Solve TOV equations for a single central pressure.

    Parameters
    ----------
    P_mev : array_like
        Pressure table [MeV/fm³].
    eps_mev : array_like
        Energy density table [MeV/fm³].
    P_central_mev : float
        Central pressure [MeV/fm³].
    r_max : float
        Maximum integration radius [km].

    Returns
    -------
    M : float or None
        Gravitational mass [M☉]. None if integration fails.
    R : float or None
        Stellar radius [km]. None if integration fails.
    """
    eps_of_P, P_min, P_max = _build_eos_interpolator(P_mev, eps_mev)
    Pc = P_central_mev * MEV_FM3_TO_GEO

    if Pc <= 0:
        return None, None

    # Initial conditions at small r₀
    r0 = 1e-4  # km
    eps_c = eps_of_P(Pc)
    if eps_c <= 0:
        return None, None

    m0 = (4.0 / 3.0) * np.pi * r0**3 * eps_c
    y0 = [m0, Pc]

    # Surface event: P drops to a small fraction of central pressure
    P_surface = Pc * 1e-12

    def surface_event(r, y, eos):
        return y[1] - P_surface
    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(
        _tov_rhs, [r0, r_max], y0, args=(eps_of_P,),
        method='RK45', rtol=1e-8, atol=1e-12,
        events=surface_event, max_step=0.1,
        dense_output=False,
    )

    if sol.t_events[0].size > 0:
        R = sol.t_events[0][0]
        M_geo = sol.y_events[0][0][0]
        M = M_geo / M_SUN_KM
        return M, R
    elif sol.success:
        # Reached r_max without finding surface
        R = sol.t[-1]
        M_geo = sol.y[0, -1]
        M = M_geo / M_SUN_KM
        return M, R
    else:
        return None, None


def mass_radius_curve(P_mev, eps_mev, n_b_mev=None,
                      P_central_range=None, n_central_range=None,
                      n_points=100):
    """
    Compute a mass-radius sequence by varying the central pressure.

    Provide either P_central_range (direct) or n_central_range + n_b_mev
    (central densities mapped to pressures via interpolation).

    Parameters
    ----------
    P_mev : array_like
        Pressure table [MeV/fm³].
    eps_mev : array_like
        Energy density table [MeV/fm³].
    n_b_mev : array_like, optional
        Baryon density table [fm⁻³]. Needed if using n_central_range.
    P_central_range : array_like, optional
        Array of central pressures [MeV/fm³].
    n_central_range : array_like, optional
        Array of central baryon densities [fm⁻³].
    n_points : int
        Number of points if generating a default range.

    Returns
    -------
    results : ndarray, shape (N, 3) or (N, 2)
        Columns: [M (M☉), R (km)] or [n_c (fm⁻³), M (M☉), R (km)].
    """
    if P_central_range is None and n_central_range is None:
        # Default: span from ~1.5x to ~10x saturation density in pressure
        if n_b_mev is not None:
            from physics.units import N_0
            n_central_range = np.geomspace(1.0 * N_0, 8.0 * N_0, n_points)
        else:
            # Use pressure range directly
            P_valid = P_mev[P_mev > 0]
            P_central_range = np.geomspace(
                P_valid.min() * 10, P_valid.max() * 0.9, n_points
            )

    if n_central_range is not None and n_b_mev is not None:
        # Map n_central → P_central via interpolation
        mask = (n_b_mev > 0) & (P_mev > 0)
        P_of_n = interp1d(
            np.log(n_b_mev[mask]), np.log(P_mev[mask]),
            kind='cubic', fill_value='extrapolate'
        )
        P_central_range = np.exp(P_of_n(np.log(n_central_range)))

    results = []
    for i, Pc in enumerate(P_central_range):
        M, R = solve_tov(P_mev, eps_mev, Pc)
        if M is not None and M > 0 and R is not None and R > 0:
            if n_central_range is not None:
                results.append([n_central_range[i], M, R])
            else:
                results.append([M, R])

    if len(results) == 0:
        return np.empty((0, 3 if n_central_range is not None else 2))

    return np.array(results)


def find_max_mass(P_mev, eps_mev, n_b_mev=None, n_points=150):
    """
    Find the maximum gravitational mass and corresponding radius.

    Returns
    -------
    M_max : float
        Maximum mass [M☉].
    R_at_Mmax : float
        Radius at maximum mass [km].
    """
    curve = mass_radius_curve(P_mev, eps_mev, n_b_mev=n_b_mev,
                              n_points=n_points)
    if len(curve) == 0:
        return None, None

    mass_col = 1 if curve.shape[1] == 3 else 0
    radius_col = 2 if curve.shape[1] == 3 else 1

    idx = np.argmax(curve[:, mass_col])
    return curve[idx, mass_col], curve[idx, radius_col]


def find_radius_at_mass(curve, target_mass):
    """
    Interpolate the M-R curve to find R at a given mass (stable branch).

    Parameters
    ----------
    curve : ndarray
        M-R curve from mass_radius_curve.
    target_mass : float
        Target mass [M☉].

    Returns
    -------
    R : float or None
        Radius [km] at the target mass, or None if outside range.
    """
    mass_col = 1 if curve.shape[1] == 3 else 0
    radius_col = 2 if curve.shape[1] == 3 else 1

    masses = curve[:, mass_col]
    radii = curve[:, radius_col]

    # Use only the stable branch (up to M_max)
    idx_max = np.argmax(masses)
    masses_stable = masses[:idx_max + 1]
    radii_stable = radii[:idx_max + 1]

    if target_mass < masses_stable.min() or target_mass > masses_stable.max():
        return None

    # Interpolate
    R_of_M = interp1d(masses_stable, radii_stable, kind='cubic')
    return float(R_of_M(target_mass))
