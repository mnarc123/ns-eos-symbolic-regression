"""
Unit conversions for neutron star EoS calculations.

Natural units: c = ℏ = 1
Geometric units: G = c = 1

Key conversion factors between nuclear physics units (MeV, fm)
and geometric units (km) used in the TOV solver.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
HBAR_C = 197.3269804  # ℏc in MeV·fm
C_CGS = 2.99792458e10  # speed of light in cm/s
G_CGS = 6.67430e-8  # gravitational constant in cm³/(g·s²)
M_SUN_CGS = 1.98892e33  # solar mass in g

# ---------------------------------------------------------------------------
# Nuclear physics constants
# ---------------------------------------------------------------------------
N_0 = 0.16  # nuclear saturation density in fm⁻³
M_NEUTRON = 939.565  # neutron mass in MeV
M_PROTON = 938.272  # proton mass in MeV
M_NUCLEON = 938.918  # average nucleon mass in MeV (approximate)

# ---------------------------------------------------------------------------
# Geometric unit conversions (G = c = 1)
# ---------------------------------------------------------------------------
# 1 MeV/fm³ → km⁻² in geometric units
# Derivation: P [MeV/fm³] → P [g/(cm·s²)] → P [km⁻²]
#   MeV/fm³ = 1.602176634e-33 erg / (1e-39 cm³)
#           = 1.602176634e6 erg/cm³ = 1.602176634e6 dyn/cm²
#   In geometric: P_geo = (G/c⁴) P_cgs, then express in km⁻²
MEV_FM3_TO_GEO = 1.3234e-6  # MeV/fm³ → km⁻² (geometric)

# Solar mass in km (geometric units)
M_SUN_KM = 1.4766  # G M_☉ / c² in km

# Solar mass in geometric density units (for convenience)
# 1 M_☉ = M_SUN_KM in geometric length units

# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def mev_fm3_to_geo(value):
    """Convert pressure or energy density from MeV/fm³ to km⁻² (geometric)."""
    return value * MEV_FM3_TO_GEO


def geo_to_mev_fm3(value):
    """Convert pressure or energy density from km⁻² (geometric) to MeV/fm³."""
    return value / MEV_FM3_TO_GEO


def mass_geo_to_msun(m_km):
    """Convert mass from geometric units (km) to solar masses."""
    return m_km / M_SUN_KM


def mass_msun_to_geo(m_msun):
    """Convert mass from solar masses to geometric units (km)."""
    return m_msun * M_SUN_KM


def density_to_x(n_b):
    """Normalize baryon density to saturation density: x = n_b / n_0."""
    return n_b / N_0


def x_to_density(x):
    """Convert normalized density back to fm⁻³: n_b = x * n_0."""
    return x * N_0


def energy_density_from_mass_density(n_b, binding_energy_per_nucleon=0.0):
    """
    Approximate energy density from baryon density.

    epsilon ≈ n_b * (m_N + B/A) where B/A is binding energy per nucleon.
    At saturation: B/A ≈ -16 MeV.

    Parameters
    ----------
    n_b : array_like
        Baryon number density in fm⁻³.
    binding_energy_per_nucleon : float
        Binding energy per nucleon in MeV (negative for bound matter).

    Returns
    -------
    epsilon : array_like
        Energy density in MeV/fm³.
    """
    return n_b * (M_NUCLEON + binding_energy_per_nucleon)
