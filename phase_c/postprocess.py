"""
Post-processing for Phase C MCMC chains.

Generates:
  - Corner plots of parameter posteriors
  - Credible bands for c_s²(x)
  - Credible bands for M-R diagrams with NICER ellipses
  - Lambda_tilde posterior comparison with GW170817
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import corner
import emcee
import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from phase_c.parametrize import FORMULAS, PARAM_NAMES, THETA_0
from phase_c.likelihood import (
    eos_from_theta, compute_observables, NICER_PULSARS, GW170817
)
from phase_c.crust import load_crust_from_sly4
from phase_c.tidal import combined_tidal_deformability
from physics.tov_solver import find_radius_at_mass
from search.prepare_training_data import prepare_single_eos, EOS_FILES

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
CHAINS_DIR = RESULTS_DIR / "chains"


def load_samples(formula_name, nburn=1000):
    """Load flat samples from HDF5 chain file."""
    chain_file = str(CHAINS_DIR / f"{formula_name}_chain.h5")
    reader = emcee.backends.HDFBackend(chain_file, read_only=True)
    samples = reader.get_chain(discard=nburn, flat=True)
    print(f"Loaded {len(samples)} samples for {formula_name}")
    return samples


def make_corner_plot(formula_name, nburn=1000):
    """Corner plot of parameter posteriors."""
    samples = load_samples(formula_name, nburn)
    labels = PARAM_NAMES[formula_name]

    fig = corner.corner(
        samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".5f",
    )
    out = RESULTS_DIR / f"corner_{formula_name}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def make_cs2_credible_band(formula_name, nburn=1000, n_draw=500):
    """
    Plot 68% and 95% credible bands for c_s²(x), overlaid with
    tabular EoS data from Phase A/B.
    """
    samples = load_samples(formula_name, nburn)
    cs2_func = FORMULAS[formula_name]

    x = np.linspace(0.5, 10.0, 300)
    cs2_draws = np.zeros((n_draw, len(x)))

    idx = np.random.choice(len(samples), min(n_draw, len(samples)),
                           replace=False)
    for i, j in enumerate(idx):
        try:
            cs2_draws[i] = cs2_func(x, samples[j])
        except Exception:
            cs2_draws[i] = np.nan

    # Remove failed draws
    valid = ~np.any(np.isnan(cs2_draws), axis=1)
    cs2_draws = cs2_draws[valid]

    if len(cs2_draws) < 10:
        print("ERROR: Too few valid draws for credible band")
        return

    median = np.median(cs2_draws, axis=0)
    lo68, hi68 = np.percentile(cs2_draws, [16, 84], axis=0)
    lo95, hi95 = np.percentile(cs2_draws, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Credible bands
    ax.fill_between(x, lo95, hi95, alpha=0.15, color='steelblue',
                    label='95% CI')
    ax.fill_between(x, lo68, hi68, alpha=0.3, color='steelblue',
                    label='68% CI')
    ax.plot(x, median, 'k-', lw=2, label='Median')

    # Overlay tabular EoS
    eos_colors = {
        'sly4': '#d62728', 'apr4': '#ff7f0e', 'bsk24': '#2ca02c',
        'dd2': '#9467bd', 'fsu2r': '#8c564b', 'qmcrmf3': '#e377c2',
    }
    eos_labels = {
        'sly4': 'SLy4', 'apr4': 'APR4', 'bsk24': 'BSk24',
        'dd2': 'DD2', 'fsu2r': 'FSU2R', 'qmcrmf3': 'QMC-RMF3',
    }
    for name, fpath in EOS_FILES.items():
        if fpath.exists():
            try:
                xd, cs2d = prepare_single_eos(name)
                ax.plot(xd, cs2d, '.', color=eos_colors.get(name, 'gray'),
                        markersize=1.5, alpha=0.5,
                        label=eos_labels.get(name, name))
            except Exception:
                pass

    # Physics bounds
    ax.axhline(1.0/3.0, ls='--', color='gray', alpha=0.5,
               label=r'$c_s^2 = 1/3$ (conformal)')
    ax.axhline(1.0, ls=':', color='red', alpha=0.3,
               label=r'$c_s^2 = 1$ (causality)')
    ax.axhline(0.0, ls='-', color='gray', alpha=0.2)

    ax.set_xlabel(r'$x = n_b / n_0$', fontsize=14)
    ax.set_ylabel(r'$c_s^2$', fontsize=14)
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.set_title(f'{formula_name}: Speed of sound squared '
                 r'$c_s^2(x)$ with credible bands', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = RESULTS_DIR / f"cs2_band_{formula_name}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def make_mr_credible_band(formula_name, nburn=1000, n_draw=200):
    """
    Plot credible bands for M-R diagram with NICER ellipses.
    """
    samples = load_samples(formula_name, nburn)
    cs2_func = FORMULAS[formula_name]
    crust = load_crust_from_sly4()

    idx = np.random.choice(len(samples), min(n_draw, len(samples)),
                           replace=False)

    all_curves = []
    for j in idx:
        theta = samples[j]
        eos = eos_from_theta(cs2_func, theta, crust)
        if eos is None:
            continue
        obs = compute_observables(eos)
        if obs is None:
            continue
        mr = obs['mr_curve']  # columns: M, R, Lambda
        all_curves.append((mr[:, 0], mr[:, 1]))

    if len(all_curves) < 5:
        print("ERROR: Too few valid M-R curves for credible band")
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    # Draw individual M-R curves (transparent)
    for M, R in all_curves:
        ax.plot(R, M, color='steelblue', alpha=0.03, lw=0.5)

    # Compute envelope on a regular M grid
    M_grid = np.linspace(0.5, 2.6, 200)
    R_at_M = []
    for M_arr, R_arr in all_curves:
        try:
            idx_max = np.argmax(M_arr)
            M_stable = M_arr[:idx_max + 1]
            R_stable = R_arr[:idx_max + 1]
            if len(M_stable) > 3:
                from scipy.interpolate import interp1d
                R_of_M = interp1d(M_stable, R_stable, kind='linear',
                                  bounds_error=False, fill_value=np.nan)
                R_at_M.append(R_of_M(M_grid))
        except Exception:
            pass

    if len(R_at_M) > 10:
        R_at_M = np.array(R_at_M)
        R_med = np.nanmedian(R_at_M, axis=0)
        R_lo68 = np.nanpercentile(R_at_M, 16, axis=0)
        R_hi68 = np.nanpercentile(R_at_M, 84, axis=0)
        R_lo95 = np.nanpercentile(R_at_M, 2.5, axis=0)
        R_hi95 = np.nanpercentile(R_at_M, 97.5, axis=0)

        valid = ~np.isnan(R_med)
        ax.fill_betweenx(M_grid[valid], R_lo95[valid], R_hi95[valid],
                         alpha=0.15, color='steelblue', label='95% CI')
        ax.fill_betweenx(M_grid[valid], R_lo68[valid], R_hi68[valid],
                         alpha=0.3, color='steelblue', label='68% CI')
        ax.plot(R_med[valid], M_grid[valid], 'k-', lw=1.5, label='Median')

    # NICER ellipses
    nicer_colors = {
        'J0030': 'darkorange', 'J0740': 'crimson', 'J0437': 'forestgreen'
    }
    for name, data in NICER_PULSARS.items():
        color = nicer_colors.get(name, 'gray')
        for n_sigma in [1, 2]:
            ell = Ellipse(
                (data['R'], data['M']),
                width=2 * n_sigma * data['sigma_R'],
                height=2 * n_sigma * data['sigma_M'],
                fill=False, edgecolor=color,
                lw=2.0 if n_sigma == 1 else 1.0,
                ls='-' if n_sigma == 1 else '--',
                alpha=0.7 if n_sigma == 1 else 0.4,
            )
            ax.add_patch(ell)
        ax.plot(data['R'], data['M'], 'x', color=color, markersize=10,
                markeredgewidth=2)
        ax.annotate(f"NICER {name}", (data['R'], data['M']),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, color=color)

    # 2 M_sun line
    ax.axhline(2.0, ls=':', color='gray', alpha=0.5)
    ax.text(14.5, 2.02, r'$2\,M_\odot$', fontsize=9, color='gray')

    ax.set_xlabel(r'Radius $R$ [km]', fontsize=14)
    ax.set_ylabel(r'Mass $M$ [$M_\odot$]', fontsize=14)
    ax.set_xlim(8, 16)
    ax.set_ylim(0.5, 2.8)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(f'{formula_name}: Mass-radius with credible bands',
                 fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = RESULTS_DIR / f"mr_band_{formula_name}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def make_lambda_posterior(formula_name, nburn=1000, n_draw=200):
    """
    Plot predicted Lambda_tilde distribution vs GW170817 posterior.

    Uses compute_observables (single TOV+Love pass) for speed.
    """
    samples = load_samples(formula_name, nburn)
    cs2_func = FORMULAS[formula_name]
    crust = load_crust_from_sly4()

    q = GW170817['q_median']
    M_chirp = GW170817['M_chirp']
    M_tot = M_chirp / (q / (1 + q)**2)**(3.0 / 5.0)
    M1 = M_tot / (1 + q)
    M2 = M_tot * q / (1 + q)

    idx = np.random.choice(len(samples), min(n_draw, len(samples)),
                           replace=False)

    Lambda_tilde_samples = []
    for i, j in enumerate(idx):
        theta = samples[j]
        eos = eos_from_theta(cs2_func, theta, crust)
        if eos is None:
            continue
        obs = compute_observables(eos)
        if obs is None or obs.get('Lambda_of_M') is None:
            continue
        try:
            L1 = float(obs['Lambda_of_M'](M1))
            L2 = float(obs['Lambda_of_M'](M2))
        except (ValueError, IndexError):
            continue
        if np.isnan(L1) or np.isnan(L2):
            continue
        Lt = combined_tidal_deformability(L1, L2, q)
        if 0 < Lt < 5000:
            Lambda_tilde_samples.append(Lt)
        if (i + 1) % 50 == 0:
            print(f"  Lambda draws: {i+1}/{len(idx)} "
                  f"({len(Lambda_tilde_samples)} valid)")

    if len(Lambda_tilde_samples) < 10:
        print("ERROR: Too few valid Lambda_tilde samples")
        return

    Lambda_tilde_samples = np.array(Lambda_tilde_samples)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Predicted histogram
    ax.hist(Lambda_tilde_samples, bins=40, density=True, alpha=0.5,
            color='steelblue', label=f'{formula_name} posterior')

    # GW170817 Gaussian approximation
    x_lam = np.linspace(0, 1500, 300)
    from scipy.stats import norm
    gw_pdf = norm.pdf(x_lam, GW170817['Lambda_tilde_mean'],
                      GW170817['Lambda_tilde_sigma'])
    ax.plot(x_lam, gw_pdf, 'r-', lw=2, label='GW170817 (Gaussian approx)')

    # 90% CL from LVK
    ax.axvspan(70, 720, alpha=0.08, color='red', label='GW170817 90% CL')

    ax.set_xlabel(r'$\tilde{\Lambda}$', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)
    ax.set_title(f'{formula_name}: Tidal deformability posterior', fontsize=14)
    ax.set_xlim(0, 1500)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = RESULTS_DIR / f"lambda_tilde_{formula_name}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary
    med = np.median(Lambda_tilde_samples)
    lo, hi = np.percentile(Lambda_tilde_samples, [5, 95])
    print(f"  Lambda_tilde: {med:.0f} ({lo:.0f}, {hi:.0f}) [90% CI]")


def run_all_postprocessing(formula_name, nburn=1000):
    """Run all post-processing for a given formula."""
    print(f"\n{'='*60}")
    print(f"Post-processing: {formula_name}")
    print(f"{'='*60}\n")

    make_corner_plot(formula_name, nburn)
    make_cs2_credible_band(formula_name, nburn)
    make_mr_credible_band(formula_name, nburn)
    make_lambda_posterior(formula_name, nburn)

    print(f"\nAll plots saved for {formula_name}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase C post-processing: plots from MCMC chains")
    parser.add_argument("--formula", type=str, default="B_c15",
                        choices=["B_c15", "B_c17"])
    parser.add_argument("--nburn", type=int, default=1000)
    args = parser.parse_args()

    run_all_postprocessing(args.formula, args.nburn)
