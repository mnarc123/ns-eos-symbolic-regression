"""
MCMC sampler for Phase C: Bayesian parameter estimation.

Uses emcee (affine-invariant ensemble sampler) with HDF5 backend
for incremental checkpointing.

Usage:
    python phase_c/mcmc_run.py --formula B_c15 --nwalkers 32 --nsteps 5000
    python phase_c/mcmc_run.py --formula B_c17 --nwalkers 40 --nsteps 5000
"""

import numpy as np
import emcee
import json
import argparse
import pathlib
import sys
import time
import multiprocessing
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from phase_c.parametrize import FORMULAS, THETA_0, PARAM_NAMES
from phase_c.prior import PRIORS
from phase_c.likelihood import log_likelihood_total
from phase_c.crust import load_crust_from_sly4

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
CHAINS_DIR = RESULTS_DIR / "chains"
CHAINS_DIR.mkdir(parents=True, exist_ok=True)


# Module-level globals for multiprocessing workers
_CS2_FUNC = None
_LOG_PRIOR_FUNC = None
_CRUST = None


def _init_worker(cs2_func, log_prior_func, crust):
    """Initializer for pool workers — sets module-level globals."""
    global _CS2_FUNC, _LOG_PRIOR_FUNC, _CRUST
    _CS2_FUNC = cs2_func
    _LOG_PRIOR_FUNC = log_prior_func
    _CRUST = crust


def log_posterior(theta):
    """Log-posterior = log-prior + log-likelihood.

    Uses module-level globals so it can be pickled for multiprocessing.
    """
    lp = _LOG_PRIOR_FUNC(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_total(theta, _CS2_FUNC, _CRUST)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_mcmc(formula_name, nwalkers=32, nsteps=5000, nburn=1000,
             resume=False, ncores=1):
    """
    Run MCMC for a given formula.

    Parameters
    ----------
    formula_name : str
        'B_c15' or 'B_c17'.
    nwalkers : int
        Number of walkers (must be >= 2 * ndim).
    nsteps : int
        Total MCMC steps.
    nburn : int
        Burn-in steps to discard.
    resume : bool
        If True, resume from existing chain file.

    Returns
    -------
    flat_samples : ndarray, shape (n_effective, ndim)
    """
    cs2_func = FORMULAS[formula_name]
    log_prior_func = PRIORS[formula_name]
    theta_0 = THETA_0[formula_name]
    ndim = len(theta_0)

    # Load crust data
    print("Loading crust EoS (SLy4)...")
    crust = load_crust_from_sly4()
    print(f"  Crust: {len(crust['n_b'])} points, "
          f"n_match = {crust['n_b'][-1]:.4f} fm^-3")

    # Test likelihood at best-fit point
    print(f"\nTesting likelihood at theta_0 = {theta_0} ...")
    t0 = time.time()
    ll_test = log_likelihood_total(theta_0, cs2_func, crust)
    dt = time.time() - t0
    print(f"  log-L(theta_0) = {ll_test:.4f}  ({dt:.2f} s)")

    if not np.isfinite(ll_test):
        print("ERROR: Likelihood is -inf at the best-fit point!")
        print("Check that the EoS reconstruction works for these parameters.")
        return None

    est_time_h_serial = nwalkers * nsteps * dt / 3600
    effective_speedup = min(ncores, nwalkers)
    est_time_h = est_time_h_serial / effective_speedup
    print(f"\nEstimated runtime: ~{est_time_h:.1f} hours "
          f"({ncores} cores, {dt:.2f} s/eval, "
          f"~{effective_speedup}x speedup vs serial {est_time_h_serial:.1f}h)")

    # HDF5 backend for checkpointing
    chain_file = str(CHAINS_DIR / f"{formula_name}_chain.h5")
    backend = emcee.backends.HDFBackend(chain_file)

    if not resume:
        backend.reset(nwalkers, ndim)

    # Initialize walkers: small Gaussian ball around theta_0
    if resume and backend.iteration > 0:
        print(f"Resuming from iteration {backend.iteration}")
        pos = None  # emcee will pick up from last position
        remaining = nsteps - backend.iteration
        if remaining <= 0:
            print("Chain already complete.")
            flat_samples = backend.get_chain(discard=nburn, flat=True)
            return flat_samples
    else:
        # Perturbation scale: 1% of parameter value (or 0.01 if param~0)
        scale = np.maximum(np.abs(theta_0) * 0.01, 0.001)
        pos = theta_0 + scale * np.random.randn(nwalkers, ndim)
        remaining = nsteps

    # Initialize module-level globals for this process too
    _init_worker(cs2_func, log_prior_func, crust)

    print(f"\n{'='*60}")
    print(f"MCMC for {formula_name}")
    print(f"{'='*60}")
    print(f"  ndim = {ndim}")
    print(f"  nwalkers = {nwalkers}")
    print(f"  nsteps = {remaining} (total target: {nsteps})")
    print(f"  nburn = {nburn}")
    print(f"  ncores = {ncores}")
    print(f"  theta_0 = {theta_0}")
    print(f"  chain file: {chain_file}")
    print()

    # Run with multiprocessing pool if ncores > 1
    if ncores > 1:
        with multiprocessing.Pool(
            processes=ncores,
            initializer=_init_worker,
            initargs=(cs2_func, log_prior_func, crust),
        ) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_posterior,
                pool=pool, backend=backend,
            )
            sampler.run_mcmc(pos, remaining, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            backend=backend,
        )
        sampler.run_mcmc(pos, remaining, progress=True)

    # Diagnostics
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation time: {tau}")
        print(f"Effective samples: {nsteps * nwalkers / np.max(tau):.0f}")
    except emcee.autocorr.AutocorrError:
        tau = np.array([np.nan] * ndim)
        print("\nWarning: Could not estimate autocorrelation time "
              "(chain may be too short)")

    # Extract post burn-in samples
    flat_samples = sampler.get_chain(discard=nburn, flat=True)
    print(f"Samples after burn-in: {len(flat_samples)}")

    # Summary statistics
    median = np.median(flat_samples, axis=0)
    std = np.std(flat_samples, axis=0)
    p16 = np.percentile(flat_samples, 16, axis=0)
    p84 = np.percentile(flat_samples, 84, axis=0)
    p2_5 = np.percentile(flat_samples, 2.5, axis=0)
    p97_5 = np.percentile(flat_samples, 97.5, axis=0)

    print(f"\nParameter summary ({formula_name}):")
    param_labels = PARAM_NAMES[formula_name]
    for i in range(ndim):
        print(f"  {param_labels[i]}: "
              f"{median[i]:.6f} "
              f"({p16[i]:.6f}, {p84[i]:.6f}) [68%] "
              f"({p2_5[i]:.6f}, {p97_5[i]:.6f}) [95%]")

    # Save summary JSON
    summary = {
        "formula": formula_name,
        "ndim": ndim,
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "nburn": nburn,
        "theta_0": theta_0.tolist(),
        "autocorr_time": tau.tolist() if tau is not None else None,
        "median": median.tolist(),
        "std": std.tolist(),
        "percentiles_16": p16.tolist(),
        "percentiles_84": p84.tolist(),
        "percentiles_2.5": p2_5.tolist(),
        "percentiles_97.5": p97_5.tolist(),
        "acceptance_fraction": float(np.mean(
            sampler.acceptance_fraction)),
    }

    summary_file = RESULTS_DIR / f"{formula_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_file}")

    return flat_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase C: MCMC parameter estimation for SR formulas")
    parser.add_argument("--formula", type=str, default="B_c15",
                        choices=["B_c15", "B_c17"],
                        help="Formula to sample (default: B_c15)")
    parser.add_argument("--nwalkers", type=int, default=32,
                        help="Number of walkers (default: 32)")
    parser.add_argument("--nsteps", type=int, default=5000,
                        help="Total MCMC steps (default: 5000)")
    parser.add_argument("--nburn", type=int, default=1000,
                        help="Burn-in steps (default: 1000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing chain")
    parser.add_argument("--ncores", type=int, default=0,
                        help="Number of CPU cores (0 = auto-detect)")
    args = parser.parse_args()

    ncores = args.ncores
    if ncores <= 0:
        ncores = max(1, os.cpu_count() - 2)  # leave 2 cores free

    run_mcmc(
        formula_name=args.formula,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        nburn=args.nburn,
        resume=args.resume,
        ncores=ncores,
    )
