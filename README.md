# Analytic Neutron Star Equation of State from Physics-Informed Symbolic Regression

Companion code to M. Narcisi, *Phys. Rev. D* (submitted, 2026).

## The formula

Physics-informed symbolic regression (PySR) trained on six microscopic
equation-of-state tables discovers a two-parameter analytic formula for
the speed of sound squared in dense nuclear matter:

$$c_s^2(x) \;=\; \tanh\!\bigl(x\,\tanh(a\,x)\bigr)\;\tanh\!\bigl(e^{-\tanh(x-b)}\bigr)$$

where $x = n_b / n_0$ is the baryon density normalised to saturation
($n_0 = 0.16\;\mathrm{fm}^{-3}$), with Bayesian posterior values

$$a = 0.060^{+0.009}_{-0.008},\qquad b = 9.7^{+3.5}_{-3.8}\qquad(68\%\;\text{CI}).$$

The formula automatically satisfies causality ($0 \le c_s^2 \le 1$),
thermodynamic stability ($c_s^2 \ge 0$), and recovers the conformal
limit $c_s^2 \to 1/3$ at high density via tanh(e⁻¹) ≈ 0.35.

### Predicted observables

| | SR best-fit | MCMC median |
|---|---|---|
| M_max | 2.18 M☉ | 2.33 M☉ |
| R_1.4 | 11.0 km | 12.1 km |
| Λ̃ (GW170817) | — | 439 (+191/−131, 90% CI) |

## Quick start

```python
import numpy as np

def cs2(x, a=0.060, b=9.7):
    """Speed of sound squared, B_c15 formula."""
    return np.tanh(x * np.tanh(a * x)) * np.tanh(np.exp(-np.tanh(x - b)))

# Example: evaluate at twice saturation density
x = 2.0
print(f"c_s^2(2 n_0) = {cs2(x):.4f}")
```

## Reproducing the paper

The three phases of the analysis can be reproduced independently.

### Phase A — Single-EoS validation (SLy4)

```bash
python search/sr_search.py --phase a --eos sly4 --niter 2000 --procs 20
```

Recovers the SLy4 speed of sound with RMSE = 0.020 and
M_max = 2.087 M☉ (Table II in the paper).

### Phase B — Universal formula from six EoS

```bash
python search/sr_search.py --phase b --niter 5000 --procs 20
```

Discovers the B_c15 formula with weighted MSE = 0.00115 across
SLy4, APR4, BSk24, DD2, FSU2R, and QMC-RMF3 (Table III).

### Phase C — Bayesian parameter estimation

```bash
python phase_c/mcmc_run.py
```

MCMC with emcee using NICER mass–radius posteriors (J0030+0451,
J0740+6620, J0437−4715) and the GW170817 tidal constraint.
Runtime: ~4 hours on a single core, ~1.5 hours with 8 cores.

## Repository structure

```
├── data/                        # EoS tables (CSV) and generation script
├── physics/
│   ├── eos_from_cs2.py          # c_s²(x) → μ(n), ε(n), P(n) integration
│   └── tov_solver.py            # TOV + tidal deformability (Λ)
├── search/
│   ├── sr_search.py             # PySR with physics-informed loss (Julia)
│   └── validate_candidates.py   # TOV validation of SR candidates
├── phase_c/
│   ├── mcmc_run.py              # emcee sampler
│   ├── likelihood.py            # NICER + GW170817 + M_max likelihood
│   └── postprocess.py           # Corner plots, credible bands
├── results/
│   ├── phase_a_results.json     # Phase A Pareto front
│   ├── phase_b_results.json     # Phase B Pareto front
│   ├── B_c15_summary.json       # MCMC posterior summary
│   ├── B_c17_summary.json       # MCMC posterior summary (3-param variant)
│   └── figures/                 # All 10 paper figures
└── requirements.txt
```

## Requirements

- Python ≥ 3.11
- Julia 1.10 LTS (for PySR backend)
- Key packages: pysr, emcee, scipy, numpy, matplotlib, corner

```bash
pip install -r requirements.txt
python -c 'import pysr; pysr.install()'
```

## Training EoS

Six nucleonic equations of state from published microphysics calculations,
available from the [CompOSE database](https://compose.obspm.fr):

| EoS | Method | Reference | M_max (M☉) |
|---|---|---|---|
| SLy4 | Skyrme HF | Douchin & Haensel (2001) | 2.05 |
| APR4 | Variational | Akmal et al. (1998) | 2.21 |
| BSk24 | Skyrme HF | Goriely et al. (2013) | 2.28 |
| DD2 | RMF (DD) | Typel et al. (2010) | 2.42 |
| FSU2R | RMF (NL) | Tolos et al. (2017) | 2.07 |
| QMC-RMF3 | QMC+RMF | Whittenbury et al. (2014) | 2.10 |

## Observational data

NICER posterior samples are available on Zenodo:
- PSR J0740+6620: [Salmi et al. 2024](https://doi.org/10.5281/zenodo.10519473)
- PSR J0030+0451: [Vinciguerra et al. 2024](https://doi.org/10.5281/zenodo.8239000)
- PSR J0437−4715: [Choudhury et al. 2024](https://doi.org/10.5281/zenodo.10889032)

GW170817 tidal constraint from [Abbott et al. 2019](https://doi.org/10.1103/PhysRevX.9.011001)
(Λ̃ = 300 +420/−230, 90% CL).

## Citation

If you use this code or the B_c15 formula, please cite:

```bibtex
@article{Narcisi2026,
  title   = {Analytic formulas for the neutron star equation of state
             from physics-informed symbolic regression},
  author  = {Narcisi, M.},
  journal = {Phys. Rev. D},
  year    = {2026},
  note    = {submitted}
}
```
[![DOI](https://zenodo.org/badge/1168339376.svg)](https://doi.org/10.5281/zenodo.18799902)
## License

MIT
