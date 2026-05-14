# FRAME
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
## FRactionation And Mixing Evaluation of isotopes.

Markov-chain Monte Carlo engine for calculation of isotope mixing wrapped in a simple Qt user interface.

See the [webpage](https://malewick.github.io/frame/) for a user guide, information on newest releases and more.

## Install
### Executables
- **Windows**: see the zipped executables in [releases](https://github.com/malewick/frame/releases/) (only Windows 10 compatible for now)
- **Linux** and **OSX**: no executables were prepared yet. Let us know if you wish to use our software on these platforms or run the code from sources as described below.

### Install from source
To get the sources from git do

```
$ git clone https://github.com/malewick/frame.git
```

Install the packages listed in `requirements.txt` (either with conda or pip). Then run:

```
python FRAME.py
```

Alternatively you can run the code without the graphical interface with (see the source for details and `input` directory for some sample input files):

```
python run/run_batch.py <data_file> <sources_file>
```

## Docs and user guide

The user guide can be found under [this link](https://malewick.github.io/frame/).

A detailed scientific description of the model is published in:

> Lewicki MP, Lewicka-Szczebak D, Skrzypek G (2022) **FRAME—Monte Carlo model for evaluation of the stable isotope mixing and fractionation.** *PLOS ONE* 17(11): e0277204. https://doi.org/10.1371/journal.pone.0277204

## Key features

- **N-dimensional mixing model** — supports 1D, 2D and 3D stable isotope systems with an arbitrary number of sources.
- **Source uncertainty** — sources can be described either as Gaussian point estimates (`stdev`/`sigma`) or as flat uniform distributions (`spread`/`delta`). Likelihoods for the latter are computed via exact numerical FFT convolution (introduced in v2).
- **Auxiliary parameters** — custom model equations (open-system fractionation, Rayleigh-type fractionation, equilibrium fractionation, etc.) with user-defined auxiliary variables `r`.
- **Selectable prior for auxiliary variables** — the sampling prior for each auxiliary variable can be set programmatically with `model.set_aux_prior(var_name, prior_type, r_min, r_max)`:
  - `'uniform'` — `r ~ Uniform[r_min, r_max]` (default)
  - `'loguniform'` — `log(r) ~ Uniform[log(r_min), log(r_max)]`, suitable when `r` spans orders of magnitude
  - The GUI exposes prior type, min and max controls per auxiliary variable.
- **Batch mode** — headless runs via `run/run_batch.py` with full CLI argument support (including `--aux_prior_type`, `--aux_r_min`, `--aux_r_max`).
- **Colab-ready** — see `docs/colab_demo.ipynb` for an interactive walkthrough.

## Changelog

### v2.0 (2026)

> **Note for v1.0 users:** if your model used `spread`/`delta` parameters for source uncertainty, re-running your analysis with v2.0 is strongly recommended — the likelihood function for uniform distributions has been corrected (see below).

#### Bug fix — uniform-distribution likelihood (breaking for spread-based models)

In v1.0, when source uncertainty was expressed as a uniform spread (`delta`/`spread`), the likelihood was approximated by summing spreads linearly and applying a single `erf`-based Gaussian envelope. This is mathematically incorrect: the convolution of multiple uniform distributions with a Gaussian is not itself a Gaussian, and the approximation introduced a bias that grew with the number of sources and the magnitude of the spreads.

v2.0 replaces this with an exact per-isotope numerical convolution using FFT (`scipy.signal.fftconvolve`). For each MCMC iteration and each isotope dimension, the compound PDF is built by:
1. starting from the combined Gaussian (all `stdev` contributions, propagated analytically),
2. convolving in each uniform component separately.

The resulting normalised PDF is evaluated at the measurement point to give the correct likelihood. Models that use only Gaussian uncertainties (`stdev`/`sigma`) are unaffected and take the fast analytical path as before.

#### New feature — selectable prior for auxiliary variables

The auxiliary variable `r` (fractionation progress, reduction fraction, etc.) was previously always sampled from `Uniform(0, 1)`. v2.0 introduces a configurable prior:

```python
# uniform prior — default, identical to v1.0 behaviour
model.set_aux_prior('r', prior_type='uniform', r_min=0.0, r_max=1.0)

# log-uniform prior — suitable when r is expected to be small
# but could in principle span orders of magnitude
model.set_aux_prior('r', prior_type='loguniform', r_min=0.001, r_max=1.0)
```

The GUI exposes dropdown + min/max fields per auxiliary variable. The batch script supports `--aux_prior_type`, `--aux_r_min`, `--aux_r_max`.

#### Performance improvements

- `eval()` calls on derivative/model strings replaced with precompiled `compile()` code objects — significant speedup for long chains.
- FFT convolution grid reduced to 200 points (sufficient accuracy, ~5× faster than the initial 1000-point prototype).
- Uniform PDF computed directly with NumPy instead of constructing a `scipy.stats.uniform` object per call.

#### GUI

- Prior type / min / max controls added to the "Aux. variables" panel (loaded after an aux file is selected).
- Plot columns now receive all horizontal stretch when the window is resized.

---

### v1.0 (2022)

Initial public release accompanying the paper (Lewicki et al., *PLOS ONE*, 2022).

---

## Discussion and bugs report

Our mailing list is at https://groups.google.com/g/frame-isotopes. You can ask questions about the underlying algorithms, usage, etc.

Our issue tracker is at https://github.com/malewick/frame/issues. Please report any bugs that you find.

## Code
In the `src` directory:
- `FRAME.py` is the main script producing the GUI.
- `NDimModel.py` contains the implementation of the model and statistical calculations.
- `TimeSeriesPlot.py`, `PathPlot.py`, `CorrelationPlot.py` are the classes with matplotlib plot implementations.

`requirements.txt` lists all the required packages and versions (PySide2, pandas, numpy, scipy, sympy, matplotlib).

## Attribution
Please cite the following paper when using this code:

> Lewicki MP, Lewicka-Szczebak D, Skrzypek G (2022) **FRAME—Monte Carlo model for evaluation of the stable isotope mixing and fractionation.** *PLOS ONE* 17(11): e0277204. https://doi.org/10.1371/journal.pone.0277204

*Copyright (C) 2020–2026  Maciej P. Lewicki*
