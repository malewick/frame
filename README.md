# FRAME
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
## FRactionation And Mixing Evaluation of isotopes.

Markov-chain Monte Carlo engine for calculation of isotope mixing wrapped in a simple Qt user interface.

See the [webpage](https://malewick.github.io/frame/) for a user guide, information on newest releases and more.

## Try it online — no installation needed

> **Have a Google account and a little Python?  That's all you need.**

Click the badge below to open an interactive demo in Google Colab.  The notebook clones this repository, installs all dependencies automatically, and walks you through a complete mixing analysis — including plots — entirely in your browser.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malewick/frame/blob/main/docs/colab_demo.ipynb)

The demo covers:
- a simple 2-D mixing example (two isotope ratios, three end-member sources)
- a fractionation example with selectable priors for the auxiliary variable `r`

No local software installation, no command line, no configuration files — just click and run.

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

### Windows / PowerShell (executable)

Pre-built Windows executables are available in [releases](https://github.com/malewick/frame/releases/).
`FRAME.exe` launches the graphical interface. `FRAME_batch.exe` is the headless batch runner, equivalent to `run_batch.py`, and is controlled entirely through command-line arguments:

**Required arguments**

| Argument | Description |
|---|---|
| `data_file` | Path to the input data CSV file |
| `sources_file` | Path to the sources CSV file |

**Optional arguments**

| Argument | Default | Description |
|---|---|---|
| `--aux_file` | *(none)* | Path to the auxiliary parameters CSV file |
| `--output_dir` | `output` | Directory for saving results |
| `--output_filenames` | *(auto)* | Custom tag prepended to output file names |
| `--output_formats` | `pdf,png` | Comma-separated list of plot formats |
| `--niter` | `1000000` | Maximum number of MCMC iterations |
| `--burnout` | `100` | Number of burn-in iterations |
| `--chain_length` | `500` | Number of accepted steps to record |
| `--plot_online` | `True` | Show live plots during the run (`True`/`False`) |
| `--aux_prior_type` | `uniform` | Prior for the auxiliary variable `r`: `uniform` or `loguniform` |
| `--aux_r_min` | `0.0` | Lower bound of the auxiliary variable prior |
| `--aux_r_max` | `1.0` | Upper bound of the auxiliary variable prior |

**Example — simple 2D mixing (PowerShell)**

```powershell
.\FRAME_batch.exe data\data.csv config\sources.csv `
    --output_dir results --output_filenames run1 `
    --output_formats pdf,png --niter 500000 --burnout 200 --chain_length 1000 `
    --plot_online False
```

**Example — 2D mixing with fractionation and a log-uniform prior for `r` (PowerShell)**

```powershell
.\FRAME_batch.exe data\data.csv config\sources.csv `
    --aux_file data\aux.csv `
    --aux_prior_type loguniform --aux_r_min 0.001 --aux_r_max 1.0 `
    --output_dir results --output_filenames run1 --plot_online False
```

> Tip: quote paths that contain spaces, e.g. `"my data\data.csv"`.

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
- **Colab-ready** — run everything in your browser with the [interactive demo notebook](https://colab.research.google.com/github/malewick/frame/blob/main/docs/colab_demo.ipynb); no local installation required.

## Changelog

### v2.0 (2026)

> **Note for v1.0 users:** if your model used `spread`/`delta` parameters for source uncertainty, re-running your analysis with v2.0 is strongly recommended — the likelihood function for uniform distributions has been corrected (see below).

#### Improved likelihood for uniform-distribution sources (breaking for spread-based models)

In v1.0, when source uncertainty was expressed as a uniform spread (`delta`/`spread`), the likelihood was computed using a Gaussian approximation based on linearly summed spreads. While practical, this approximation becomes less accurate as the number of sources grows or the spread values are large relative to the measurement uncertainty.

v2.0 replaces this with an exact per-isotope numerical convolution using FFT (`scipy.signal.fftconvolve`). For each MCMC iteration and each isotope dimension, the compound PDF is built by:
1. starting from the combined Gaussian (all `stdev` contributions, propagated analytically),
2. convolving in each uniform component separately.

The resulting normalised PDF is evaluated at the measurement point to give the likelihood. Models that use only Gaussian uncertainties (`stdev`/`sigma`) are unaffected and take the fast analytical path as before.

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
- FFT convolution optimized to handle any probability distribution numerically (using 200 points - decrease for performance improvement, increase for more accuracy).
- Uniform PDF computed directly with NumPy instead of constructing a `scipy.stats.uniform` object per call.

#### GUI

- Prior type / min / max controls added to the "Aux. variables" panel (loaded after an aux file is selected).

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
