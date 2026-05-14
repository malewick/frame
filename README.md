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
