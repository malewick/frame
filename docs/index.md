# FRAME
## FRactionation And Mixing Evaluation

Welcome to FRAME — a Bayesian stable isotope mixing model with a friendly graphical interface.

---

## About

FRAME estimates the fractional contributions of multiple isotope sources to a mixture, with optional simultaneous estimation of isotopic fractionation progress (e.g. denitrification, evaporation, N₂O reduction). The core algorithm is a **Markov-Chain Monte Carlo** model (Metropolis-Hastings) implemented in Python.

Key capabilities:

- **1D, 2D and 3D** stable isotope systems; arbitrary number of sources
- Source isotope signatures defined either as **Gaussian point estimates** (`stdev`) or **flat uniform ranges** (`spread`) — likelihoods for uniform sources are computed via exact FFT convolution (v2.0+)
- **Custom model equations** for any fractionation process (open-system, Rayleigh-type, equilibrium, etc.)
- **Selectable prior** for auxiliary variables (`uniform` or `log-uniform`), configurable from the GUI or the Python API
- **Graphical interface** and **headless batch mode** (Python script or Windows executable)

The scientific background and detailed case studies are published in:

> Lewicki MP, Lewicka-Szczebak D, Skrzypek G (2022) **FRAME — Monte Carlo model for evaluation of the stable isotope mixing and fractionation.** *PLOS ONE* 17(11): e0277204.  
> [https://doi.org/10.1371/journal.pone.0277204](https://doi.org/10.1371/journal.pone.0277204)

---

## Install

**Latest release: v2.0**

| Platform | Download |
|---|---|
| Windows (GUI + batch) | [Download EXE — v2.0](https://github.com/malewick/frame/releases/latest) |
| All platforms (source) | [GitHub repository](https://github.com/malewick/frame) |

For source installation instructions see the [README](https://github.com/malewick/frame#install).

---

## Quick start — graphical interface

The program works in five steps:

| Step | Screenshot |
|---|---|
| 1. Load data on your measured samples | [<img width="500" alt="screenshot loading data" src="https://user-images.githubusercontent.com/24914567/204164202-4426b6dc-0f3d-4f73-81bf-cbb5fe2ce698.png">](https://user-images.githubusercontent.com/24914567/204164202-4426b6dc-0f3d-4f73-81bf-cbb5fe2ce698.png) |
| 2. Load isotopic signatures of considered sources | [<img width="500" alt="screenshot loading sources" src="https://user-images.githubusercontent.com/24914567/204164209-0d958310-640f-4eb0-a50a-664395ad0a7a.png">](https://user-images.githubusercontent.com/24914567/204164209-0d958310-640f-4eb0-a50a-664395ad0a7a.png) |
| 3. (Optional) Load auxiliary fractionation parameters | [<img width="500" alt="screenshot loading fractionation" src="https://user-images.githubusercontent.com/24914567/204164214-d3dd12e6-0f70-4ec7-a802-e87e64018f40.png">](https://user-images.githubusercontent.com/24914567/204164214-d3dd12e6-0f70-4ec7-a802-e87e64018f40.png) |
| 4. Verify your loaded data | [<img width="500" alt="screenshot data plotted" src="https://user-images.githubusercontent.com/24914567/204164250-67b016b9-c780-4cad-ad26-68a0aa868a9b.png">](https://user-images.githubusercontent.com/24914567/204164250-67b016b9-c780-4cad-ad26-68a0aa868a9b.png) |
| 5. Run the model | [<img width="260" alt="screenshot run model" src="https://user-images.githubusercontent.com/24914567/204164219-64025d4e-db2d-45cf-b668-3ae0bc5d4816.png">](https://user-images.githubusercontent.com/24914567/204164219-64025d4e-db2d-45cf-b668-3ea0bc5d4816.png) |

Voilà — the simulation is running!

[<img width="800" alt="screenshot model running" src="https://user-images.githubusercontent.com/24914567/204164285-dd67e181-00f9-4abe-9c34-34ea7ffe56fe.png">](https://user-images.githubusercontent.com/24914567/204164285-dd67e181-00f9-4abe-9c34-34ea7ffe56fe.png)

**A full user guide with input file formats, model equation syntax and output interpretation is [here](user_guide.md).**

The example input files are included with the release. Real-world datasets are available in [`input/real world examples`](https://github.com/malewick/frame/tree/main/input/real%20world%20examples).

---

## Batch / scripting mode

FRAME can be run without the GUI — useful for processing many samples or automating analyses.

**Windows (PowerShell) — using the pre-built executable:**

```powershell
.\FRAME_batch.exe data.csv sources.csv --aux_file frac.csv `
    --output_dir results --niter 500000 --burnout 200 --chain_length 500 `
    --plot_online False
```

With a log-uniform prior on the fractionation variable `r`:

```powershell
.\FRAME_batch.exe data.csv sources.csv --aux_file frac.csv `
    --aux_prior_type loguniform --aux_r_min 0.001 --aux_r_max 1.0 `
    --output_dir results --plot_online False
```

**All platforms — from source:**

```bash
python run/run_batch.py data.csv sources.csv --aux_file frac.csv \
    --output_dir results --niter 500000 --chain_length 500 --plot_online False
```

| Argument | Default | Description |
|---|---|---|
| `--aux_file` | *(none)* | Path to auxiliary parameters CSV |
| `--output_dir` | `output` | Directory for results |
| `--output_filenames` | *(auto)* | Custom tag for output file names |
| `--output_formats` | `pdf,png` | Comma-separated plot formats |
| `--niter` | `1000000` | Maximum MCMC iterations |
| `--burnout` | `100` | Burn-in steps |
| `--chain_length` | `500` | Accepted steps to record |
| `--plot_online` | `True` | Live plotting (`True`/`False`) |
| `--aux_prior_type` | `uniform` | Prior for `r`: `uniform` or `loguniform` |
| `--aux_r_min` | `0.0` | Lower bound of the `r` prior |
| `--aux_r_max` | `1.0` | Upper bound of the `r` prior |

A worked Colab notebook is available: [docs/colab_demo.ipynb](https://github.com/malewick/frame/blob/main/docs/colab_demo.ipynb).

---

## Discussion and bugs

- **Questions / discussion:** mailing list at [groups.google.com/g/frame-isotopes](https://groups.google.com/g/frame-isotopes)
- **Bug reports:** issue tracker at [github.com/malewick/frame/issues](https://github.com/malewick/frame/issues)

---

## Attribution

Please cite FRAME in publications using:

> Lewicki MP, Lewicka-Szczebak D, Skrzypek G (2022) *FRAME — Monte Carlo model for evaluation of the stable isotope mixing and fractionation.* PLoS ONE 17(11): e0277204. [https://doi.org/10.1371/journal.pone.0277204](https://doi.org/10.1371/journal.pone.0277204)

---

## License

FRAME is [GPL v3](https://github.com/malewick/frame/blob/main/LICENSE) licensed. Please cite our work when using it and consider contributing your changes back.

FRAME is provided free of charge and is distributed in the hope that it will be useful. This software is provided "as is" and without warranty. Use at your own risk.

---

## Authors

- **Maciej Lewicki** — malewick[at]cern.ch — [malewick.web.cern.ch](https://malewick.web.cern.ch/)
- **Dominika Lewicka-Szczebak** — dominika.lewicka-szczebak[at]uwr.edu.pl
- **Grzegorz Skrzypek** — grzegorz.skrzypek[at]uwa.edu.au — [gskrzypek.com](http://www.gskrzypek.com/)
