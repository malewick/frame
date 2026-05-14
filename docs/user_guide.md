# FRAME — User Guide
## FRactionation And Mixing Evaluation

---

## Starting the program

**Windows executable:** double-click `FRAME.exe`. A console window appears first (normal) and the GUI launches after a few seconds.

**From source:**
```bash
python run/FRAME.py
```

---

## Main interface

![screenshot0](https://user-images.githubusercontent.com/24914567/118247339-5bef2400-b4a3-11eb-9928-9b9d0d2f8916.png)

---

## Loading input data

### 1. Load samples

The measured isotopic signatures of the samples are provided as a CSV file with the following columns:

| Column | Description |
|---|---|
| `group` | Integer group number; samples in the same group are modelled together |
| `label` | Free-text identifier for each sample |
| `δ¹⁸O`, `δ¹⁵N`, … | Measured isotopic δ values (one column per isotope) |
| `stdev(δ¹⁸O)`, … | Analytical measurement uncertainty (1σ) for each isotope |

### 2. Load sources

Each source is one row. Required columns:

| Column | Description |
|---|---|
| `source` | Name or abbreviation of the source |
| `δ¹⁸O`, `δ¹⁵N`, … | Mean isotopic signature (column names must match the data file) |
| `spread(δ¹⁸O)`, … | Half-width of the uniform range: the source is treated as equally probable in [mean − spread, mean + spread]. Set to 0 to omit. |
| `stdev(δ¹⁸O)`, … | Analytical uncertainty of the source determination (Gaussian margin added outside the uniform range). Set to 0 to omit. |

**Choosing between `spread` and `stdev`:**
- Use `spread` when the source has genuine *natural variability* — a range of equally probable values.
- Use `stdev` for a precisely determined source value with analytical measurement uncertainty.
- Both can be used simultaneously for the same source.

### 3. Load auxiliary data (optional)

This file adds fractionation or other auxiliary parameters to the model. It has a special structure:

- **Row 1** — model equation (see below)
- **Row 2** — column headers: `name` plus the same isotope columns as in the data/sources files (`δ¹⁸O`, `stdev(δ¹⁸O)`, `spread(δ¹⁸O)`, …)
- **Subsequent rows** — one row per auxiliary parameter (e.g. the fractionation factor `E`)

Once an aux file is loaded, the **"Aux. variables" panel** in the GUI shows a prior selector for each unknown variable (`r`, `r1`, `r2`, …):

| Control | Description |
|---|---|
| Dropdown | `uniform` — samples `r ~ Uniform[min, max]`; `loguniform` — samples `log(r) ~ Uniform[log(min), log(max)]` (requires min > 0) |
| min field | Lower bound (default 0.0) |
| max field | Upper bound (default 1.0) |

---

## Model equation syntax

The model equation defines how the final isotopic signature `M` is calculated from the source mixture `M0` and any auxiliary parameters. It is entered in the first row of the aux file.

Variables available in the equation:

| Symbol | Meaning |
|---|---|
| `M0[i]` | Weighted isotopic mixture of all sources for isotope dimension `i` |
| `r` | Unknown auxiliary variable (e.g. residual fraction); use `r1`, `r2` for multiple unknowns |
| `A[i]`, `B[i]`, … | Auxiliary parameters loaded from the aux file (capital letters) |
| `log(x)` | Natural logarithm |

### Common fractionation equations

**Open-system fractionation** (steady-state, e.g. denitrification in an open system):
```
M = M0[i] + E[i]*r
```
where `E[i]` is the fractionation factor and `r` is the unreacted fraction.

**Closed-system (Rayleigh-type) fractionation** (e.g. N₂O reduction):
```
M = M0[i] + E[i]*log(r)
```
where `r` is the residual unreacted fraction (0, 1].

**Equilibrium fractionation** (e.g. isotope exchange with water):
```
M = M0[i]*(1 - r*A[i]) + D[i]*r*A[i]
```
where `D[i]` is the isotopic signature after complete equilibration, `r` is the equilibrated fraction and `A[i]` can encode isotope-specific weights (1 for equilibrating isotopes, 0 for non-equilibrating ones).

Example files for all three cases are included in the `input/` directory.

---

## Running the model

Set the MCMC parameters in the control panel:

| Parameter | Description | Typical value |
|---|---|---|
| Max iterations | Upper limit on total MCMC steps | 1 000 000 |
| Burn-in | Initial steps discarded (stabilisation phase) | 100 |
| Chain length | Number of accepted steps to record | 500 |

Click **Run model**. The interface switches to the "Running MCMC" tab showing the progress bar and console output. Three diagnostic plots update in real time (turn off online plotting to speed up batch runs over many samples).

![screenshot1](https://user-images.githubusercontent.com/24914567/118250026-52b38680-b4a6-11eb-9c6a-681721fce6aa.png)

---

## Output

### Diagnostic plots

| Plot | What to look for |
|---|---|
| **Time series** | The chain should resemble stable random noise around a mean. Drift or step-changes indicate insufficient burn-in or a multimodal likelihood. |
| **Correlation plot** | Histograms on the diagonal show the posterior distribution of each variable. Off-diagonal panels show pairwise correlations — negative correlations between fractions are expected (they must sum to 1). |
| **Path plot** | The cloud of accepted model values in isotope space should be centred on the measurement. A sharp boundary at the source polygon edge is normal. If the measurement lies well outside the cloud, the model may be incomplete. |

### Results CSV

For each sample group a CSV file is saved with the mean, median and 68% / 95% confidence intervals for each evaluated variable (`f_source1`, `f_source2`, …, `r`, …). A z-score is also reported as a goodness-of-fit indicator.

---

## Saving and loading configurations

The current setup (data file paths, model equation, iteration parameters) can be saved to an XML file via **File → Save** and reloaded later with **File → Load**.

---

## Batch / headless mode

For processing many samples or scripting, FRAME can be run without the GUI.

**Windows (PowerShell):**
```powershell
.\FRAME_batch.exe data.csv sources.csv --aux_file frac.csv `
    --output_dir results --niter 500000 --burnout 200 --chain_length 500 `
    --plot_online False
```

**Python (all platforms):**
```bash
python run/run_batch.py data.csv sources.csv --aux_file frac.csv \
    --aux_prior_type loguniform --aux_r_min 0.001 --aux_r_max 1.0 \
    --output_dir results --plot_online False
```

Full argument reference: see the [README on GitHub](https://github.com/malewick/frame#windows--powershell-executable).

An interactive worked example is available as a [Colab notebook](https://github.com/malewick/frame/blob/main/docs/colab_demo.ipynb).

---

## Tips

- **Multiple samples:** put them in the same data CSV with different `group` numbers. Each group is modelled independently.
- **Turning off online plotting** (`--plot_online False`) is strongly recommended for batch runs — it can reduce computation time by an order of magnitude.
- **Log-uniform prior for `r`:** if your fractionation variable is expected to be small (e.g. < 10 %) but in principle could span several decades, a log-uniform prior explores the parameter space more evenly than a uniform one. Set it from the GUI prior dropdown or pass `--aux_prior_type loguniform --aux_r_min 0.001` on the command line.
- **Insufficient chain:** if distributions look ragged, increase `chain_length`. If they are not centred (poor convergence), increase `burnout`.
