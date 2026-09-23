# Partial revelation and doubly robust inference in zero-inflated Bernoulli models

This repository contains the computational code accompanying the manuscript
**“Partial revelation and doubly robust inference in zero-inflated Bernoulli models.”**

The code corresponds to the reported analysis. In particular, the simulation study uses a revelation propensity that is bounded away from zero and one, the AIPW pseudo-gate is not clipped to the unit interval, the DR estimator solves the estimating equations directly, and the NHANES analysis uses the final operational gate/recognition specification reported in the manuscript.

## Repository contents

- **`01_simulation_study.ipynb`** (`01_simulation_study.py`)  
  Reproduces the main Monte Carlo study and the supplementary NHANES-like high-revelation / rare-positive-label experiment. The reported design uses 500 Monte Carlo replications, five-fold cross-fitting, fixed random seeds, and `n = 200, 500, 2500` with `rho = 0.01, 0.02, 0.05, 0.10, 0.20`. It also produces coefficient-level summaries and revelation/ESS diagnostics.

- **`01_simulation_study_bootstrap.ipynb`** (`01_simulation_study_bootstrap.py`)  
  Reproduces the targeted bootstrap experiment in the Supplementary Materials. For the four reported informative-history cells `(n, rho) = (2500, 0.02), (2500, 0.05), (2500, 0.10), (500, 0.20)`, each of 500 Monte Carlo samples is bootstrapped `B = 1000` times. Every resample refits PR_MLE and, for DR_CF, re-estimates the nuisance functions and resolves the cross-fitted estimating equations. Checkpoint/resume is enabled.

- **`02_nhanes_analysis.ipynb`** (`02_nhanes_analysis.py`)  
  Reproduces the NHANES 2017–March 2020 pre-pandemic diabetes analysis. The primary analysis uses HbA1c, with FPG as a sensitivity definition. The gate model uses age, BMI, and sex; the recognition model uses insurance, age, BMI, and sex. Usual source of care remains in the nuisance history but is excluded from the primary unpenalized recognition block because the full model is separation-prone. The script also produces the full-covariate sensitivity diagnostics reported in the Supplementary Materials.

- **`requirements.txt`**  
  Python dependencies used by the analysis.

- **`reference_results/`**  
  Compact CSV summaries of the reported results, provided as validation targets for a fresh run.

The `.py` and `.ipynb` versions contain the same analysis code. The notebooks are convenient for Google Colab; the scripts are convenient for command-line execution and version control.

## Mapping to the manuscript

| Manuscript output | Code |
| --- | --- |
| Main simulation table and RMSE/curvature figures | `01_simulation_study` |
| Supplementary coefficient-level tables | `01_simulation_study` |
| Supplementary sample-size results | `01_simulation_study` |
| Supplementary NHANES-like simulation | `01_simulation_study` |
| Supplementary targeted `B=1000` bootstrap table | `01_simulation_study_bootstrap` |
| Main NHANES descriptive, coefficient, and sanity-check tables | `02_nhanes_analysis` |
| Supplementary NHANES sensitivity diagnostics | `02_nhanes_analysis` |

## Statistical implementation

For the main simulation, the zero-layer revelation propensity is generated as

```text
e0(H) = eps + (1 - 2 eps) expit(c_rho + gamma^T H),  eps = 0.005,
```

with `c_rho` calibrated so that the mean revelation probability among observed zeros equals the target `rho`. Thus the main data-generating mechanism satisfies the stated positivity condition by construction.

`DR_CF` uses five-fold cross-fitting. The revelation propensity is fitted with the bounded-logistic family used in the data-generating mechanism, whereas the zero-layer gate regression is an ordinary logistic working model. The AIPW pseudo-gate is left unprojected, and the reported primary estimator does not use statistical propensity truncation.

The NHANES code implements all four observed-data likelihood patterns, including revealed operational negatives `(Y=0, R=1, W=0)`. Analysis weights are used as fixed weights. Consequently, the reported NHANES standard errors are weighted model/estimating-equation standard errors rather than full NHANES design-based standard errors.

## Software

The analysis is written in Python 3 and uses NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib, and joblib. The code was run with up to seven worker processes; BLAS threads are restricted to one per worker to avoid oversubscription.

Install the dependencies with

```bash
python -m pip install -r requirements.txt
```

## Reproducing the reported results

From the repository root:

```bash
python 01_simulation_study.py
python 01_simulation_study_bootstrap.py
python 02_nhanes_analysis.py
```

The full simulation and especially the `B=1000` bootstrap are computationally intensive. For smoke tests:

```bash
PRZIB_MODE=quick python 01_simulation_study.py
PRZIB_BOOT_QUICK=1 python 01_simulation_study_bootstrap.py
```

The number of parallel workers can be changed with `PRZIB_N_JOBS`, for example

```bash
PRZIB_N_JOBS=4 python 01_simulation_study.py
```

Random seeds are fixed in the scripts. Floating-point optimization can lead to very small platform-dependent differences, especially in numerically difficult low-revelation cells.

## NHANES data

The NHANES 2017–March 2020 pre-pandemic XPT files are publicly available from CDC/NCHS. `02_nhanes_analysis.py` downloads the required demographic, diabetes questionnaire, health-insurance, health-care-access, body-measures, HbA1c, and fasting-glucose files directly from the CDC/NCHS public data server.

The analysis uses the MEC examination weight for HbA1c and the fasting-subsample weight for FPG, each rescaled to have mean one within the analytic sample. A laboratory measurement is treated as exact only relative to the prespecified operational classification used in the paper, not as error-free biological disease status.

## Output directories

By default the scripts write to

```text
results/simulation/
results/bootstrap/
results/nhanes/
```

The locations can be changed with `PRZIB_OUTDIR`, `PRZIB_BOOT_OUTDIR`, and `PRZIB_NHANES_OUTDIR`.

## License

The repository retains the existing MIT license.
