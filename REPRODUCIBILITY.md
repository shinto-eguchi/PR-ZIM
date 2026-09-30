# Reproducing the revised PR-ZIB results

## Fixed source version

The corrected analysis and reference results are available in
https://github.com/shinto-eguchi/PR-ZIM at commit
489a66ffa4d86404c11a84545547692d34b02600

This commit identifies the completed code-and-results correction. A later
commit adding this document does not change the version fixed here.

```bash
git clone https://github.com/shinto-eguchi/PR-ZIM.git
cd PR-ZIM
git checkout 489a66ffa4d86404c11a84545547692d34b02600
python -m pip install -r requirements.txt
```

## NHANES correction and rerun

Both `02_nhanes_analysis.py` and `02_nhanes_analysis.ipynb` restore stored
zero survey weights that some XPT readers decode as 2**(-260). Nonnegative
values below 1e-50 are restored to zero before positive-weight selection,
age/BMI standardization, and model fitting. The correction implements the
positive-weight cohort rule consistently.

```bash
python 02_nhanes_analysis.py
```

The program downloads the public NHANES 2017–March 2020 pre-pandemic XPT
files and writes its results to `results/nhanes/` by default. The notebook
contains the same analysis code and is suitable for Google Colab.

| Generated output | Published reference |
| --- | --- |
| `results/nhanes/all_descriptive.csv` | `reference_results/nhanes_reported_descriptive.csv` |
| `results/nhanes/all_coefficients.csv` | `reference_results/nhanes_reported_coefficients.csv` |
| `results/nhanes/all_sanity_checks.csv` | `reference_results/nhanes_reported_sanity_checks.csv` |
| `results/nhanes/all_diagnostics.csv` | `reference_results/nhanes_reported_diagnostics.csv` |

The corrected FPG primary cohort has 3,622 records: 608 reported cases and
3,014 reported non-cases. Among non-cases, 157 have revealed S=1 and 2,857
have revealed S=0; none are unresolved. Both the unweighted sample and
survey-weighted revelation rates are one. The primary PR/DR point estimates
agree at the displayed precision. FPG curvature condition numbers are
780.1 (observed ZIB) and 128.1 (PR). HbA1c has 8,114 analytic records.

The primary gate uses age, BMI, and sex. The recognition model also includes
insurance. Usual source of care remains in the revelation-history model and
is excluded from the primary recognition block because the unpenalized
full-covariate model shows quasi-complete separation, especially for HbA1c.
Random seeds are fixed. Small platform-dependent optimization differences
can occur, particularly in separation-prone full-covariate fits.

The denominator audit imports the corrected analysis program:

```bash
python nhanes_weight_audit.py --repo . --output reference_results/nhanes_denominator_audit.csv
```

## Simulation reproduction

The current simulation scripts and notebooks already calculate pooled
parameter and product-prediction RMSE by averaging squared errors before
taking the square root. No additional pooled-RMSE patch is needed at the
fixed source version above.

To recompute the exact published simulation summaries without rerunning the
Monte Carlo fits, extract
`replication_data/simulation_replications_20260928.zip` into the repository
root. It provides `simulation_full/` and
`sensitivity_matched_{005,020,050}/`. Then run:

```bash
python recompute_results.py --simulation simulation_full --sensitivity-prefix sensitivity_matched --outdir reference_results
```

To rerun the main simulations and targeted bootstrap instead:

```bash
python 01_simulation_study.py
python 01_simulation_study_bootstrap.py
```

These use 500 Monte Carlo replications and B=1000 for the targeted bootstrap;
the full calculations are computationally intensive. See `README.md` for
quick modes, output directories, and the additional positivity and
equal-weight sensitivity experiments. Individual NHANES participant records
are downloaded from CDC/NCHS and are not redistributed in this repository.
