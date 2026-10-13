# Reproducing the revised PR-ZIB results

## Fixed source version

The submission version, including the survey-weighted PR_MLE score-sandwich
standard errors in Table 7 and the revised axes of Figures 1--2, is the
GitHub release tag **`revision-1`**. The exact commit SHA is displayed by
GitHub under that tag; the tag identifies the same immutable commit once
published. Do not use the earlier `489a66f` commit to reproduce Table 7.

```bash
git clone https://github.com/shinto-eguchi/PR-ZIM.git
cd PR-ZIM
git checkout revision-1
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

## Table 7: PR_MLE score-sandwich reproducibility (revision-1)

The original `02_nhanes_analysis.py` computes curvature-based PR_MLE SEs.
The paper's revised Table 7 instead reports the fixed-analysis-weight,
independent-observation **score-sandwich** SEs. Reproduce the revised table
from the original CDC/NCHS data as follows:

```bash
python 02_nhanes_analysis.py
python recompute_PR_sandwich_SE.py
python apply_PR_sandwich_csv.py --input results/nhanes/all_coefficients.csv \
  --reference results/pr_sandwich/PR_MLE_weighted_sandwich_SE.csv
```

`PR_MLE_weighted_sandwich_SE.csv` is the fixed reference for all 18 PR
coefficients, while `nhanes_reported_coefficients.csv` and
`reference_results/nhanes_reported_coefficients.csv` contain the updated
published SEs. The transformation preserves all other rows and estimates.
For FPG (complete revelation), PR and DR SEs should agree to reported
precision if the same score sandwich is used. These SEs treat analysis
weights as fixed and do **not** account for NHANES strata or clusters;
they are not design-based survey SEs. Source and reference CSVs do not
redistribute restricted individual-level NHANES records.

## Figures 1 and 2 (revision-1)

The revised source `recompute_results.py` uses `Target revelation rate` on
both horizontal axes. To regenerate the four figure PDFs from the archived
replicate outputs (no new 500-replication fitting required):

```bash
python -m zipfile -e replication_data/simulation_replications_20260928.zip .
python recompute_results.py --simulation simulation_full \
  --sensitivity-prefix sensitivity_matched --outdir reference_results
cp reference_results/fig1_rmse_*_20260928.pdf .
cp reference_results/fig2_mineig_*_20260928.pdf .
```

The PDF copies in the repository root are those included in the manuscript.
The original September 2026 commit `489a66f` remains available as a
historical snapshot and is superseded for Table 7 and figure labels.

## Table 7: PR_MLE score-sandwich reproducibility (revision-1)

The original `02_nhanes_analysis.py` computes curvature-based PR_MLE SEs.
The paper's revised Table 7 instead reports the fixed-analysis-weight,
independent-observation **score-sandwich** SEs. Reproduce the revised table
from the original CDC/NCHS data as follows:

```bash
python 02_nhanes_analysis.py
python recompute_PR_sandwich_SE.py
python apply_PR_sandwich_csv.py --input results/nhanes/all_coefficients.csv \
  --reference results/pr_sandwich/PR_MLE_weighted_sandwich_SE.csv
```

`PR_MLE_weighted_sandwich_SE.csv` is the fixed reference for all 18 PR
coefficients, while `nhanes_reported_coefficients.csv` and
`reference_results/nhanes_reported_coefficients.csv` contain the updated
published SEs. The transformation preserves all other rows and estimates.
For FPG (complete revelation), PR and DR SEs should agree to reported
precision if the same score sandwich is used. These SEs treat analysis
weights as fixed and do **not** account for NHANES strata or clusters;
they are not design-based survey SEs. Source and reference CSVs do not
redistribute restricted individual-level NHANES records.

## Figures 1 and 2 (revision-1)

The revised source `recompute_results.py` uses `Target revelation rate` on
both horizontal axes. To regenerate the four figure PDFs from the archived
replicate outputs (no new 500-replication fitting required):

```bash
python -m zipfile -e replication_data/simulation_replications_20260928.zip .
python recompute_results.py --simulation simulation_full \
  --sensitivity-prefix sensitivity_matched --outdir reference_results
cp reference_results/fig1_rmse_*_20260928.pdf .
cp reference_results/fig2_mineig_*_20260928.pdf .
```

The PDF copies in the repository root are those included in the manuscript.
The original September 2026 commit `489a66f` remains available as a
historical snapshot and is superseded for Table 7 and figure labels.
