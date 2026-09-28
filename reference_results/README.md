# Reference results for the revised PR-ZIB manuscript

These CSVs and four vector figures match the 2026-09-28 manuscript revision.
The simulation and NHANES files supersede the earlier reference tables.
All pooled root mean squared errors take the square root AFTER averaging
squared errors across successful Monte Carlo replications.

- `simulation_reported_method_summary.csv`: 114 reported main and NHANES-like cells; `param_rmse_mean` and `pred_rmse_mean` now mean POOLED RMSE.
- `pooled_method_summary.csv`: all 126 available simulation design cells with the pooled parameter and product-prediction RMSEs and their Monte Carlo SEs.
- `simulation_reported_coefficient_summary_n2500.csv`: PR_MLE/DR_CF coefficient tables; includes true and mean estimates.
- `simulation_reported_std_coefficient_summary_n2500.csv`: one STD_ZIB coefficient table (the observed-data-only estimator is invariant to revelation rate and scenario).
- `bootstrap_reported_method_summary.csv`: the existing B=1000 coverage summaries with the original-sample `param_rmse_mean` and its Monte Carlo SE corrected to pooled RMSE. The eight original-sample sample means and success counts match the corresponding main-simulation cells within 2e-9 and exactly, respectively.
- `nhanes_reported_*.csv`: survey-weighted HbA1c and FPG fits, replacing the old reference files.
- `nhanes_equal_weight_*.csv`: equal-weight HbA1c sensitivity fit; it targets a different population.
- `nhanes_denominator_audit.csv`: direct weighted rho and the test-measurement denominator identity.
- `positivity_sensitivity.csv`: matched-seed epsilon={0.005,0.02,0.05} experiment.
- `fig1_*.pdf` and `fig2_*.pdf`: manuscript-ready pooled-RMSE and curvature plots.

The complete per-replication CSVs are archived at
`../replication_data/simulation_replications_20260928.zip`.
Extract that archive into the repository root to obtain `simulation_full/`
and `sensitivity_matched_{005,020,050}/`; these generated directories are
not required to run the code and are not included as loose tracked CSVs.
Recompute the tables and figures directly from the extracted raw outputs:

```bash
python recompute_results.py --simulation simulation_full --sensitivity-prefix sensitivity_matched --outdir reference_results
```

CDC NHANES XPT files are downloaded by `02_nhanes_analysis.py` when it runs;
individual NHANES participant records are not redistributed here.
