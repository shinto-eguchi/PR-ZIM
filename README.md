# Partial revelation for zero-inflated Bernoulli models

This repository contains the computational code for the manuscript

**Partial revelation and doubly robust inference for zero-inflated Bernoulli models.**

## Contents

- `01_simulation_study.ipynb`  
  Reproduces the main numerical experiments comparing the standard
  zero-inflated Bernoulli estimator, the partial-revelation likelihood
  estimator, and the cross-fitted doubly robust estimator.

- `01_simulation_study_bootstrap.ipynb`  
  Provides additional bootstrap-based simulation diagnostics.  In particular,
  it examines the sensitivity of the cross-fitted doubly robust estimator to
  propensity-score truncation and computes bootstrap percentile intervals.
  It also includes bootstrap coverage calculations for the standard
  zero-inflated Bernoulli estimator and the partial-revelation likelihood
  estimator.

- `02_nhanes_analysis.ipynb`  
  Reproduces the NHANES diabetes analysis based on self-reported diabetes and
  laboratory-based auxiliary information.

## Software

The analyses were conducted in Python using Jupyter notebooks. The outputs are
retained in the notebooks. The main packages used are

- numpy
- pandas
- scipy
- statsmodels
- scikit-learn
- matplotlib
- seaborn
- joblib

## Data

The NHANES data are publicly available from CDC/NCHS. The NHANES notebook
describes the variables used in the analysis and the preprocessing steps.

## Reproducibility

The numerical results in the manuscript can be reproduced by running the
notebooks as follows.

1. Run `01_simulation_study.ipynb` for the main simulation results.
2. Run `01_simulation_study_bootstrap.ipynb` for the additional bootstrap
   diagnostics and bootstrap coverage results.
3. Run `02_nhanes_analysis.ipynb` for the NHANES data analysis.

Random seeds are fixed in the notebooks. Some simulation results may vary
slightly depending on the computing environment and package versions.
The bootstrap simulation notebook may take substantially longer to run than
the main simulation notebook, especially when using 100 Monte Carlo replications
and 100 bootstrap replications.
