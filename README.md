\# Partial revelation for zero-inflated Bernoulli models

This repository contains the computational code for the manuscript

Partial revelation and doubly robust inference for zero-inflated Bernoulli models.

\#\# Contents

\- \`notebooks/01\_simulation\_study.ipynb\`    
  Reproduces the numerical experiments comparing the standard zero-inflated  
  Bernoulli estimator, the partial-revelation likelihood estimator, and the  
  cross-fitted doubly robust estimator.

\- \`notebooks/02\_nhanes\_analysis.ipynb\`    
  Reproduces the NHANES diabetes analysis based on self-reported diabetes and  
  laboratory-based auxiliary information.

\#\# Software

The analyses were conducted in Python using Jupyter notebooks. The outputs  
are retained in the notebooks. The main packages used are numpy, pandas,  
scipy, statsmodels, scikit-learn, matplotlib, and seaborn.

\#\# Data

The NHANES data are publicly available from CDC/NCHS. The NHANES notebook  
describes the variables used in the analysis and the preprocessing steps.

\#\# Reproducibility

The numerical results in the manuscript can be reproduced by running the two  
notebooks in the following order:

1\. \`notebooks/01\_simulation\_study.ipynb\`  
2\. \`notebooks/02\_nhanes\_analysis.ipynb\`

Random seeds are fixed in both notebooks. Some simulation results may vary  
slightly depending on the computing environment and package versions.  
