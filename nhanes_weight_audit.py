"""Check the NHANES denominator identity from the public PR-ZIM analysis code.

Usage: python nhanes_weight_audit.py --repo /path/to/PR-ZIM --output audit.csv
The upstream script downloads the public 2017--March 2020 NHANES XPT files.
"""
import argparse
import importlib.util
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("nhanes_denominator_audit.csv"))
    a=ap.parse_args()
    spec=importlib.util.spec_from_file_location(
        "nhanes_upstream",a.repo/"02_nhanes_analysis.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dat=mod.load_nhanes()
    rows=[]
    for mode in ["A1C","FPG"]:
        base,dvar,wvar=mod.make_analysis_base(dat,mode)
        w=base[wvar].to_numpy(float)
        y=base.Y.to_numpy(float)
        d=base[dvar].to_numpy(float)
        observed=base[dvar].notna().to_numpy()
        case=(d==1)&observed
        noncase=(y==0)
        reveal=base.R.to_numpy(float)
        positive=base.W.to_numpy(float)
        measured=mod.weighted_mean(observed,w)
        prevalence=mod.weighted_mean(d[observed],w[observed])
        noncase_given_case=mod.weighted_mean(noncase[case],w[case])
        py0=mod.weighted_mean(noncase,w)
        rho=mod.weighted_mean(reveal[noncase],w[noncase])
        q=mod.weighted_mean(positive[noncase&(reveal==1)],
                            w[noncase&(reveal==1)])
        left=measured*prevalence*noncase_given_case
        right=py0*rho*q
        rows.append(dict(mode=mode,pD_given_measured=prevalence,
                         pY0_given_D1=noncase_given_case,
                         p_measured=measured,rho_weighted=rho,
                         q_weighted=q,pY0=py0,left=left,right=right))
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    main()
