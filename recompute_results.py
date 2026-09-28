"""Reproduce the paper's pooled RMSE and revised figures.

Run with the corrected PR-ZIM simulation script and its replicate CSV:
python recompute_results.py --simulation /path/to/simulation_full \
    --sensitivity-prefix /path/to/sensitivity --outdir .
The three sensitivity directories have suffixes _005, _020, _050.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY=["experiment","scenario","n","rho","q_target","method"]

def summarize(path):
    raw=pd.read_csv(path)
    success=raw.loc[raw.fit_success.eq(1)]
    out=success.groupby(KEY,dropna=False).param_rmse.agg(
        n_success="count",mean_sq=lambda z:np.mean(z**2),
        sd_sq=lambda z:np.std(z**2,ddof=1)).reset_index()
    out["pooled_rmse"]=np.sqrt(out.mean_sq)
    out["pooled_mcse"]=out.sd_sq/(2*out.pooled_rmse*np.sqrt(out.n_success))
    pred=success.groupby(KEY,dropna=False).pred_rmse.agg(
        pred_n="count",pred_mean_sq=lambda z:np.mean(z**2),
        pred_sd_sq=lambda z:np.std(z**2,ddof=1)).reset_index()
    out=out.merge(pred,on=KEY,validate="1:1")
    out["pooled_pred_rmse"]=np.sqrt(out.pred_mean_sq)
    out["pooled_pred_mcse"]=out.pred_sd_sq/(2*out.pooled_pred_rmse*np.sqrt(out.pred_n))
    attempt=raw.groupby(KEY,dropna=False).size().rename("n_attempt").reset_index()
    out=out.merge(attempt,on=KEY,validate="1:1")
    out["failure_rate"]=1-out.n_success/out.n_attempt
    return raw,out

def figures(raw,out,folder):
    for scenario,suffix in [("clean","clean"),("informative_H","informative")]:
        tab=out[(out.experiment=="A")&(out.n==2500)&(out.scenario==scenario)]
        fig,ax=plt.subplots(figsize=(5.1,3.8))
        for method,color in [("PR_MLE","#1769a2"),("DR_CF","#c65f28")]:
            sub=tab[tab.method==method].sort_values("rho")
            ax.errorbar(sub.rho,sub.pooled_rmse,yerr=sub.pooled_mcse,
                        fmt="o-",color=color,lw=1.6,ms=4.5,capsize=2.5,label=method)
        std=tab[tab.method=="STD_ZIB"].iloc[0]
        ax.axhline(std.pooled_rmse,color="#666666",ls="--",lw=1.5,
                   label="STD_ZIB")
        ax.set_xscale("log");ax.set_xticks([.01,.02,.05,.1,.2])
        ax.set_xticklabels([".01",".02",".05",".10",".20"])
        ax.set(xlabel=r"Revelation probability $\rho$",
               ylabel="Pooled parameter RMSE")
        ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/f"fig1_rmse_{suffix}_20260928.pdf",bbox_inches="tight")
        plt.close(fig)

        eig=(raw[(raw.experiment=="A")&(raw.n==2500)&
                 (raw.scenario==scenario)&raw.fit_success.eq(1)]
             .groupby(["rho","method"]).min_eig_or_sing.mean().reset_index())
        pr=eig[eig.method=="PR_MLE"].sort_values("rho")
        std=eig[eig.method=="STD_ZIB"].iloc[0]
        fig,ax=plt.subplots(figsize=(5.1,3.8))
        ax.plot(pr.rho,pr.min_eig_or_sing,"o-",ms=4.5,lw=1.6,
                color="#1769a2",label="PR_MLE")
        ax.axhline(std.min_eig_or_sing,color="#666666",ls="--",lw=1.5,
                   label=r"STD_ZIB ($\rho=0$ baseline)")
        ax.set_xscale("log");ax.set_xticks([.01,.02,.05,.1,.2])
        ax.set_xticklabels([".01",".02",".05",".10",".20"])
        ax.set(xlabel=r"Revelation probability $\rho$",
               ylabel="Mean smallest curvature eigenvalue")
        ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/f"fig2_mineig_{suffix}_20260928.pdf",bbox_inches="tight")
        plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--simulation",type=Path,required=True)
    p.add_argument("--sensitivity-prefix",type=Path,required=True)
    p.add_argument("--outdir",type=Path,default=Path("."))
    args=p.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    raw,pooled=summarize(args.simulation/"replicate_results.csv")
    pooled.to_csv(args.outdir/"pooled_method_summary.csv",index=False)
    figures(raw,pooled,args.outdir)
    records=[]
    for key,epsilon in [("005",.005),("020",.02),("050",.05)]:
        dat,summary=summarize(Path(str(args.sensitivity_prefix)+"_"+key)/
                              "replicate_results.csv")
        for _,r in summary[(summary.experiment=="A")&
                           (summary.scenario=="informative_H")&
                           (summary.n==2500)&(summary.rho==.1)].iterrows():
            fits=dat[(dat.method==r.method)&dat.fit_success.eq(1)]
            records.append({"epsilon":epsilon,"method":r.method,
                            "rmse":r.pooled_rmse,"mcse":r.pooled_mcse,
                            "failure_rate":r.failure_rate,
                            "realized_min_e":fits.true_e_min_zero.mean(),
                            "ess":fits.ess.mean()})
    pd.DataFrame(records).to_csv(args.outdir/"positivity_sensitivity.csv",
                                 index=False)
if __name__=="__main__":
    main()
