"""Run the paper's added sensitivity analyses using the public PR-ZIM code.

Examples:
  python reproduce_additional_analyses.py --repo ../PR-ZIM --kind positivity
  python reproduce_additional_analyses.py --repo ../PR-ZIM --kind equal-weight
The script creates temporary modified upstream scripts; it does not edit the
upstream repository or submit any data to an external service.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

def change(source, old, new):
    assert old in source, "Unexpected upstream code: "+old
    return source.replace(old,new,1)

def positivity(source,epsilon):
    source=change(source,"RUN_EXPERIMENT_B = True",
                  "RUN_EXPERIMENT_B = False")
    source=change(source,"N_GRID_A = [200, 500, 2500]",
                  "N_GRID_A = [2500]")
    source=change(source,
        "RHO_GRID_A = [0.01, 0.02, 0.05, 0.10, 0.20]",
        "RHO_GRID_A = [0.10]")
    source=change(source,
        'SCENARIOS_A = ["clean", "informative_H"]',
        'SCENARIOS_A = ["informative_H"]')
    # Retain the original informative-history scenario index of one.
    # Thus epsilon=.005 gives exactly the matching main-simulation cell.
    source=change(source,
        "for si,scenario in enumerate(SCENARIOS_A):",
        'for si,scenario in [(1,"informative_H")]:')
    source=change(source,"POSITIVITY_EPS = 0.005",
                  "POSITIVITY_EPS = "+str(epsilon))
    return source

def equal_weight(source):
    source=change(source,'DVAR, WVAR = "D_A1C", "WTMECPRP"',
                  'DVAR, WVAR = "D_A1C", "UNIT_W"')
    source=change(source,"dat = load_nhanes()",
                  'dat = load_nhanes()\n    dat["UNIT_W"] = 1.0')
    source=change(source,'enumerate(["A1C", "FPG"])',
                  'enumerate(["A1C"])')
    source=change(source,"RUN_FULL_COVARIATE_SENSITIVITY = True",
                  "RUN_FULL_COVARIATE_SENSITIVITY = False")
    return source

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--kind",choices=["positivity","equal-weight"],required=True)
    ap.add_argument("--outdir",type=Path,default=Path("."))
    ap.add_argument("--jobs",type=int,default=2)
    a=ap.parse_args()
    a.outdir=a.outdir.resolve()
    a.outdir.mkdir(parents=True,exist_ok=True)
    source=a.repo/("01_simulation_study.py" if a.kind=="positivity"
                   else "02_nhanes_analysis.py")
    raw=source.read_text()
    with tempfile.TemporaryDirectory(prefix="przib_repro_") as temp:
        work=Path(temp)
        if a.kind=="positivity":
            for epsilon,key in [(.005,"005"),(.02,"020"),(.05,"050")]:
                code=work/("sensitivity_"+key+".py")
                code.write_text(positivity(raw,epsilon))
                env=os.environ.copy()
                env["PRZIB_N_JOBS"]=str(a.jobs)
                env["PRZIB_OUTDIR"]=str(a.outdir/("sensitivity_matched_"+key))
                subprocess.run(["python",str(code)],cwd=a.repo,env=env,check=True)
        else:
            code=work/"nhanes_unweighted.py"
            code.write_text(equal_weight(raw))
            env=os.environ.copy()
            env["PRZIB_NHANES_OUTDIR"]=str(a.outdir/"nhanes_unweighted")
            subprocess.run(["python",str(code)],cwd=a.repo,env=env,check=True)

if __name__=="__main__":
    main()
