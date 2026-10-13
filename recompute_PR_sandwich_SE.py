"""Recompute the analysis-weighted PR sandwich standard errors.

Place this file in the root of https://github.com/shinto-eguchi/PR-ZIM at
commit 489a66f (or the later main branch with the same analysis code), then run:
    python recompute_PR_sandwich_SE.py

The program fetches the original public NHANES XPT inputs through the
unchanged 02_nhanes_analysis.py code and writes
    results/pr_sandwich/PR_MLE_weighted_sandwich_SE.csv

Does not change the reference code, original estimates, or table values.
This is an independent-observation, fixed-analysis-weight score sandwich;
it does NOT account for the NHANES strata or cluster design.
"""
from importlib import util
from pathlib import Path
import os
import numpy as np
import pandas as pd

def find_przim_root():
    """Find the GitHub repo whether run as a .py file or pasted in Colab."""
    candidates = []
    if os.environ.get("PRZIM_ROOT"):
        candidates.append(Path(os.environ["PRZIM_ROOT"]).expanduser())
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:  # Colab / Jupyter cell
        pass
    candidates.extend([
        Path.cwd(), Path.cwd() / "PR-ZIM",
        Path("/content/PR-ZIM"), Path("/content"),
        Path("/content/drive/MyDrive/PR-ZIM"),
    ])
    for root in candidates:
        if (root / "02_nhanes_analysis.py").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "02_nhanes_analysis.py が見つかりません。\n"
        "Colab で先に以下を実行してください。\n"
        "  !git clone https://github.com/shinto-eguchi/PR-ZIM.git /content/PR-ZIM\n"
        "  !git -C /content/PR-ZIM checkout 489a66f\n"
        "または環境変数 PRZIM_ROOT でリポジトリの場所を指定してください。"
    )

ROOT = find_przim_root()
print(f"Using PR-ZIM source: {ROOT}")
SPEC = util.spec_from_file_location("przim_nhanes", ROOT / "02_nhanes_analysis.py")
M = util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def pr_score_rows(theta, X, Z, y, R, W):
    """Unweighted per-subject score for corrected four-pattern PR model."""
    px = X.shape[1]
    pi = M.clip01(M.expit(X @ theta[:px]))
    mu = M.clip01(M.expit(Z @ theta[px:]))
    den = M.clip01(1 - pi * mu)
    ga, gb = np.zeros(len(y)), np.zeros(len(y))
    y1 = (y == 1)
    r1 = (y == 0) & (R == 1) & (W == 1)
    r0 = (y == 0) & (R == 1) & (W == 0)
    unresolved = (y == 0) & (R == 0)
    assert np.all((y1.astype(int) + r1.astype(int) + r0.astype(int)
                   + unresolved.astype(int)) == 1)
    ga[y1], gb[y1] = 1 - pi[y1], 1 - mu[y1]
    ga[r1], gb[r1] = 1 - pi[r1], -mu[r1]
    ga[r0], gb[r0] = -pi[r0], 0
    ga[unresolved] = -(mu[unresolved] * pi[unresolved]
                       * (1-pi[unresolved])) / den[unresolved]
    gb[unresolved] = -(pi[unresolved] * mu[unresolved]
                       * (1-mu[unresolved])) / den[unresolved]
    return np.hstack([X * ga[:, None], Z * gb[:, None]])


def run():
    dat = M.load_nhanes()
    results = []
    for imode, mode in enumerate(["A1C", "FPG"]):
        base, _, wvar = M.make_analysis_base(dat, mode)
        xcols, zcols = M.PRIMARY_X_COLS, M.PRIMARY_Z_COLS
        X, Z = base[xcols].to_numpy(float), base[zcols].to_numpy(float)
        y, R, W = (base[c].to_numpy(float) for c in ["Y", "R", "W"])
        w = M.normalize_weights(base[wvar].to_numpy(float))
        pr = M.fit_pr(X, Z, y, R, W, w, seed=M.SEED + 10000*imode + 2)
        score = pr_score_rows(pr["theta"], X, Z, y, R, W)
        # Assertion that per-subject scores match the public objective gradient.
        grad_reference = M.pr_nll_grad(pr["theta"], X, Z, y, R, W, w)[1]
        grad_rows = -np.sum(w[:, None] * score, axis=0)
        assert np.allclose(grad_reference, grad_rows, atol=1e-7, rtol=1e-7)
        H = pr["H"]  # Hessian of the SUM of w_i*(-log likelihood_i)
        B = (w[:, None] * score).T @ (w[:, None] * score)
        Hinv = M.stable_pinv(H)
        cov = Hinv @ B @ Hinv.T
        se = np.sqrt(np.maximum(np.diag((cov+cov.T)/2), 0))
        terms = [('alpha', term) for term in xcols] + [('beta', term) for term in zcols]
        for j, (block, term) in enumerate(terms):
            results.append({"mode":mode,"design":"primary_no_usualcare",
                            "method":"PR_MLE_corrected","block":block,"term":term,
                            "estimate":pr['theta'][j],
                            "curvature_se_original":pr['se'][j],
                            "sandwich_se":se[j]})
        print(mode, 'n=', len(y), 'PR sandwich SE:', np.round(se, 5))
    outdir = ROOT/'results'/'pr_sandwich'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(outdir/'PR_MLE_weighted_sandwich_SE.csv', index=False)
    print('Saved:', outdir/'PR_MLE_weighted_sandwich_SE.csv')

if __name__ == '__main__':
    run()
