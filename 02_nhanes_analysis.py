# ============================================================
# PR-ZIB NHANES 2017--March 2020 pre-pandemic analysis
#
# Operational definitions
#   Y = reported physician-diagnosed diabetes.
#   Among Y=0, an observed laboratory value reveals the operational gate S
#   relative to the prespecified HbA1c or FPG threshold.
#
# Correct partial-revelation likelihood patterns
#   Y=1                  -> pi * mu
#   Y=0, R=1, W=1       -> pi * (1-mu)
#   Y=0, R=1, W=0       -> 1-pi
#   Y=0, R=0             -> 1-pi*mu
#
# Primary model used in the manuscript
#   pi: age, BMI, sex
#   mu: insurance, age, BMI, sex
# Usual source of care is retained in the nuisance history but excluded from
# the primary unpenalized mu-block because the full model is separation-prone.
# A full-covariate sensitivity fit is also produced.
#
# Reported SEs treat NHANES analysis weights as fixed; they are not full
# survey-design-based standard errors.
# ============================================================

# ---- keep BLAS single-threaded before importing numpy/scipy ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root, least_squares
from scipy.special import expit, logit
from statsmodels.tools.numdiff import approx_hess
from sklearn.model_selection import StratifiedKFold, KFold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=6)

# ============================================================
# SETTINGS
# ============================================================
ADULT_AGE = 20
SEED = 20260922
N_SPLITS = 5
N_JOBS = min(int(os.environ.get("PRZIB_N_JOBS", "7")), os.cpu_count() or 1)
MAXITER = 4000
BOUND = 20.0
ROOT_BOUND = 30.0
PROB_EPS = 1e-10
NUISANCE_RIDGE = 1e-4

# Run both the primary scientific allocation and a full-covariate sensitivity.
RUN_FULL_COVARIATE_SENSITIVITY = True

# Optional nonparametric bootstrap; not used for the reported main NHANES tables.
RUN_BOOTSTRAP = False
B_BOOT = 500
BOOT_MIN_SUCCESS = 0.80

OUTDIR = Path(os.environ.get("PRZIB_NHANES_OUTDIR", "results/nhanes"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Primary scientific allocation used in the manuscript.
PRIMARY_X_COLS = ["const", "AGE_YEARS_Z", "BMI_Z", "FEMALE"]
PRIMARY_Z_COLS = ["const", "INSURED", "AGE_YEARS_Z", "BMI_Z", "FEMALE"]
FULL_COLS = ["const", "INSURED", "USUALCARE", "AGE_YEARS_Z", "BMI_Z", "FEMALE"]
# Nuisance history: union of scientifically relevant observed covariates.
H_COLS = ["INSURED", "USUALCARE", "AGE_YEARS_Z", "BMI_Z", "FEMALE"]

# ============================================================
# BASIC HELPERS
# ============================================================
def clip01(x, eps=PROB_EPS):
    return np.clip(np.asarray(x, float), eps, 1.0 - eps)


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if m.sum() == 0 or w[m].sum() <= 0:
        return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def normalize_weights(w):
    w = np.asarray(w, float)
    return w / np.mean(w)


def stable_pinv(A, rcond=1e-10):
    A = np.asarray(A, float)
    return np.linalg.pinv(A, rcond=rcond)


def condition_number_sym(H):
    H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)
    pos = ev[ev > 1e-10]
    if len(pos) == 0:
        return np.inf
    return float(pos.max() / pos.min())


def read_xpt(url):
    return pd.read_sas(url, format="xport")

# ============================================================
# WEIGHTED LOGISTIC REGRESSION FOR STARTS / NUISANCE MODELS
# ============================================================
def fit_weighted_logit(X, y, w, start=None, ridge=0.0, bounds=True):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = normalize_weights(w)
    p = X.shape[1]
    if start is None:
        start = np.zeros(p)
        py = np.clip(weighted_mean(y, w), 1e-5, 1-1e-5)
        # If first column is an intercept, this is a useful starting value.
        if np.allclose(X[:, 0], 1.0):
            start[0] = logit(py)

    def fg(th):
        pr = clip01(expit(X @ th))
        nll = -np.sum(w * (y*np.log(pr) + (1-y)*np.log(1-pr)))
        grad = X.T @ (w * (pr-y))
        if ridge > 0:
            pen = th.copy()
            if np.allclose(X[:, 0], 1.0):
                pen[0] = 0.0
            nll += 0.5 * ridge * np.sum(pen**2)
            grad += ridge * pen
        return nll, grad

    kw = {}
    if bounds:
        kw["bounds"] = [(-BOUND, BOUND)] * p
    res = minimize(lambda th: fg(th)[0], start, jac=lambda th: fg(th)[1],
                   method="L-BFGS-B", options={"maxiter": MAXITER}, **kw)
    return res


def fit_weighted_logit_predict(Xtr, ytr, wtr, Xte, ridge=NUISANCE_RIDGE):
    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float)
    wtr = np.asarray(wtr, float)
    Xte = np.asarray(Xte, float)
    if len(ytr) == 0:
        return np.full(len(Xte), 0.5)
    vals = np.unique(ytr)
    if len(vals) < 2:
        p = np.clip(weighted_mean(ytr, wtr), 1e-6, 1-1e-6)
        return np.full(len(Xte), p)
    res = fit_weighted_logit(Xtr, ytr, wtr, ridge=ridge)
    th = res.x if np.all(np.isfinite(res.x)) else np.zeros(Xtr.shape[1])
    return np.clip(expit(Xte @ th), 1e-6, 1-1e-6)

# ============================================================
# OBSERVED-DATA ZIB COMPARATOR
# ============================================================
def obs_nll_grad(par, X, Z, y, w):
    pa = X.shape[1]
    a, b = par[:pa], par[pa:]
    pi = clip01(expit(X @ a))
    mu = clip01(expit(Z @ b))
    den = clip01(1.0 - pi*mu)

    ll = y*(np.log(pi)+np.log(mu)) + (1-y)*np.log(den)
    ga = y*(1-pi) - (1-y)*(mu*pi*(1-pi))/den
    gb = y*(1-mu) - (1-y)*(pi*mu*(1-mu))/den
    grad = -np.r_[X.T @ (w*ga), Z.T @ (w*gb)]
    return -np.sum(w*ll), grad

# ============================================================
# CORRECTED PARTIAL-REVELATION LIKELIHOOD
# ============================================================
def pr_nll_grad(par, X, Z, y, R, W, w):
    """Correct four-pattern PR likelihood and gradient."""
    pa = X.shape[1]
    a, b = par[:pa], par[pa:]
    pi = clip01(expit(X @ a))
    mu = clip01(expit(Z @ b))
    den = clip01(1.0 - pi*mu)

    y1 = (y == 1)
    zr1 = (y == 0) & (R == 1) & (W == 1)
    zr0 = (y == 0) & (R == 1) & (W == 0)
    zu = (y == 0) & (R == 0)

    # Exhaustive and mutually exclusive by construction.
    if not np.all(y1 | zr1 | zr0 | zu):
        raise RuntimeError("Observed-data patterns are not exhaustive.")

    ll = np.zeros(len(y), float)
    ga = np.zeros(len(y), float)
    gb = np.zeros(len(y), float)

    # Y=1: S=1 and event recognized/reported.
    ll[y1] = np.log(pi[y1]) + np.log(mu[y1])
    ga[y1] = 1.0 - pi[y1]
    gb[y1] = 1.0 - mu[y1]

    # Y=0, revealed S=1: operationally positive but not recognized/reported.
    ll[zr1] = np.log(pi[zr1]) + np.log(1.0-mu[zr1])
    ga[zr1] = 1.0 - pi[zr1]
    gb[zr1] = -mu[zr1]

    # Y=0, revealed S=0: contribution from an operationally negative revealed zero.
    ll[zr0] = np.log(1.0-pi[zr0])
    ga[zr0] = -pi[zr0]
    gb[zr0] = 0.0

    # Y=0, unresolved S.
    ll[zu] = np.log(den[zu])
    ga[zu] = -(mu[zu]*pi[zu]*(1.0-pi[zu])) / den[zu]
    gb[zu] = -(pi[zu]*mu[zu]*(1.0-mu[zu])) / den[zu]

    grad = -np.r_[X.T @ (w*ga), Z.T @ (w*gb)]
    return -np.sum(w*ll), grad


def make_pr_starts(X, Z, y, R, W, w, seed=SEED):
    rng = np.random.default_rng(seed)
    px, pz = X.shape[1], Z.shape[1]

    py = np.clip(weighted_mean(y, w), 1e-5, 1-1e-5)
    zr = (y == 0) & (R == 1)
    q = weighted_mean(W[zr], w[zr]) if zr.any() else 0.05
    q = np.clip(q, 1e-4, 1-1e-4)
    # Descriptive operational-prevalence anchor; only for optimization starts.
    ps = np.clip(py + (1-py)*q, py + 1e-4, 1-1e-4)
    murec = np.clip(py/ps, 1e-4, 1-1e-4)

    anchor = np.r_[np.r_[logit(ps), np.zeros(px-1)],
                   np.r_[logit(murec), np.zeros(pz-1)]]

    # Additional starts based on ordinary weighted logistic regressions.
    a_y = fit_weighted_logit(X, y, w).x
    b_y = fit_weighted_logit(Z, y, w).x
    starts = [anchor, np.r_[a_y, b_y], np.zeros(px+pz)]
    for _ in range(5):
        starts.append(anchor + rng.normal(scale=0.25, size=px+pz))
    return starts


def multi_start_fit(fg, starts):
    best = None
    for s in starts:
        try:
            res = minimize(lambda th: fg(th)[0], s, jac=lambda th: fg(th)[1],
                           method="L-BFGS-B",
                           bounds=[(-BOUND, BOUND)]*len(s),
                           options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-8})
            if np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
                if best is None or res.fun < best.fun:
                    best = res
        except Exception:
            pass
    if best is None:
        raise RuntimeError("All optimization starts failed.")
    return best


def _finish_likelihood_fit(res, fg, X, Z):
    H = approx_hess(res.x, lambda th: fg(th)[0])
    H = 0.5*(H+H.T)
    cov = stable_pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    px = X.shape[1]
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "nll": float(res.fun),
        "theta": np.asarray(res.x),
        "alpha": np.asarray(res.x[:px]),
        "beta": np.asarray(res.x[px:]),
        "se": se,
        "se_alpha": se[:px],
        "se_beta": se[px:],
        "H": H,
        "cov": cov,
        "cond_hess": condition_number_sym(H),
        "max_abs_coef": float(np.max(np.abs(res.x)))
    }


def fit_obs_zib(X, Z, y, w, seed=SEED):
    starts = make_pr_starts(X, Z, y, np.zeros_like(y), np.zeros_like(y), w, seed=seed)
    fg = lambda th: obs_nll_grad(th, X, Z, y, w)
    res = multi_start_fit(fg, starts)
    return _finish_likelihood_fit(res, fg, X, Z)


def fit_pr(X, Z, y, R, W, w, seed=SEED):
    starts = make_pr_starts(X, Z, y, R, W, w, seed=seed)
    fg = lambda th: pr_nll_grad(th, X, Z, y, R, W, w)
    res = multi_start_fit(fg, starts)
    return _finish_likelihood_fit(res, fg, X, Z)

# ============================================================
# CROSS-FITTED NUISANCE FUNCTIONS
# ============================================================
def _make_zero_folds(y, R, W, n_splits=N_SPLITS, seed=SEED):
    zidx = np.where(y == 0)[0]
    strata = (R[zidx] + W[zidx]).astype(int)  # 0=unresolved, 1=revealed S0, 2=revealed S1
    counts = pd.Series(strata).value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(zidx[tr], zidx[te]) for tr, te in sp.split(zidx, strata)]
    sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(zidx[tr], zidx[te]) for tr, te in sp.split(zidx)]


def crossfit_empirical_nuisance(H, y, R, W, w, n_splits=N_SPLITS, seed=SEED):
    n = len(y)
    ehat = np.ones(n)
    mhat = np.ones(n)
    folds = _make_zero_folds(y, R, W, n_splits=n_splits, seed=seed)

    for k, (tr, te) in enumerate(folds):
        # Include intercept in nuisance regressions.
        Htr = np.column_stack([np.ones(len(tr)), H[tr]])
        Hte = np.column_stack([np.ones(len(te)), H[te]])
        ehat[te] = fit_weighted_logit_predict(Htr, R[tr], w[tr], Hte,
                                              ridge=NUISANCE_RIDGE)

        rev = tr[R[tr] == 1]
        if len(rev) == 0:
            mhat[te] = 0.5
        else:
            Hr = np.column_stack([np.ones(len(rev)), H[rev]])
            mhat[te] = fit_weighted_logit_predict(Hr, W[rev], w[rev], Hte,
                                                  ridge=NUISANCE_RIDGE)

    z = (y == 0)
    ehat[z] = np.clip(ehat[z], 1e-6, 1-1e-6)
    mhat[z] = np.clip(mhat[z], 1e-6, 1-1e-6)
    return ehat, mhat, folds


def crossfit_structural_m(X, Z, y, R, W, w, folds, seed=SEED):
    """Out-of-fold structural m_theta = pi(1-mu)/(1-pi*mu) from corrected PR fits."""
    n = len(y)
    mhat = np.ones(n)
    y1_idx = np.where(y == 1)[0]
    for k, (ztr, zte) in enumerate(folds):
        # All Y=1 are observed S=1 and can remain in every nuisance training fold;
        # only zero-layer observations are cross-fitted.
        tr = np.r_[y1_idx, ztr]
        fit = fit_pr(X[tr], Z[tr], y[tr], R[tr], W[tr], w[tr], seed=seed+1000+k)
        pi = clip01(expit(X[zte] @ fit["alpha"]))
        mu = clip01(expit(Z[zte] @ fit["beta"]))
        den = clip01(1.0 - pi*mu)
        mhat[zte] = np.clip(pi*(1.0-mu)/den, 1e-6, 1-1e-6)
    return mhat


def build_aipw_gate(y, R, W, ehat, mhat):
    e = np.maximum(np.asarray(ehat, float), 1e-6)
    Se = y + (1-y)*(mhat + R*(W-mhat)/e)
    return Se

# ============================================================
# DR ESTIMATING EQUATIONS
# ============================================================
def solve_score(score_fun, jac_fun, starts):
    best = None
    for s in starts:
        try:
            rr = root(score_fun, s, jac=jac_fun, method="hybr")
            val = float(np.linalg.norm(score_fun(rr.x)))
            if np.all(np.isfinite(rr.x)) and np.max(np.abs(rr.x)) <= ROOT_BOUND:
                if best is None or val < best[0]:
                    best = (val, rr.x)
        except Exception:
            pass
    if best is not None and best[0] < 1e-6:
        return best[1], best[0]

    for s in starts:
        try:
            ls = least_squares(score_fun, s, jac=jac_fun,
                               bounds=(-ROOT_BOUND, ROOT_BOUND),
                               xtol=1e-11, ftol=1e-11, gtol=1e-11,
                               max_nfev=2000)
            val = float(np.linalg.norm(score_fun(ls.x)))
            if np.all(np.isfinite(ls.x)) and (best is None or val < best[0]):
                best = (val, ls.x)
        except Exception:
            pass
    if best is None or best[0] > 5e-5:
        raise RuntimeError("DR estimating equation has no stable numerical root.")
    return best[1], best[0]


def fit_dr(X, Z, y, R, W, w, ehat, mhat, theta_init=None):
    Se = build_aipw_gate(y, R, W, ehat, mhat)
    sw = np.sum(w)
    px, pz = X.shape[1], Z.shape[1]
    if theta_init is None:
        a0, b0 = np.zeros(px), np.zeros(pz)
    else:
        a0 = np.asarray(theta_init[:px])
        b0 = np.asarray(theta_init[px:px+pz])

    def score_a(a):
        pi = expit(X @ a)
        return X.T @ (w*(Se-pi)) / sw
    def jac_a(a):
        pi = expit(X @ a)
        return -(X.T @ ((w*pi*(1-pi))[:, None]*X)) / sw

    def score_b(b):
        mu = expit(Z @ b)
        return Z.T @ (w*(y-Se*mu)) / sw
    def jac_b(b):
        mu = expit(Z @ b)
        return -(Z.T @ ((w*Se*mu*(1-mu))[:, None]*Z)) / sw

    ahat, norm_a = solve_score(score_a, jac_a, [a0, np.zeros(px)])
    bhat, norm_b = solve_score(score_b, jac_b, [b0, np.zeros(pz)])
    theta = np.r_[ahat, bhat]

    pi = clip01(expit(X @ ahat))
    mu = clip01(expit(Z @ bhat))
    Aaa = X.T @ ((w*pi*(1-pi))[:, None]*X)
    Abb = Z.T @ ((w*Se*mu*(1-mu))[:, None]*Z)
    A = np.block([[Aaa, np.zeros((px, pz))],
                  [np.zeros((pz, px)), Abb]])
    psi = np.concatenate([(w*(Se-pi))[:, None]*X,
                          (w*(y-Se*mu))[:, None]*Z], axis=1)
    B = psi.T @ psi
    Ainv = stable_pinv(A)
    cov = Ainv @ B @ Ainv.T
    cov = 0.5*(cov+cov.T)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return {
        "success": True,
        "theta": theta,
        "alpha": ahat,
        "beta": bhat,
        "se": se,
        "se_alpha": se[:px],
        "se_beta": se[px:],
        "cov": cov,
        "A": A,
        "score_norm_alpha": norm_a,
        "score_norm_beta": norm_b,
        "cond_A": float(np.linalg.cond(A)),
        "Se": Se,
        "Se_min": float(np.min(Se)),
        "Se_q01": float(np.quantile(Se, .01)),
        "Se_q99": float(np.quantile(Se, .99)),
        "Se_max": float(np.max(Se)),
        "max_abs_coef": float(np.max(np.abs(theta)))
    }

# ============================================================
# DESCRIPTIVE / SANITY CHECKS
# ============================================================
def model_sanity(method, alpha, beta, X, Z, y, R, W, w_raw, Se=None):
    pi = clip01(expit(X @ alpha))
    mu = clip01(expit(Z @ beta))
    qy = pi*mu
    mtheta = pi*(1-mu)/clip01(1-pi*mu)

    z = (y == 0)
    zr = z & (R == 1)
    py_obs = weighted_mean(y, w_raw)
    rho = weighted_mean(R[z], w_raw[z])
    q_obs = weighted_mean(W[zr], w_raw[zr]) if zr.any() else np.nan
    rough_ps = py_obs + (1-py_obs)*q_obs if np.isfinite(q_obs) else np.nan

    out = {
        "method": method,
        "observed_weighted_PY1": py_obs,
        "observed_weighted_rho": rho,
        "observed_weighted_q": q_obs,
        "rough_PS_anchor_PY1_plus_PY0q": rough_ps,
        "model_mean_pi": weighted_mean(pi, w_raw),
        "model_mean_pi_mu": weighted_mean(qy, w_raw),
        "model_recognition_among_S1": float(np.sum(w_raw*pi*mu)/np.sum(w_raw*pi)),
        "model_mtheta_among_revealed_zeros": weighted_mean(mtheta[zr], w_raw[zr]) if zr.any() else np.nan,
        "model_mtheta_among_all_zeros": weighted_mean(mtheta[z], w_raw[z])
    }
    if Se is not None:
        out["AIPW_mean_S"] = weighted_mean(Se, w_raw)
        out["AIPW_Se_min"] = float(np.min(Se))
        out["AIPW_Se_q01"] = float(np.quantile(Se, .01))
        out["AIPW_Se_q99"] = float(np.quantile(Se, .99))
        out["AIPW_Se_max"] = float(np.max(Se))
    return out


def reveal_diagnostics(base, DVAR, WVAR):
    y = base["Y"].to_numpy(float)
    R = base["R"].to_numpy(float)
    W = base["W"].to_numpy(float)
    w = base[WVAR].to_numpy(float)
    z = (y == 0)
    zr = z & (R == 1)
    zr1 = zr & (W == 1)
    zr0 = zr & (W == 0)
    zu = z & (R == 0)

    if int(z.sum()) != int(zr1.sum()+zr0.sum()+zu.sum()):
        raise RuntimeError("Zero-layer pattern counts do not add up.")
    if np.any(W > R):
        raise RuntimeError("W=1 found with R=0.")

    return {
        "n": int(len(base)),
        "n_y1": int((y==1).sum()),
        "n_y0": int(z.sum()),
        "n_y0_R1_W1": int(zr1.sum()),
        "n_y0_R1_W0": int(zr0.sum()),
        "n_y0_R0": int(zu.sum()),
        "sample_rho": float(R[z].mean()),
        "weighted_rho": weighted_mean(R[z], w[z]),
        "sample_q": float(W[zr].mean()) if zr.any() else np.nan,
        "weighted_q": weighted_mean(W[zr], w[zr]) if zr.any() else np.nan,
        "weighted_PY1": weighted_mean(y, w),
        "weighted_proxy_positive_among_observed": weighted_mean(
            base.loc[base[DVAR].notna(), DVAR].to_numpy(float),
            base.loc[base[DVAR].notna(), WVAR].to_numpy(float)
        ) if base[DVAR].notna().any() else np.nan
    }

# ============================================================
# TABLE HELPERS
# ============================================================
def coefficient_table(fits, xcols, zcols):
    rows = []
    for method, fit in fits.items():
        for j, term in enumerate(xcols):
            rows.append({"method": method, "block": "alpha", "term": term,
                         "estimate": fit["alpha"][j], "se": fit["se_alpha"][j]})
        for j, term in enumerate(zcols):
            rows.append({"method": method, "block": "beta", "term": term,
                         "estimate": fit["beta"][j], "se": fit["se_beta"][j]})
    return pd.DataFrame(rows)

# ============================================================
# OPTIONAL NONPARAMETRIC BOOTSTRAP FOR NHANES POINT ANALYSIS
# ============================================================
def _bootstrap_one(base_arrays, design, seed):
    rng = np.random.default_rng(seed)
    n = len(base_arrays["y"])
    ii = rng.integers(0, n, size=n)
    X = base_arrays["X"][ii]
    Z = base_arrays["Z"][ii]
    H = base_arrays["H"][ii]
    y = base_arrays["y"][ii]
    R = base_arrays["R"][ii]
    W = base_arrays["W"][ii]
    w = normalize_weights(base_arrays["w"][ii])
    try:
        pr = fit_pr(X, Z, y, R, W, w, seed=seed+11)
        ehat, mhat, _ = crossfit_empirical_nuisance(H, y, R, W, w, seed=seed+23)
        dr = fit_dr(X, Z, y, R, W, w, ehat, mhat, theta_init=pr["theta"])
        return {"ok": True, "theta_pr": pr["theta"], "theta_dr": dr["theta"]}
    except Exception:
        return {"ok": False, "theta_pr": None, "theta_dr": None}


def run_bootstrap(base_arrays, design_name, B=B_BOOT, seed=SEED):
    seeds = [seed + 100000 + b for b in range(B)]
    ans = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_bootstrap_one)(base_arrays, design_name, s) for s in seeds
    )
    ok = np.array([a["ok"] for a in ans], bool)
    out = {"B": B, "success_fraction": float(ok.mean())}
    if ok.sum() >= max(20, int(BOOT_MIN_SUCCESS*B)):
        pr = np.vstack([a["theta_pr"] for a in ans if a["ok"]])
        dr = np.vstack([a["theta_dr"] for a in ans if a["ok"]])
        out["pr_draws"] = pr
        out["dr_draws"] = dr
    else:
        out["pr_draws"] = None
        out["dr_draws"] = None
    return out

# ============================================================
# NHANES DATA
# ============================================================
def load_nhanes():
    BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    FILES = {
        "DEMO": "P_DEMO.XPT",
        "DIQ":  "P_DIQ.XPT",
        "HIQ":  "P_HIQ.XPT",
        "HUQ":  "P_HUQ.XPT",
        "BMX":  "P_BMX.XPT",
        "GHB":  "P_GHB.XPT",
        "GLU":  "P_GLU.XPT",
    }
    print("Downloading NHANES 2017-March 2020 pre-pandemic files ...")
    dfs = {k: read_xpt(BASE + v) for k, v in FILES.items()}
    df = dfs["DEMO"].copy()
    for k in ["DIQ", "HIQ", "HUQ", "BMX", "GHB", "GLU"]:
        df = df.merge(dfs[k], on="SEQN", how="left")

    dat = pd.DataFrame()
    dat["SEQN"] = df["SEQN"]
    dat["AGE_YEARS"] = df["RIDAGEYR"]
    dat["FEMALE"] = np.where(df["RIAGENDR"] == 2, 1.0,
                      np.where(df["RIAGENDR"] == 1, 0.0, np.nan))
    dat["BMI"] = df["BMXBMI"]
    dat["Y"] = np.where(df["DIQ010"] == 1, 1.0,
                 np.where(df["DIQ010"] == 2, 0.0, np.nan))
    dat["INSURED"] = np.where(df["HIQ011"] == 1, 1.0,
                       np.where(df["HIQ011"] == 2, 0.0, np.nan))
    huq = df["HUQ030"]
    dat["USUALCARE"] = np.select([huq.isin([1,2]), huq == 3], [1.0, 0.0], default=np.nan)
    dat["D_A1C"] = np.where(df["LBXGH"].notna(), (df["LBXGH"] >= 6.5).astype(float), np.nan)
    dat["D_FPG"] = np.where(df["LBXGLU"].notna(), (df["LBXGLU"] >= 126.0).astype(float), np.nan)
    dat["WTMECPRP"] = df["WTMECPRP"]
    dat["WTSAFPRP"] = df["WTSAFPRP"]
    return dat[dat["AGE_YEARS"] >= ADULT_AGE].copy()


def make_analysis_base(dat, mode):
    if mode == "A1C":
        DVAR, WVAR = "D_A1C", "WTMECPRP"
    elif mode == "FPG":
        DVAR, WVAR = "D_FPG", "WTSAFPRP"
    else:
        raise ValueError(mode)

    needed = ["Y", "INSURED", "USUALCARE", "AGE_YEARS", "BMI", "FEMALE", WVAR]
    base = dat.dropna(subset=needed).copy()
    base = base[base[WVAR] > 0].copy()

    base["AGE_YEARS_Z"] = (base["AGE_YEARS"] - base["AGE_YEARS"].mean()) / base["AGE_YEARS"].std(ddof=0)
    base["BMI_Z"] = (base["BMI"] - base["BMI"].mean()) / base["BMI"].std(ddof=0)
    base["const"] = 1.0

    # Operational partial verification among Y=0.
    base["R"] = ((base["Y"] == 0) & base[DVAR].notna()).astype(float)
    # W=R*S. Among verified Y=0, S is the operational biomarker indicator D.
    base["W"] = ((base["Y"] == 0) & (base["R"] == 1) & (base[DVAR] == 1)).astype(float)
    return base, DVAR, WVAR

# ============================================================
# ONE ANALYSIS
# ============================================================
def run_one_design(base, DVAR, WVAR, mode, design_name, xcols, zcols, seed=SEED):
    print("\n" + "="*110)
    print(f"{mode} | {design_name}")
    print("="*110)

    X = base[xcols].to_numpy(float)
    Z = base[zcols].to_numpy(float)
    H = base[H_COLS].to_numpy(float)
    y = base["Y"].to_numpy(float)
    R = base["R"].to_numpy(float)
    W = base["W"].to_numpy(float)
    w_raw = base[WVAR].to_numpy(float)
    w = normalize_weights(w_raw)

    desc = reveal_diagnostics(base, DVAR, WVAR)
    desc.update({"mode": mode, "design": design_name, "weight_variable": WVAR})
    print("\nObserved-data pattern check")
    print(pd.Series(desc).to_string())

    # Critical sanity check: there must be many revealed negatives.
    if desc["n_y0_R1_W0"] == 0:
        raise RuntimeError("No revealed negatives found; check W/D coding.")

    # Corrected PR and ordinary ZIB comparator.
    zib = fit_obs_zib(X, Z, y, w, seed=seed+1)
    pr = fit_pr(X, Z, y, R, W, w, seed=seed+2)

    # Empirical cross-fitted nuisance models.
    ehat, mhat, folds = crossfit_empirical_nuisance(H, y, R, W, w, seed=seed+3)
    dr = fit_dr(X, Z, y, R, W, w, ehat, mhat, theta_init=pr["theta"])

    # Structural m_theta sensitivity: useful because under M1-M3 it is exactly m0.
    mstruct = crossfit_structural_m(X, Z, y, R, W, w, folds, seed=seed+4)
    dr_struct = fit_dr(X, Z, y, R, W, w, ehat, mstruct, theta_init=pr["theta"])

    fits = {
        "OBS_ZIB": zib,
        "PR_MLE_corrected": pr,
        "DR_CF_empirical_m": dr,
        "DR_CF_structural_m": dr_struct,
    }
    coef = coefficient_table(fits, xcols, zcols)

    # Sanity checks for fitted latent decomposition.
    sanity = []
    sanity.append(model_sanity("PR_MLE_corrected", pr["alpha"], pr["beta"], X, Z, y, R, W, w_raw))
    sanity.append(model_sanity("DR_CF_empirical_m", dr["alpha"], dr["beta"], X, Z, y, R, W, w_raw, Se=dr["Se"]))
    sanity.append(model_sanity("DR_CF_structural_m", dr_struct["alpha"], dr_struct["beta"], X, Z, y, R, W, w_raw, Se=dr_struct["Se"]))
    sanity = pd.DataFrame(sanity)

    z = (y == 0)
    zr = z & (R == 1)
    nuisance = pd.DataFrame([{
        "mode": mode,
        "design": design_name,
        "ehat_mean_zero": weighted_mean(ehat[z], w_raw[z]),
        "ehat_min_zero": float(np.min(ehat[z])),
        "ehat_q01_zero": float(np.quantile(ehat[z], .01)),
        "ehat_q99_zero": float(np.quantile(ehat[z], .99)),
        "mhat_emp_mean_zero": weighted_mean(mhat[z], w_raw[z]),
        "mhat_emp_mean_revealed_zero": weighted_mean(mhat[zr], w_raw[zr]),
        "mhat_struct_mean_zero": weighted_mean(mstruct[z], w_raw[z]),
        "mhat_struct_mean_revealed_zero": weighted_mean(mstruct[zr], w_raw[zr]),
        "mean_abs_emp_minus_struct_m_zero": weighted_mean(np.abs(mhat[z]-mstruct[z]), w_raw[z]),
        "Se_emp_min": dr["Se_min"],
        "Se_emp_q01": dr["Se_q01"],
        "Se_emp_q99": dr["Se_q99"],
        "Se_emp_max": dr["Se_max"],
    }])

    diag = pd.DataFrame([
        {"method":"OBS_ZIB", "success":zib["success"], "objective":zib["nll"],
         "condition":zib["cond_hess"], "max_abs_coef":zib["max_abs_coef"]},
        {"method":"PR_MLE_corrected", "success":pr["success"], "objective":pr["nll"],
         "condition":pr["cond_hess"], "max_abs_coef":pr["max_abs_coef"]},
        {"method":"DR_CF_empirical_m", "success":dr["success"], "objective":np.nan,
         "condition":dr["cond_A"], "max_abs_coef":dr["max_abs_coef"],
         "score_norm_alpha":dr["score_norm_alpha"], "score_norm_beta":dr["score_norm_beta"]},
        {"method":"DR_CF_structural_m", "success":dr_struct["success"], "objective":np.nan,
         "condition":dr_struct["cond_A"], "max_abs_coef":dr_struct["max_abs_coef"],
         "score_norm_alpha":dr_struct["score_norm_alpha"], "score_norm_beta":dr_struct["score_norm_beta"]},
    ])

    print("\nSanity checks")
    print(sanity.round(5).to_string(index=False))
    print("\nCoefficient estimates")
    print(coef.round(5).to_string(index=False))
    print("\nNumerical diagnostics")
    print(diag.round(5).to_string(index=False))
    print("\nNuisance diagnostics")
    print(nuisance.round(5).to_string(index=False))

    prefix = f"{mode.lower()}_{design_name}"
    pd.DataFrame([desc]).to_csv(OUTDIR/f"descriptive_{prefix}.csv", index=False)
    coef.to_csv(OUTDIR/f"coefficients_{prefix}.csv", index=False)
    sanity.to_csv(OUTDIR/f"sanity_{prefix}.csv", index=False)
    nuisance.to_csv(OUTDIR/f"nuisance_{prefix}.csv", index=False)
    diag.to_csv(OUTDIR/f"diagnostics_{prefix}.csv", index=False)

    boot_summary = None
    if RUN_BOOTSTRAP and design_name == "primary_no_usualcare":
        print(f"\nRunning B={B_BOOT} nonparametric bootstrap with {N_JOBS} workers ...")
        arr = {"X":X, "Z":Z, "H":H, "y":y, "R":R, "W":W, "w":w_raw}
        boot = run_bootstrap(arr, design_name, B=B_BOOT, seed=seed+5000)
        boot_summary = {"mode":mode, "design":design_name,
                        "B":B_BOOT, "success_fraction":boot["success_fraction"]}
        if boot["pr_draws"] is not None:
            rows=[]
            names = [f"alpha:{c}" for c in xcols] + [f"beta:{c}" for c in zcols]
            for method, draws, theta0 in [
                ("PR_MLE_corrected", boot["pr_draws"], pr["theta"]),
                ("DR_CF_empirical_m", boot["dr_draws"], dr["theta"]),
            ]:
                for j, nm in enumerate(names):
                    lo, hi = np.quantile(draws[:,j], [.025,.975])
                    rows.append({"method":method,"parameter":nm,
                                 "estimate":theta0[j],
                                 "boot_se":np.std(draws[:,j], ddof=1),
                                 "boot_p025":lo,"boot_p975":hi})
            pd.DataFrame(rows).to_csv(OUTDIR/f"bootstrap_{prefix}.csv", index=False)
        pd.DataFrame([boot_summary]).to_csv(OUTDIR/f"bootstrap_summary_{prefix}.csv", index=False)

    return {"desc":desc, "coef":coef, "sanity":sanity, "nuisance":nuisance,
            "diag":diag, "boot_summary":boot_summary}

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*110)
    print("PR-ZIB NHANES CORRECTED REANALYSIS")
    print("="*110)
    print(f"5-fold DR cross-fitting; optional bootstrap cores={N_JOBS}; RUN_BOOTSTRAP={RUN_BOOTSTRAP}")
    print("Primary allocation: pi <- age/BMI/sex; mu <- insurance + age/BMI/sex (USUALCARE excluded from primary mu-block)")
    print("The PR likelihood includes all four observed-data patterns, including revealed negatives Y=0,R=1,W=0.")

    dat = load_nhanes()
    all_sanity=[]
    all_coef=[]
    all_desc=[]
    all_diag=[]

    for imode, mode in enumerate(["A1C", "FPG"]):
        base, DVAR, WVAR = make_analysis_base(dat, mode)

        designs = [("primary_no_usualcare", PRIMARY_X_COLS, PRIMARY_Z_COLS)]
        if RUN_FULL_COVARIATE_SENSITIVITY:
            designs.append(("full_covariate", FULL_COLS, FULL_COLS))

        for j, (dname, xcols, zcols) in enumerate(designs):
            out = run_one_design(base, DVAR, WVAR, mode, dname, xcols, zcols,
                                 seed=SEED + 10000*imode + 100*j)
            dd = pd.DataFrame([out["desc"]]); all_desc.append(dd)
            cc = out["coef"].copy(); cc.insert(0,"mode",mode); cc.insert(1,"design",dname); all_coef.append(cc)
            ss = out["sanity"].copy(); ss.insert(0,"mode",mode); ss.insert(1,"design",dname); all_sanity.append(ss)
            gg = out["diag"].copy(); gg.insert(0,"mode",mode); gg.insert(1,"design",dname); all_diag.append(gg)

    pd.concat(all_desc, ignore_index=True).to_csv(OUTDIR/"all_descriptive.csv", index=False)
    pd.concat(all_coef, ignore_index=True).to_csv(OUTDIR/"all_coefficients.csv", index=False)
    pd.concat(all_sanity, ignore_index=True).to_csv(OUTDIR/"all_sanity_checks.csv", index=False)
    pd.concat(all_diag, ignore_index=True).to_csv(OUTDIR/"all_diagnostics.csv", index=False)

    config = {
        "seed":SEED, "n_splits":N_SPLITS, "n_jobs":N_JOBS,
        "primary_X_pi":PRIMARY_X_COLS, "primary_Z_mu":PRIMARY_Z_COLS,
        "nuisance_H":H_COLS, "run_full_covariate_sensitivity":RUN_FULL_COVARIATE_SENSITIVITY,
        "run_bootstrap":RUN_BOOTSTRAP, "B_boot":B_BOOT,
        "note":"Primary mu-block excludes USUALCARE because the unpenalized recognition model showed quasi-complete separation when USUALCARE was included. DR sandwich treats cross-fitted nuisance predictions as plug-in; optional bootstrap refits nuisances."
    }
    with open(OUTDIR/"config.json","w") as f:
        json.dump(config,f,indent=2)

    zip_path = shutil.make_archive(str(OUTDIR), "zip", root_dir=str(OUTDIR))
    print("\n" + "="*110)
    print("DONE")
    print("Outputs:", OUTDIR)
    print("ZIP:", zip_path)
    print("="*110)


if __name__ == "__main__":
    main()