# ============================================================
# PR-ZIB targeted bootstrap for the manuscript
#
# This file contains the same model-fitting and data-generation functions used
# in the main simulation, followed by the targeted nonparametric bootstrap.
# Each bootstrap sample refits PR_MLE; for DR_CF it also re-estimates the
# cross-fitted nuisance functions and resolves the estimating equations.
# ============================================================

import os, json, math, warnings, shutil
warnings.filterwarnings("ignore")

# Avoid thread oversubscription when joblib uses 7 worker processes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.optimize import brentq, minimize, root, least_squares
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODE = "main"
N_JOBS = min(int(os.environ.get("PRZIB_N_JOBS", "7")), os.cpu_count() or 1)
BASE_SEED = 20260920
N_SPLITS = 5
OUTDIR = os.environ.get("PRZIB_BOOT_BASE_OUTDIR", "results")
os.makedirs(OUTDIR, exist_ok=True)

RUN_EXPERIMENT_A = True
# Experiment B was unchanged by the positivity correction.  It is False by default
# so the already completed NHANES-like run need not be repeated.  Set True for a
# fully self-contained rerun.
RUN_EXPERIMENT_B = False
# Optional practical-positivity stress test; not part of the reported bootstrap table.
RUN_STRESS_TEST = False
# Optional truncation sensitivity; not part of the reported bootstrap table.
RUN_TRIM_SENSITIVITY = False

# The targeted B=1000 bootstrap should be run after inspecting these results,
# so that unstable / transition / stable cells are chosen deliberately.
RUN_BOOTSTRAP_PHASE = False

if MODE == "quick":
    MC_REPS_A = 12
    MC_REPS_B = 12
    N_GRID_A = [200, 500]
    RHO_GRID_A = [0.01, 0.05, 0.10]
    N_GRID_B = [4000]
    Q_GRID_B = [0.02, 0.04]
    RHO_GRID_B = [0.90, 0.95]
else:
    MC_REPS_A = 500
    MC_REPS_B = 500
    N_GRID_A = [200, 500, 2500]
    RHO_GRID_A = [0.01, 0.02, 0.05, 0.10, 0.20]
    N_GRID_B = [4000, 8000]
    Q_GRID_B = [0.02, 0.04]
    RHO_GRID_B = [0.80, 0.90, 0.95]

SCENARIOS_A = ["clean", "informative_H"]
ALPHA_BASE = np.array([-0.8, 1.2, -0.9, 0.6], dtype=float)
BETA_BASE  = np.array([ 0.2, 0.5,  1.0, -0.6], dtype=float)

# NHANES-like target prevalence of reported physician-diagnosed diabetes.
NHANES_PY1_TARGET = 0.118

# Strict positivity constant for the main revelation DGP.  With eps=0.005,
# e0(H) is guaranteed to lie in [0.005, 0.995].  The smallest target rho=0.01
# therefore remains feasible while still representing severe but valid positivity.
POSITIVITY_EPS = 0.005

# Truncation is a sensitivity analysis, not the primary DR estimator.
TRIM_TAUS = [0.001, 0.005, 0.01, 0.02, 0.05]
TRIM_N = {500, 2500}
TRIM_RHO = {0.01, 0.02, 0.05, 0.10}

# Numerical guards only.
MACHINE_E_FLOOR = 1e-8
PROB_EPS = 1e-10
MAX_ABS_THETA = 30.0

# Fixed test set for product prediction RMSE.
TEST_N = 5000 if MODE == "main" else 1500
TEST_XRAW = np.random.default_rng(BASE_SEED + 999).normal(size=(TEST_N, 3))
TEST_X = np.column_stack([np.ones(TEST_N), TEST_XRAW])

PARAM_NAMES = ["alpha0", "alpha1", "alpha2", "alpha3",
               "beta0", "beta1", "beta2", "beta3"]

# ------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------
def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def stable_pinv(A, rcond=1e-10):
    return np.linalg.pinv(A, rcond=rcond)


def pinv_psd(H, floor=1e-8):
    Hs = 0.5 * (H + H.T)
    vals, vecs = np.linalg.eigh(Hs)
    vals = np.maximum(vals, floor)
    return (vecs / vals) @ vecs.T


def numerical_hessian_from_grad(grad_fun, theta, eps=1e-4):
    d = len(theta)
    H = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        gp = grad_fun(theta + e)
        gm = grad_fun(theta - e)
        H[:, j] = (gp - gm) / (2 * eps)
    return 0.5 * (H + H.T)


def calibrate_intercept(lp, target, lo=-50.0, hi=50.0):
    if target <= 0:
        return lo
    if target >= 1:
        return hi
    f = lambda c: expit(c + lp).mean() - target
    return brentq(f, lo, hi)


def bounded_logistic_prob(eta, eps):
    return eps + (1.0 - 2.0*eps) * expit(eta)


def calibrate_bounded_intercept(lp, target, eps, lo=-60.0, hi=60.0):
    """Choose c so mean[eps+(1-2eps)expit(c+lp)] equals target."""
    if not (0.0 <= eps < 0.5):
        raise ValueError("eps must be in [0, 0.5)")
    if eps == 0:
        return calibrate_intercept(lp, target, lo, hi)
    if not (eps < target < 1.0-eps):
        raise ValueError(f"target rho={target} must lie strictly inside ({eps}, {1-eps})")
    latent_target = (target - eps) / (1.0 - 2.0*eps)
    return calibrate_intercept(lp, latent_target, lo, hi)


def kish_ess(w):
    w = np.asarray(w, float)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return 0.0
    den = np.sum(w * w)
    return float((np.sum(w) ** 2) / den) if den > 0 else 0.0


def make_starts(p, seed):
    rng = np.random.default_rng(seed)
    r1 = rng.normal(scale=0.6, size=2*p)
    r2 = rng.normal(scale=1.0, size=2*p)
    z = np.zeros(2*p)
    return [z, r1, r2, np.r_[r1[p:], r1[:p]], np.r_[r2[p:], r2[:p]]]


def fit_bfgs(loss_grad_fun, theta0, maxiter=500):
    return minimize(
        fun=lambda th: loss_grad_fun(th)[0],
        x0=np.asarray(theta0, float),
        jac=lambda th: loss_grad_fun(th)[1],
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": maxiter, "disp": False},
    )


def multi_start_fit(fun, starts, max_abs_theta=MAX_ABS_THETA):
    best = None
    best_any = None
    for th0 in starts:
        try:
            res = fit_bfgs(fun, th0)
            finite = np.isfinite(res.fun) and np.all(np.isfinite(res.x))
            if finite and (best_any is None or res.fun < best_any.fun):
                best_any = res
            if finite and np.max(np.abs(res.x)) <= max_abs_theta:
                if best is None or res.fun < best.fun:
                    best = res
        except Exception:
            pass
    if best is not None:
        return best
    if best_any is not None:
        raise RuntimeError(f"Only exploding solution: max|theta|={np.max(np.abs(best_any.x)):.3g}")
    raise RuntimeError("All optimization starts failed")

# ------------------------------------------------------------
# STRUCTURAL DATA GENERATION
# ------------------------------------------------------------
def simulate_structural_zib(n, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    xraw = rng.normal(size=(n, 3))
    X = add_intercept(xraw)
    pi = expit(X @ alpha)
    mu = expit(X @ beta)
    S = rng.binomial(1, pi)
    Y = rng.binomial(1, S * mu)
    return dict(X=X, xraw=xraw, S=S, Y=Y, pi=pi, mu=mu,
                alpha_true=np.asarray(alpha).copy(), beta_true=np.asarray(beta).copy())


def add_auxiliary_history(structural, scenario, seed):
    rng = np.random.default_rng(seed)
    out = dict(structural)
    xraw, S = structural["xraw"], structural["S"]
    if scenario == "clean":
        H = xraw.copy()
    elif scenario == "informative_H":
        B = 1.5*S + 0.6*xraw[:,0] - 0.4*xraw[:,1] + rng.normal(scale=0.7, size=len(S))
        H = np.column_stack([xraw, B])
    else:
        raise ValueError(scenario)
    out.update(H=H, scenario=scenario)
    return out


def attach_partial_revelation(base, rho_target, seed, mechanism=None, positivity_eps=0.0):
    """
    Generate R only on Y=0.

    For clean/informative_H, positivity_eps>0 uses
        e0(H)=eps+(1-2eps)expit(c+lp(H)),
    so e0(H)>=eps by construction.  Setting positivity_eps=0 gives an
    optional practical-positivity stress test.
    """
    rng = np.random.default_rng(seed)
    H, Y, S = base["H"], base["Y"], base["S"]
    zero = (Y == 0)
    scenario = base.get("scenario", "clean") if mechanism is None else mechanism

    e0 = np.ones(len(Y), float)
    intercept = np.nan
    if zero.sum() == 0:
        R = np.zeros(len(Y), int)
    else:
        if scenario == "nhanes_mcar":
            e0[zero] = rho_target
        elif scenario == "clean":
            lp = 0.9*H[:,0] - 0.7*H[:,1] + 0.4*H[:,2]
            intercept = calibrate_bounded_intercept(lp[zero], rho_target, positivity_eps)
            e0[zero] = bounded_logistic_prob(intercept + lp[zero], positivity_eps)
        elif scenario == "informative_H":
            lp = 0.5*H[:,0] - 0.3*H[:,1] + 1.8*H[:,3]
            intercept = calibrate_bounded_intercept(lp[zero], rho_target, positivity_eps)
            e0[zero] = bounded_logistic_prob(intercept + lp[zero], positivity_eps)
        else:
            raise ValueError(scenario)
        R = np.zeros(len(Y), int)
        R[zero] = rng.binomial(1, e0[zero])

    R[~zero] = 0
    W = R*S
    out = dict(base)
    out.update(R=R, W=W, e0=e0, positivity_eps=float(positivity_eps),
               revelation_intercept=float(intercept) if np.isfinite(intercept) else np.nan,
               rho_realized=float(R[zero].mean()) if zero.any() else np.nan,
               q_revealed=float(S[zero & (R==1)].mean()) if np.any(zero & (R==1)) else np.nan)
    return out

# ------------------------------------------------------------
# NHANES-LIKE TRUTH CALIBRATION
# ------------------------------------------------------------
def calibrate_truth_for_py1_q(py1_target, q_target, seed=123456, m=250000):
    """
    Solve for alpha/beta intercepts, keeping slopes fixed, so that
      E[pi(X) mu(X)] = P(Y=1) = py1_target
      E[pi(X)(1-mu(X))] / (1-E[pi(X)mu(X)]) = P(S=1|Y=0) = q_target.
    """
    rng = np.random.default_rng(seed)
    xraw = rng.normal(size=(m, 3))
    Xa = ALPHA_BASE[1:]
    Xb = BETA_BASE[1:]
    lpa = xraw @ Xa
    lpb = xraw @ Xb

    def equations(z):
        a0, b0 = z
        pi = expit(a0 + lpa)
        mu = expit(b0 + lpb)
        py1 = np.mean(pi*mu)
        p_s1_y0 = np.mean(pi*(1-mu))
        q = p_s1_y0 / max(1-py1, 1e-12)
        return np.array([py1-py1_target, q-q_target])

    sol = root(equations, x0=np.array([-2.5, 2.0]), method="hybr")
    if not sol.success or np.linalg.norm(equations(sol.x)) > 1e-7:
        ls = least_squares(equations, x0=np.array([-2.5, 2.0]), bounds=(-10,10), xtol=1e-12, ftol=1e-12)
        z = ls.x
    else:
        z = sol.x
    alpha = ALPHA_BASE.copy(); alpha[0] = z[0]
    beta  = BETA_BASE.copy();  beta[0]  = z[1]

    # deterministic verification on same integration sample
    pi = expit(alpha[0] + lpa)
    mu = expit(beta[0] + lpb)
    py1 = np.mean(pi*mu)
    q = np.mean(pi*(1-mu)) / (1-py1)
    return alpha, beta, float(py1), float(q)

# ------------------------------------------------------------
# LIKELIHOOD LOSSES / GRADIENTS
# ------------------------------------------------------------
def standard_loss_grad(theta, X, Y):
    p = X.shape[1]
    alpha, beta = theta[:p], theta[p:]
    pi, mu = expit(X@alpha), expit(X@beta)
    q = np.clip(pi*mu, PROB_EPS, 1-PROB_EPS)
    ll = np.sum(Y*np.log(q) + (1-Y)*np.log(1-q))
    ga = X.T @ ((Y-q)*(1-pi)/np.clip(1-q, PROB_EPS, None))
    gb = X.T @ ((Y-q)*(1-mu)/np.clip(1-q, PROB_EPS, None))
    return -ll, -np.r_[ga, gb]


def pr_loss_grad(theta, X, Y, R, W):
    p = X.shape[1]
    alpha, beta = theta[:p], theta[p:]
    pi, mu = expit(X@alpha), expit(X@beta)
    q = np.clip(pi*mu, PROB_EPS, 1-PROB_EPS)

    y1 = (Y==1)
    zr1 = (Y==0)&(R==1)&(W==1)
    zr0 = (Y==0)&(R==1)&(W==0)
    zu = (Y==0)&(R==0)
    ll = 0.0
    ga = np.zeros(p); gb = np.zeros(p)
    if y1.any():
        ll += np.sum(np.log(np.clip(pi[y1],PROB_EPS,None)) + np.log(np.clip(mu[y1],PROB_EPS,None)))
        ga += X[y1].T@(1-pi[y1]); gb += X[y1].T@(1-mu[y1])
    if zr1.any():
        ll += np.sum(np.log(np.clip(pi[zr1],PROB_EPS,None)) + np.log(np.clip(1-mu[zr1],PROB_EPS,None)))
        ga += X[zr1].T@(1-pi[zr1]); gb += X[zr1].T@(-mu[zr1])
    if zr0.any():
        ll += np.sum(np.log(np.clip(1-pi[zr0],PROB_EPS,None)))
        ga += X[zr0].T@(-pi[zr0])
    if zu.any():
        ll += np.sum(np.log(np.clip(1-q[zu],PROB_EPS,None)))
        ga += X[zu].T@(-(q[zu]*(1-pi[zu])/np.clip(1-q[zu],PROB_EPS,None)))
        gb += X[zu].T@(-(q[zu]*(1-mu[zu])/np.clip(1-q[zu],PROB_EPS,None)))
    return -ll, -np.r_[ga, gb]

# ------------------------------------------------------------
# NUISANCE MODELS / CROSS-FITTING
# ------------------------------------------------------------
def fit_regularized_logistic(X_train, y_train, X_pred, seed):
    seed = int(seed % (2**32 - 1))
    y = np.asarray(y_train, int)
    if len(y) == 0:
        return np.full(len(X_pred), 0.5)
    u = np.unique(y)
    if len(u) < 2:
        p = np.clip(float(y.mean()), 1e-6, 1-1e-6)
        return np.full(len(X_pred), p)
    try:
        fit = LogisticRegression(C=2.0, solver="lbfgs", max_iter=1000, random_state=seed)
        fit.fit(X_train, y)
        return np.clip(fit.predict_proba(X_pred)[:,1], 1e-8, 1-1e-8)
    except Exception:
        p = np.clip(float(y.mean()), 1e-6, 1-1e-6)
        return np.full(len(X_pred), p)


def fit_bounded_logistic(X_train, y_train, X_pred, eps, ridge=0.1):
    """
    Penalized MLE for p(x)=eps+(1-2eps) expit(b0+x^T b).
    The ridge penalty is O(1) while the log likelihood is O(n), so it vanishes
    asymptotically and mainly prevents numerical separation in sparse folds.
    """
    X_train = np.asarray(X_train, float)
    X_pred = np.asarray(X_pred, float)
    y = np.asarray(y_train, float)
    n, d = X_train.shape
    if n == 0:
        return np.full(len(X_pred), 0.5)
    a = 1.0 - 2.0*eps
    pbar = float(np.mean(y))
    # Map observed mean back to the latent logistic scale for initialization.
    tbar = np.clip((pbar-eps)/a, 1e-5, 1-1e-5)
    b0 = np.r_[np.log(tbar/(1-tbar)), np.zeros(d)]
    Xa = np.column_stack([np.ones(n), X_train])
    Xpa = np.column_stack([np.ones(len(X_pred)), X_pred])

    def fg(b):
        eta = Xa @ b
        s = expit(eta)
        p = np.clip(eps + a*s, 1e-12, 1-1e-12)
        nll = -np.sum(y*np.log(p) + (1-y)*np.log(1-p)) + 0.5*ridge*np.sum(b[1:]**2)
        dp = a*s*(1-s)
        deta = (p-y)*dp/(p*(1-p))
        grad = Xa.T @ deta
        grad[1:] += ridge*b[1:]
        return nll, grad

    try:
        res = minimize(lambda b: fg(b)[0], b0, jac=lambda b: fg(b)[1], method="BFGS",
                       options={"maxiter":1000,"gtol":1e-7})
        b = res.x if np.all(np.isfinite(res.x)) else b0
        return np.clip(eps + a*expit(Xpa@b), eps, 1-eps)
    except Exception:
        return np.full(len(X_pred), np.clip(pbar, eps, 1-eps))


def crossfit_em(data, n_splits=N_SPLITS, seed=0, propensity_mode="logistic", positivity_eps=0.0):
    seed = int(seed % (2**32 - 1))
    H, Y, R, W = data["H"], data["Y"], data["R"], data["W"]
    n = len(Y)
    ehat = np.ones(n)
    mhat = np.ones(n)
    zidx = np.where(Y==0)[0]
    if len(zidx) < 2:
        return ehat, mhat
    k = min(n_splits, len(zidx))
    if k < 2:
        return ehat, mhat
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (itr, ite) in enumerate(kf.split(zidx)):
        tr, te = zidx[itr], zidx[ite]
        if propensity_mode == "bounded":
            ehat[te] = fit_bounded_logistic(H[tr], R[tr], H[te], positivity_eps)
        elif propensity_mode == "logistic":
            ehat[te] = fit_regularized_logistic(H[tr], R[tr], H[te], seed+100*fold+1)
        else:
            raise ValueError(propensity_mode)
        rev = tr[R[tr]==1]
        if len(rev)==0:
            mhat[te] = 0.5
        else:
            mhat[te] = fit_regularized_logistic(H[rev], W[rev], H[te], seed+100*fold+2)
    floor = positivity_eps if propensity_mode == "bounded" else MACHINE_E_FLOOR
    ehat[zidx] = np.clip(ehat[zidx], max(floor,MACHINE_E_FLOOR), 1-max(floor,MACHINE_E_FLOOR))
    mhat[zidx] = np.clip(mhat[zidx], 1e-8, 1-1e-8)
    return ehat, mhat


def build_aipw_gate(Y, R, W, ehat, mhat, tau=None):
    euse = np.asarray(ehat).copy()
    if tau is not None:
        euse = np.maximum(euse, tau)
    euse = np.maximum(euse, MACHINE_E_FLOOR)
    # Deliberately NOT clipped to [0,1].
    Se = Y + (1-Y)*(mhat + R*(W-mhat)/euse)
    return Se, euse

# ------------------------------------------------------------
# FITTERS
# ------------------------------------------------------------
def fit_standard_once(structural, seed):
    X,Y = structural["X"], structural["Y"]
    p = X.shape[1]
    res = multi_start_fit(lambda th: standard_loss_grad(th,X,Y), make_starts(p,seed))
    theta = res.x
    H = numerical_hessian_from_grad(lambda th: standard_loss_grad(th,X,Y)[1], theta)
    cov = pinv_psd(H)
    se = np.sqrt(np.maximum(np.diag(cov),0))
    return dict(theta=theta,se=se,H=H,success=True,boundary=float(np.max(np.abs(theta))))


def fit_pr_mle(data, theta_init, seed):
    X,Y,R,W = data["X"],data["Y"],data["R"],data["W"]
    p = X.shape[1]
    starts = [theta_init, np.zeros(2*p)] + make_starts(p, seed)[:2]
    res = multi_start_fit(lambda th: pr_loss_grad(th,X,Y,R,W), starts)
    theta = res.x
    H = numerical_hessian_from_grad(lambda th: pr_loss_grad(th,X,Y,R,W)[1], theta)
    cov = pinv_psd(H)
    se = np.sqrt(np.maximum(np.diag(cov),0))
    return dict(theta=theta,se=se,H=H,success=True,boundary=float(np.max(np.abs(theta))))


def _solve_block_score(score_fun, jac_fun, starts):
    best = None
    for s in starts:
        try:
            rr = root(score_fun, s, jac=jac_fun, method="hybr")
            val = np.linalg.norm(score_fun(rr.x))
            bounded = np.max(np.abs(rr.x)) <= MAX_ABS_THETA
            if np.isfinite(val) and bounded and (best is None or val < best[0]):
                best = (val, rr.x, bool(rr.success))
        except Exception:
            pass
    if best is not None and best[0] < 1e-6:
        return best[1]

    # fallback: solve moments in least-squares sense, but accept only a near-root
    for s in starts:
        try:
            ls = least_squares(score_fun, s, jac=jac_fun,
                               bounds=(-MAX_ABS_THETA, MAX_ABS_THETA),
                               xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=1000)
            val = np.linalg.norm(score_fun(ls.x))
            if np.isfinite(val) and (best is None or val < best[0]):
                best = (val, ls.x, bool(ls.success))
        except Exception:
            pass
    if best is None or best[0] > 5e-5:
        raise RuntimeError("DR estimating equation has no stable numerical root")
    return best[1]


def fit_dr_from_nuisance(data, ehat, mhat, theta_init, tau=None):
    X,Y,R,W = data["X"],data["Y"],data["R"],data["W"]
    p = X.shape[1]
    Se,euse = build_aipw_gate(Y,R,W,ehat,mhat,tau=tau)
    init_a = np.asarray(theta_init[:p]) if theta_init is not None else np.zeros(p)
    init_b = np.asarray(theta_init[p:]) if theta_init is not None else np.zeros(p)

    def score_a(a):
        pi = expit(X@a)
        return (X.T@(Se-pi))/len(Y)
    def jac_a(a):
        pi = expit(X@a)
        return -(X.T@((pi*(1-pi))[:,None]*X))/len(Y)

    def score_b(b):
        mu = expit(X@b)
        return (X.T@(Y-Se*mu))/len(Y)
    def jac_b(b):
        mu = expit(X@b)
        return -(X.T@((Se*mu*(1-mu))[:,None]*X))/len(Y)

    ahat = _solve_block_score(score_a, jac_a, [init_a, np.zeros(p)])
    bhat = _solve_block_score(score_b, jac_b, [init_b, np.zeros(p)])
    theta = np.r_[ahat,bhat]

    pi = expit(X@ahat); mu = expit(X@bhat)
    Aaa = X.T@((pi*(1-pi))[:,None]*X)
    Abb = X.T@((Se*mu*(1-mu))[:,None]*X)
    A = np.block([[Aaa, np.zeros((p,p))],[np.zeros((p,p)), Abb]])
    score_i = np.concatenate([((Se-pi)[:,None]*X), ((Y-Se*mu)[:,None]*X)], axis=1)
    B = score_i.T@score_i
    Ainv = stable_pinv(A)
    cov = Ainv@B@Ainv.T
    cov = 0.5*(cov+cov.T)
    se = np.sqrt(np.maximum(np.diag(cov),0))
    svals = np.linalg.svd(A,compute_uv=False)

    return dict(theta=theta,se=se,H=A,success=True,
                boundary=float(np.max(np.abs(theta))),
                Se_min=float(np.min(Se)),Se_max=float(np.max(Se)),
                min_sing_A=float(np.min(svals)),tau=np.nan if tau is None else tau,
                euse=euse)

# ------------------------------------------------------------
# METRICS / DIAGNOSTICS
# ------------------------------------------------------------
def product_prediction_rmse(theta, alpha_true, beta_true):
    p = len(alpha_true)
    qhat = expit(TEST_X@theta[:p])*expit(TEST_X@theta[p:])
    q0 = expit(TEST_X@alpha_true)*expit(TEST_X@beta_true)
    return float(np.sqrt(np.mean((qhat-q0)**2)))


def reveal_diagnostics(data, ehat=None):
    Y,R,S = data["Y"],data["R"],data["S"]
    zero = (Y==0); rev = zero&(R==1)
    n0 = int(zero.sum()); nr = int(rev.sum())
    nr1 = int((rev&(S==1)).sum()); nr0 = int((rev&(S==0)).sum())
    out = dict(n_zero=n0,n_rev=nr,n_rev1=nr1,n_rev0=nr0,
               rho_realized=float(nr/n0) if n0 else np.nan,
               q_realized=float(nr1/nr) if nr else np.nan)
    if zero.any() and "e0" in data:
        e0z = np.asarray(data["e0"])[zero]
        out["true_e_min_zero"] = float(np.min(e0z))
        out["true_e_q01_zero"] = float(np.quantile(e0z,0.01))
        out["true_e_mean_zero"] = float(np.mean(e0z))
        out["positivity_eps"] = float(data.get("positivity_eps",0.0))
    else:
        out.update(true_e_min_zero=np.nan,true_e_q01_zero=np.nan,true_e_mean_zero=np.nan,positivity_eps=np.nan)
    if ehat is not None and nr:
        w = 1/np.maximum(ehat[rev],MACHINE_E_FLOOR)
        out["ess"] = kish_ess(w)
        for s,label in [(1,"ess1"),(0,"ess0")]:
            idx = rev&(S==s)
            out[label] = kish_ess(1/np.maximum(ehat[idx],MACHINE_E_FLOOR)) if idx.any() else 0.0
        inv = 1/np.maximum(ehat[rev],MACHINE_E_FLOOR)
        out["inv_e_q99"] = float(np.quantile(inv,0.99))
        out["inv_e_max"] = float(np.max(inv))
        out["e_min_zero"] = float(np.min(ehat[zero]))
        out["e_q01_zero"] = float(np.quantile(ehat[zero],0.01))
    else:
        out.update(ess=np.nan,ess1=np.nan,ess0=np.nan,inv_e_q99=np.nan,
                   inv_e_max=np.nan,e_min_zero=np.nan,e_q01_zero=np.nan)
    return out


def fit_to_rows(experiment, scenario, n, rho, q_target, rep, method,
                fit, alpha_true, beta_true, diagnostics, extra=None):
    theta0 = np.r_[alpha_true,beta_true]
    rows_rep=[]; rows_coef=[]
    if fit is None:
        row = dict(experiment=experiment,scenario=scenario,n=n,rho=rho,q_target=q_target,
                   rep=rep,method=method,fit_success=0)
        row.update(diagnostics)
        if extra: row.update(extra)
        return [row], []

    th = fit["theta"]; se=fit["se"]
    err = th-theta0
    cov_ind = (theta0>=th-1.96*se)&(theta0<=th+1.96*se)
    row = dict(experiment=experiment,scenario=scenario,n=n,rho=rho,q_target=q_target,
               rep=rep,method=method,fit_success=1,
               param_rmse=float(np.sqrt(np.mean(err**2))),
               param_mae=float(np.mean(np.abs(err))),
               coverage_mean=float(np.mean(cov_ind)),
               pred_rmse=product_prediction_rmse(th,alpha_true,beta_true),
               max_abs_theta=float(np.max(np.abs(th))),
               min_eig_or_sing=float(np.min(np.linalg.eigvalsh(0.5*(fit["H"]+fit["H"].T)))) if method!="DR_CF" and not method.startswith("DR_TAU") else float(fit.get("min_sing_A",np.nan)),
               Se_min=float(fit.get("Se_min",np.nan)),Se_max=float(fit.get("Se_max",np.nan)))
    row.update(diagnostics)
    if extra: row.update(extra)
    rows_rep.append(row)

    for j,name in enumerate(PARAM_NAMES):
        rows_coef.append(dict(experiment=experiment,scenario=scenario,n=n,rho=rho,q_target=q_target,
                              rep=rep,method=method,param=name,true=float(theta0[j]),
                              estimate=float(th[j]),se=float(se[j]),error=float(err[j]),
                              covered=int(cov_ind[j]),fit_success=1))
    return rows_rep, rows_coef

# ------------------------------------------------------------
# ONE-REPLICATION WORKERS
# ------------------------------------------------------------
def run_one_A(n, rep, experiment="A", positivity_eps=POSITIVITY_EPS, do_trim=False):
    rep_rows=[]; coef_rows=[]
    seed0 = BASE_SEED + 10_000_000*n + 1000*rep
    structural = simulate_structural_zib(n,ALPHA_BASE,BETA_BASE,seed0)
    try:
        std = fit_standard_once(structural,seed0+7)
    except Exception:
        std = None

    for si,scenario in enumerate(SCENARIOS_A):
        base = add_auxiliary_history(structural,scenario,seed0+100_000*(si+1)+11)
        for rho in RHO_GRID_A:
            data = attach_partial_revelation(
                base,rho,seed0+100_000*(si+1)+int(rho*100000)+31,
                positivity_eps=positivity_eps)
            diag0 = reveal_diagnostics(data,None)

            # Repeat common STD baseline inside each design cell for easy tables.
            r,c = fit_to_rows(experiment,scenario,n,rho,np.nan,rep,"STD_ZIB",std,
                              ALPHA_BASE,BETA_BASE,diag0)
            rep_rows += r; coef_rows += c

            try:
                pr = fit_pr_mle(data, std["theta"] if std is not None else np.zeros(8), seed0+43)
            except Exception:
                pr = None
            r,c = fit_to_rows(experiment,scenario,n,rho,np.nan,rep,"PR_MLE",pr,
                              ALPHA_BASE,BETA_BASE,diag0)
            rep_rows += r; coef_rows += c

            propensity_mode = "bounded" if positivity_eps > 0 else "logistic"
            try:
                ehat,mhat = crossfit_em(
                    data,N_SPLITS,seed0+59+int(rho*100000)+si*1000,
                    propensity_mode=propensity_mode,positivity_eps=positivity_eps)
                diag = reveal_diagnostics(data,ehat)
                dr = fit_dr_from_nuisance(data,ehat,mhat,
                        pr["theta"] if pr is not None else (std["theta"] if std is not None else np.zeros(8)),tau=None)
            except Exception:
                ehat=mhat=None; diag=diag0; dr=None
            r,c = fit_to_rows(experiment,scenario,n,rho,np.nan,rep,"DR_CF",dr,
                              ALPHA_BASE,BETA_BASE,diag)
            rep_rows += r; coef_rows += c

            if (do_trim and RUN_TRIM_SENSITIVITY and scenario=="informative_H" and
                n in TRIM_N and rho in TRIM_RHO and ehat is not None):
                for tau in TRIM_TAUS:
                    try:
                        drt = fit_dr_from_nuisance(data,ehat,mhat,
                              pr["theta"] if pr is not None else (std["theta"] if std is not None else np.zeros(8)),tau=tau)
                    except Exception:
                        drt=None
                    r,c = fit_to_rows(experiment,scenario,n,rho,np.nan,rep,f"DR_TAU_{tau:g}",drt,
                                      ALPHA_BASE,BETA_BASE,diag,extra={"tau":tau})
                    rep_rows += r; coef_rows += c
    return rep_rows,coef_rows


def run_one_B(n, q_target, alpha_true, beta_true, rep):
    rep_rows=[]; coef_rows=[]
    qcode=int(round(q_target*10000))
    seed0=BASE_SEED+700_000_000+10_000_000*n+100_000*qcode+1000*rep
    structural=simulate_structural_zib(n,alpha_true,beta_true,seed0)
    base=dict(structural)
    base.update(H=structural["xraw"].copy(),scenario="nhanes_mcar")
    try:
        std=fit_standard_once(structural,seed0+7)
    except Exception:
        std=None
    for rho in RHO_GRID_B:
        data=attach_partial_revelation(base,rho,seed0+int(rho*100000)+31,mechanism="nhanes_mcar")
        diag0=reveal_diagnostics(data,None)
        r,c=fit_to_rows("B","nhanes_like",n,rho,q_target,rep,"STD_ZIB",std,
                        alpha_true,beta_true,diag0)
        rep_rows+=r;coef_rows+=c
        try:
            pr=fit_pr_mle(data,std["theta"] if std is not None else np.zeros(8),seed0+43)
        except Exception:
            pr=None
        r,c=fit_to_rows("B","nhanes_like",n,rho,q_target,rep,"PR_MLE",pr,
                        alpha_true,beta_true,diag0)
        rep_rows+=r;coef_rows+=c
        try:
            ehat,mhat=crossfit_em(data,N_SPLITS,seed0+59+int(rho*100000), propensity_mode="logistic", positivity_eps=0.0)
            diag=reveal_diagnostics(data,ehat)
            dr=fit_dr_from_nuisance(data,ehat,mhat,
                   pr["theta"] if pr is not None else (std["theta"] if std is not None else np.zeros(8)),tau=None)
        except Exception:
            diag=diag0;dr=None
        r,c=fit_to_rows("B","nhanes_like",n,rho,q_target,rep,"DR_CF",dr,
                        alpha_true,beta_true,diag)
        rep_rows+=r;coef_rows+=c
    return rep_rows,coef_rows

# ------------------------------------------------------------
# SUMMARIES WITH MCSE
# ------------------------------------------------------------
def summarize_method(rep_df):
    ok=rep_df[rep_df.fit_success==1].copy()
    group=["experiment","scenario","n","rho","q_target","method"]
    out=[]
    # include failures using total group size from all rows
    totals=rep_df.groupby(group,dropna=False).size().rename("n_attempt").reset_index()
    for key,g in ok.groupby(group,dropna=False):
        d=dict(zip(group,key if isinstance(key,tuple) else (key,)))
        M=len(g)
        d["n_success"]=M
        for col in ["param_rmse","param_mae","coverage_mean","pred_rmse","rho_realized","q_realized",
                    "n_rev","n_rev1","n_rev0","ess","ess1","ess0","inv_e_q99","inv_e_max",
                    "e_min_zero","e_q01_zero","true_e_min_zero","true_e_q01_zero",
                    "true_e_mean_zero","positivity_eps","Se_min","Se_max"]:
            if col in g:
                x=pd.to_numeric(g[col],errors="coerce").dropna().to_numpy()
                d[col+"_mean"]=float(np.mean(x)) if len(x) else np.nan
                d[col+"_mcse"]=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
        out.append(d)
    out=pd.DataFrame(out)
    out=totals.merge(out,on=group,how="left")
    out["failure_rate"]=1-out["n_success"].fillna(0)/out["n_attempt"]
    return out


def summarize_coefficients(coef_df):
    group=["experiment","scenario","n","rho","q_target","method","param"]
    out=[]
    for key,g in coef_df.groupby(group,dropna=False):
        d=dict(zip(group,key if isinstance(key,tuple) else (key,)))
        est=g.estimate.to_numpy(float); err=g.error.to_numpy(float); se=g.se.to_numpy(float); cov=g.covered.to_numpy(float)
        M=len(g); rmse=np.sqrt(np.mean(err**2))
        d.update(
            reps=M,
            mean_est=float(np.mean(est)),
            true=float(g.true.iloc[0]),
            bias=float(np.mean(err)),
            bias_mcse=float(np.std(err,ddof=1)/np.sqrt(M)) if M>1 else np.nan,
            rmse=float(rmse),
            rmse_mcse=float(np.std(err**2,ddof=1)/(2*rmse*np.sqrt(M))) if M>1 and rmse>0 else np.nan,
            empirical_sd=float(np.std(est,ddof=1)) if M>1 else np.nan,
            mean_se=float(np.mean(se)),
            coverage=float(np.mean(cov)),
            coverage_mcse=float(np.sqrt(np.mean(cov)*(1-np.mean(cov))/M)) if M else np.nan,
        )
        d["se_to_esd"]=d["mean_se"]/d["empirical_sd"] if d["empirical_sd"] and np.isfinite(d["empirical_sd"]) else np.nan
        out.append(d)
    return pd.DataFrame(out)

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------
def save_plots(method_summary):
    plotdir=os.path.join(OUTDIR,"plots");os.makedirs(plotdir,exist_ok=True)

    # Experiment A: param RMSE by rho, separately by n and scenario.
    A=method_summary[(method_summary.experiment=="A") & method_summary.method.isin(["STD_ZIB","PR_MLE","DR_CF"])]
    for scenario in SCENARIOS_A:
        for n in N_GRID_A:
            s=A[(A.scenario==scenario)&(A.n==n)]
            if s.empty: continue
            plt.figure(figsize=(7,5))
            for method in ["STD_ZIB","PR_MLE","DR_CF"]:
                z=s[s.method==method].sort_values("rho")
                if len(z): plt.plot(z.rho,z.param_rmse_mean,marker="o",label=method)
            plt.xlabel("rho = P(R=1 | Y=0)");plt.ylabel("Parameter RMSE")
            plt.title(f"Experiment A: {scenario}, n={n}");plt.legend();plt.grid(alpha=.25);plt.tight_layout()
            plt.savefig(os.path.join(plotdir,f"A_rmse_{scenario}_n{n}.png"),dpi=170);plt.close()

    # Experiment B: high-rho rare-positive regime.
    B=method_summary[(method_summary.experiment=="B") & method_summary.method.isin(["STD_ZIB","PR_MLE","DR_CF"])]
    for q in Q_GRID_B:
        for n in N_GRID_B:
            s=B[(np.isclose(B.q_target,q))&(B.n==n)]
            if s.empty: continue
            plt.figure(figsize=(7,5))
            for method in ["STD_ZIB","PR_MLE","DR_CF"]:
                z=s[s.method==method].sort_values("rho")
                if len(z): plt.plot(z.rho,z.param_rmse_mean,marker="o",label=method)
            plt.xlabel("rho = P(R=1 | Y=0)");plt.ylabel("Parameter RMSE")
            plt.title(f"NHANES-like: q={q:.2f}, n={n}");plt.legend();plt.grid(alpha=.25);plt.tight_layout()
            plt.savefig(os.path.join(plotdir,f"B_rmse_q{q:.2f}_n{n}.png"),dpi=170);plt.close()

    # Trimming sensitivity: delta RMSE relative to untruncated DR.
    trim_exp = "A_stress" if (method_summary.experiment=="A_stress").any() else "A"
    T=method_summary[(method_summary.experiment==trim_exp)&(method_summary.scenario=="informative_H")]
    if len(T):
        base=T[T.method=="DR_CF"][["n","rho","param_rmse_mean"]].rename(columns={"param_rmse_mean":"base_rmse"})
        tt=T[T.method.str.startswith("DR_TAU",na=False)].merge(base,on=["n","rho"],how="left")
        if len(tt):
            tt["delta_rmse"]=tt.param_rmse_mean-tt.base_rmse
            tt.to_csv(os.path.join(OUTDIR,"trimming_delta_rmse.csv"),index=False)



# ============================================================
# TARGETED B=1000 BOOTSTRAP PHASE
# Reported cells: (2500, 0.02), (2500, 0.05), (2500, 0.10), (500, 0.20)
# ============================================================

import time
from pathlib import Path

# ------------------------- USER CONFIG -------------------------
# Defaults reproduce the reported bootstrap study. Checkpoint/resume is enabled.
QUICK_TEST = bool(int(os.environ.get("PRZIB_BOOT_QUICK", "0")))
N_JOBS_BOOT = min(int(os.environ.get("PRZIB_N_JOBS", "7")), os.cpu_count() or 1)
MC_REPS_BOOT = 3 if QUICK_TEST else 500
BOOT_B = 25 if QUICK_TEST else 1000
MC_BATCH = 2 if QUICK_TEST else max(1, N_JOBS_BOOT)
BOOT_ALPHA = 0.05
MIN_BOOT_SUCCESS_FRAC = 0.80
TARGET_CELLS = [(2500, 0.02), (2500, 0.05), (2500, 0.10), (500, 0.20)]
SPLIT_NAME = "reported_cells"
BOOT_OUTDIR = Path(os.environ.get("PRZIB_BOOT_OUTDIR", "results/bootstrap"))
BOOT_OUTDIR.mkdir(parents=True, exist_ok=True)

# Methods bootstrapped. PR is essentially free once we bootstrap DR, because
# the bootstrap PR fit is also used as the starting value for DR.
METHODS_BOOT = ["PR_MLE", "DR_CF"]

print("="*100)
print("PR-ZIB TARGETED B=1000 BOOTSTRAP")
print("="*100)
print("Cells:", TARGET_CELLS)
print(f"MC reps/cell={MC_REPS_BOOT}, B={BOOT_B}, cores={N_JOBS_BOOT}, strict positivity eps={POSITIVITY_EPS}")
print("Each bootstrap resample refits PR_MLE, bounded-logistic e(H), m(H), and DR_CF.")
print("Checkpointing after each outer-MC batch; rerunning resumes completed reps.")

# ------------------------- HELPERS -------------------------
def subset_data_boot(data, idx):
    """Bootstrap rows. Fits use only observed variables; latent S/e0 are retained for diagnostics only."""
    n = len(idx)
    out = {}
    for k,v in data.items():
        if isinstance(v, np.ndarray) and len(v)==len(data["Y"]):
            out[k] = v[idx].copy()
        else:
            out[k] = v
    # rho/q are descriptive for the resample, not DGP inputs.
    zero = out["Y"]==0
    rev = zero & (out["R"]==1)
    out["rho_realized"] = float(out["R"][zero].mean()) if zero.any() else np.nan
    out["q_revealed"] = float(out["S"][rev].mean()) if rev.any() else np.nan
    return out


def percentile_ci(draws, alpha=0.05):
    draws = np.asarray(draws, float)
    if draws.ndim != 2 or draws.shape[0] < 2:
        return None, None
    lo = np.quantile(draws, alpha/2, axis=0)
    hi = np.quantile(draws, 1-alpha/2, axis=0)
    return lo, hi


def make_target_dataset(n, rho, rep):
    """Exactly matches Experiment A / informative_H / strict-positivity seed convention."""
    seed0 = BASE_SEED + 10_000_000*n + 1000*rep
    structural = simulate_structural_zib(n, ALPHA_BASE, BETA_BASE, seed0)
    # informative_H is the second scenario in the main code (si=1).
    base = add_auxiliary_history(structural, "informative_H", seed0 + 200_000 + 11)
    data = attach_partial_revelation(
        base, rho, seed0 + 200_000 + int(rho*100000) + 31,
        positivity_eps=POSITIVITY_EPS)
    return seed0, structural, data


def fit_original_dataset(structural, data, rho, seed0):
    try:
        std = fit_standard_once(structural, seed0+7)
    except Exception:
        std = None
    try:
        pr = fit_pr_mle(data, std["theta"] if std is not None else np.zeros(8), seed0+43)
    except Exception:
        pr = None
    try:
        ehat, mhat = crossfit_em(
            data, N_SPLITS, seed0+59+int(rho*100000)+1000,
            propensity_mode="bounded", positivity_eps=POSITIVITY_EPS)
        dr = fit_dr_from_nuisance(
            data, ehat, mhat,
            pr["theta"] if pr is not None else (std["theta"] if std is not None else np.zeros(8)),
            tau=None)
        diag = reveal_diagnostics(data, ehat)
    except Exception:
        ehat = mhat = None
        dr = None
        diag = reveal_diagnostics(data, None)
    return std, pr, dr, diag


def one_bootstrap_draw(data, rho, seed0, b, pr_start, dr_start):
    """One nonparametric bootstrap resample; nuisance functions and cross-fitting are re-fit."""
    n = len(data["Y"])
    rng = np.random.default_rng(seed0 + 50_000_000 + 100_003*b)
    idx = rng.integers(0, n, size=n)
    bd = subset_data_boot(data, idx)

    # PR fit first: used both as a bootstrap estimate and as the DR starting value.
    try:
        prb = fit_pr_mle(bd, pr_start, seed0 + 60_000_000 + 100_003*b)
        th_pr = prb["theta"].copy()
        pr_ok = True
    except Exception:
        th_pr = np.full(8, np.nan)
        prb = None
        pr_ok = False

    # Refit nuisance models and reconstruct the cross-fitted AIPW pseudo-gate.
    try:
        eb, mb = crossfit_em(
            bd, N_SPLITS, seed0 + 70_000_000 + 100_003*b,
            propensity_mode="bounded", positivity_eps=POSITIVITY_EPS)
        init = th_pr if pr_ok else dr_start
        drb = fit_dr_from_nuisance(bd, eb, mb, init, tau=None)
        th_dr = drb["theta"].copy()
        dr_ok = True
    except Exception:
        th_dr = np.full(8, np.nan)
        dr_ok = False

    return th_pr, th_dr, pr_ok, dr_ok


def run_one_mc_bootstrap(n, rho, rep):
    t0 = time.time()
    seed0, structural, data = make_target_dataset(n, rho, rep)
    std, pr, dr, diag = fit_original_dataset(structural, data, rho, seed0)
    theta0 = np.r_[ALPHA_BASE, BETA_BASE]

    pr_orig_ok = pr is not None and np.all(np.isfinite(pr["theta"]))
    dr_orig_ok = dr is not None and np.all(np.isfinite(dr["theta"]))
    pr_start = pr["theta"] if pr_orig_ok else (std["theta"] if std is not None else np.zeros(8))
    dr_start = dr["theta"] if dr_orig_ok else pr_start

    pr_draws=[]; dr_draws=[]
    pr_fail=0; dr_fail=0
    for b in range(BOOT_B):
        th_pr, th_dr, pr_ok, dr_ok = one_bootstrap_draw(data, rho, seed0, b, pr_start, dr_start)
        if pr_ok:
            pr_draws.append(th_pr)
        else:
            pr_fail += 1
        if dr_ok:
            dr_draws.append(th_dr)
        else:
            dr_fail += 1

    rows_rep=[]; rows_coef=[]
    for method, orig, draws, bfail in [
        ("PR_MLE", pr, pr_draws, pr_fail),
        ("DR_CF", dr, dr_draws, dr_fail),
    ]:
        orig_ok = orig is not None and np.all(np.isfinite(orig["theta"]))
        D = np.asarray(draws, float) if len(draws) else np.empty((0,8))
        bsucc = int(len(D))
        bfrac = bsucc/BOOT_B
        valid80 = int(bfrac >= MIN_BOOT_SUCCESS_FRAC)
        lo, hi = percentile_ci(D, BOOT_ALPHA) if bsucc >= 2 else (None, None)

        if orig_ok:
            th = np.asarray(orig["theta"], float)
            se = np.asarray(orig["se"], float)
            sand_lo = th - 1.96*se
            sand_hi = th + 1.96*se
            sand_cov = ((theta0>=sand_lo)&(theta0<=sand_hi)).astype(int)
            rmse = float(np.sqrt(np.mean((th-theta0)**2)))
        else:
            th=np.full(8,np.nan); se=np.full(8,np.nan)
            sand_lo=sand_hi=np.full(8,np.nan); sand_cov=np.full(8,np.nan); rmse=np.nan

        if lo is not None:
            boot_cov = ((theta0>=lo)&(theta0<=hi)).astype(int)
            boot_len = hi-lo
        else:
            lo=hi=np.full(8,np.nan); boot_cov=np.full(8,np.nan); boot_len=np.full(8,np.nan)

        rows_rep.append(dict(
            split=SPLIT_NAME,n=n,rho=rho,rep=rep,method=method,
            original_fit_success=int(orig_ok),param_rmse=rmse,
            sandwich_coverage_mean=float(np.nanmean(sand_cov)) if orig_ok else np.nan,
            bootstrap_B=BOOT_B,bootstrap_success=bsucc,bootstrap_failure=bfail,
            bootstrap_success_fraction=bfrac,bootstrap_valid80=valid80,
            bootstrap_coverage_mean=float(np.nanmean(boot_cov)) if lo is not None else np.nan,
            bootstrap_interval_length_mean=float(np.nanmean(boot_len)) if lo is not None else np.nan,
            n_zero=diag.get("n_zero",np.nan),n_rev=diag.get("n_rev",np.nan),
            n_rev1=diag.get("n_rev1",np.nan),n_rev0=diag.get("n_rev0",np.nan),
            ess=diag.get("ess",np.nan),ess1=diag.get("ess1",np.nan),ess0=diag.get("ess0",np.nan),
            true_e_min_zero=diag.get("true_e_min_zero",np.nan),
            elapsed_sec=float(time.time()-t0)
        ))

        for j,pname in enumerate(PARAM_NAMES):
            rows_coef.append(dict(
                split=SPLIT_NAME,n=n,rho=rho,rep=rep,method=method,param=pname,true=float(theta0[j]),
                estimate=float(th[j]) if orig_ok else np.nan,se=float(se[j]) if orig_ok else np.nan,
                sandwich_lo=float(sand_lo[j]) if orig_ok else np.nan,
                sandwich_hi=float(sand_hi[j]) if orig_ok else np.nan,
                sandwich_covered=float(sand_cov[j]) if orig_ok else np.nan,
                boot_lo=float(lo[j]),boot_hi=float(hi[j]),
                boot_covered=float(boot_cov[j]) if np.isfinite(boot_cov[j]) else np.nan,
                boot_length=float(boot_len[j]),bootstrap_success=bsucc,
                bootstrap_success_fraction=bfrac,bootstrap_valid80=valid80,
                original_fit_success=int(orig_ok)
            ))

    return rows_rep, rows_coef


def save_checkpoint(rep_rows, coef_rows):
    rep_df=pd.DataFrame(rep_rows)
    coef_df=pd.DataFrame(coef_rows)
    rep_df.to_csv(BOOT_OUTDIR/"bootstrap_replicate_results.csv",index=False)
    coef_df.to_csv(BOOT_OUTDIR/"bootstrap_coefficient_results.csv",index=False)
    return rep_df,coef_df


def summarize_bootstrap(rep_df, coef_df):
    # Replicate-level method summary. Bootstrap coverage is reported both for all
    # computed percentile intervals and restricted to reps with >=80% successful draws.
    mrows=[]
    for key,g in rep_df.groupby(["n","rho","method"],dropna=False):
        n,rho,method=key
        orig=g[g.original_fit_success==1]
        valid=g[(g.original_fit_success==1)&(g.bootstrap_valid80==1)&g.bootstrap_coverage_mean.notna()]
        row=dict(n=n,rho=rho,method=method,n_attempt=len(g),
                 original_success=int(g.original_fit_success.sum()),
                 original_failure_rate=float(1-g.original_fit_success.mean()),
                 mean_boot_success_fraction=float(g.bootstrap_success_fraction.mean()),
                 valid80_reps=int(len(valid)),
                 param_rmse_mean=float(orig.param_rmse.mean()) if len(orig) else np.nan,
                 sandwich_coverage_mean=float(orig.sandwich_coverage_mean.mean()) if len(orig) else np.nan,
                 bootstrap_coverage_mean_valid80=float(valid.bootstrap_coverage_mean.mean()) if len(valid) else np.nan,
                 bootstrap_interval_length_mean_valid80=float(valid.bootstrap_interval_length_mean.mean()) if len(valid) else np.nan)
        if len(valid):
            x=valid.bootstrap_coverage_mean.to_numpy(float)
            row["bootstrap_coverage_mcse_across_mc"] = float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
        else:
            row["bootstrap_coverage_mcse_across_mc"] = np.nan
        mrows.append(row)
    ms=pd.DataFrame(mrows)

    crows=[]
    cc=coef_df[(coef_df.original_fit_success==1)&(coef_df.bootstrap_valid80==1)].copy()
    for key,g in cc.groupby(["n","rho","method","param"],dropna=False):
        n,rho,method,param=key
        sand=g.sandwich_covered.dropna().to_numpy(float)
        boot=g.boot_covered.dropna().to_numpy(float)
        row=dict(n=n,rho=rho,method=method,param=param,reps=len(g),
                 sandwich_coverage=float(np.mean(sand)) if len(sand) else np.nan,
                 sandwich_coverage_mcse=float(np.sqrt(np.mean(sand)*(1-np.mean(sand))/len(sand))) if len(sand) else np.nan,
                 bootstrap_coverage=float(np.mean(boot)) if len(boot) else np.nan,
                 bootstrap_coverage_mcse=float(np.sqrt(np.mean(boot)*(1-np.mean(boot))/len(boot))) if len(boot) else np.nan,
                 mean_boot_length=float(g.boot_length.mean()),
                 mean_boot_success_fraction=float(g.bootstrap_success_fraction.mean()))
        crows.append(row)
    cs=pd.DataFrame(crows)
    ms.to_csv(BOOT_OUTDIR/"bootstrap_method_summary.csv",index=False)
    cs.to_csv(BOOT_OUTDIR/"bootstrap_coefficient_summary.csv",index=False)
    return ms,cs

# ------------------------- RESUME / RUN -------------------------
rep_file=BOOT_OUTDIR/"bootstrap_replicate_results.csv"
coef_file=BOOT_OUTDIR/"bootstrap_coefficient_results.csv"
if rep_file.exists() and coef_file.exists():
    old_rep=pd.read_csv(rep_file)
    old_coef=pd.read_csv(coef_file)
    rep_rows=old_rep.to_dict("records")
    coef_rows=old_coef.to_dict("records")
    done=set((int(r.n),float(r.rho),int(r.rep)) for r in old_rep.itertuples())
    print(f"Resuming: {len(done)} outer MC datasets already completed.")
else:
    rep_rows=[]; coef_rows=[]; done=set()

jobs=[]
for n,rho in TARGET_CELLS:
    for rep in range(MC_REPS_BOOT):
        key=(int(n),float(rho),int(rep))
        if key not in done:
            jobs.append(key)

print(f"Remaining outer MC datasets: {len(jobs)}")
for start in range(0,len(jobs),MC_BATCH):
    batch=jobs[start:start+MC_BATCH]
    print(f"\nBatch {start//MC_BATCH+1}: {batch}")
    ans=Parallel(n_jobs=N_JOBS_BOOT,backend="loky",verbose=10)(
        delayed(run_one_mc_bootstrap)(n,rho,rep) for n,rho,rep in batch)
    for rr,cc in ans:
        rep_rows.extend(rr); coef_rows.extend(cc)
    rep_df,coef_df=save_checkpoint(rep_rows,coef_rows)
    print(f"Checkpoint: completed {len(set((int(r.n),float(r.rho),int(r.rep)) for r in rep_df.itertuples()))} datasets")

rep_df,coef_df=save_checkpoint(rep_rows,coef_rows)
ms,cs=summarize_bootstrap(rep_df,coef_df)

config=dict(split=SPLIT_NAME,target_cells=TARGET_CELLS,MC_REPS_BOOT=MC_REPS_BOOT,
            BOOT_B=BOOT_B,N_JOBS=N_JOBS_BOOT,N_SPLITS=N_SPLITS,
            positivity_eps=POSITIVITY_EPS,alpha=ALPHA_BASE.tolist(),beta=BETA_BASE.tolist(),
            bootstrap="nonparametric row bootstrap; refit PR, bounded e, m, 5-fold cross-fitting, DR",
            interval="percentile",alpha_level=BOOT_ALPHA,min_boot_success_frac=MIN_BOOT_SUCCESS_FRAC,
            quick_test=QUICK_TEST)
with open(BOOT_OUTDIR/"config.json","w") as f: json.dump(config,f,indent=2)

pd.set_option("display.width",180)
pd.set_option("display.max_rows",100)
print("\n"+"="*100)
print("BOOTSTRAP METHOD SUMMARY")
print("="*100)
print(ms.round(4).to_string(index=False))
print("\nCoefficient-wise bootstrap coverage saved to bootstrap_coefficient_summary.csv")

zip_base=str(BOOT_OUTDIR.parent / "pr_zib_targeted_bootstrap_results")
if os.path.exists(zip_base+".zip"):
    os.remove(zip_base+".zip")
shutil.make_archive(zip_base,"zip",BOOT_OUTDIR)
print("\nSaved:",zip_base+".zip")
print("Done.")