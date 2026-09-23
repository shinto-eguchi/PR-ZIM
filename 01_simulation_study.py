# ============================================================
# PR-ZIB simulation study for the manuscript
#
# Reported experiments
#   A. Positivity-respecting design:
#      n x rho x {clean revelation, informative-history revelation}.
#      The revelation propensity is bounded away from 0 and 1 by construction.
#      Methods: STD_ZIB, PR_MLE, and five-fold cross-fitted DR_CF.
#      Outputs include aggregate and coefficient-level Monte Carlo summaries,
#      product-prediction error, revelation diagnostics, and Kish-type ESS.
#
#   B. NHANES-like high-revelation / rare-positive-label design:
#      P(Y=1) approximately 0.118, q in {0.02, 0.04},
#      rho in {0.80, 0.90, 0.95}, n in {4000, 8000}.
#
# Implementation used in the reported analysis
#   * The AIPW pseudo-gate is not clipped to [0,1].
#   * DR parameters are obtained by solving the estimating equations directly.
#   * The primary DR analysis uses only a machine-level numerical floor for e(H);
#     it does not use statistical propensity truncation.
#   * In Experiment A the e-model uses the correctly specified bounded-logistic
#     family, while m(H) is an ordinary logistic working model.
#
# Defaults reproduce the reported Monte Carlo design (500 replications).
# Set PRZIB_MODE=quick for a short smoke test.
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
MODE = os.environ.get("PRZIB_MODE", "main").lower()  # "quick" or "main"
N_JOBS = min(int(os.environ.get("PRZIB_N_JOBS", "7")), os.cpu_count() or 1)
BASE_SEED = 20260920
N_SPLITS = 5
OUTDIR = os.environ.get("PRZIB_OUTDIR", "results/simulation")
os.makedirs(OUTDIR, exist_ok=True)

RUN_EXPERIMENT_A = True
RUN_EXPERIMENT_B = True
# Optional exploratory practical-positivity stress test; not part of the reported tables.
RUN_STRESS_TEST = False
# Optional truncation sensitivity; not part of the reported tables.
RUN_TRIM_SENSITIVITY = False

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
    optional logistic-normal practical-positivity stress test.
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
                    "true_e_mean_zero","positivity_eps","Se_min","Se_max","min_eig_or_sing"]:
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
            median_se=float(np.median(se)),
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

    # Manuscript Figure 1 and Figure 2 panels for n=2500.
    # These use the exact filenames referenced by the manuscript.
    title_map = {"clean": "Clean revelation", "informative_H": "Informative-history revelation"}
    for scenario in SCENARIOS_A:
        s = A[(A.scenario==scenario) & (A.n==2500)].sort_values("rho")
        if not s.empty:
            fig, ax = plt.subplots(figsize=(7,5))
            for method, marker in [("PR_MLE","o"),("DR_CF","s")]:
                z = s[s.method==method].sort_values("rho")
                if len(z):
                    ax.plot(z.rho, z.param_rmse_mean, marker=marker, label=method)
            z0 = s[s.method=="STD_ZIB"]
            if len(z0):
                ax.axhline(float(z0.param_rmse_mean.iloc[0]), linestyle="--", label="STD_ZIB")
            ax.set_xticks(RHO_GRID_A, [f"{x:.2f}" for x in RHO_GRID_A])
            ax.set_xlabel(r"Target revelation rate $\rho$")
            ax.set_ylabel("Mean parameter RMSE")
            ax.set_title(title_map[scenario])
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plotdir, f"fig1_rmse_{'clean' if scenario=='clean' else 'informative'}_20260922.png"), dpi=170)
            plt.close(fig)

            zpr = s[s.method=="PR_MLE"].sort_values("rho")
            if len(zpr) and "min_eig_or_sing_mean" in zpr.columns:
                fig, ax = plt.subplots(figsize=(7,5))
                ax.plot(zpr.rho, zpr.min_eig_or_sing_mean, marker="o")
                ax.set_xticks(RHO_GRID_A, [f"{x:.2f}" for x in RHO_GRID_A])
                ax.set_xlabel(r"Target revelation rate $\rho$")
                ax.set_ylabel("Mean minimum curvature eigenvalue")
                ax.set_title(title_map[scenario])
                fig.tight_layout()
                fig.savefig(os.path.join(plotdir, f"fig2_mineig_{'clean' if scenario=='clean' else 'informative'}_20260922.png"), dpi=170)
                plt.close(fig)

    # Trimming sensitivity: delta RMSE relative to untruncated DR.
    trim_exp = "A_stress" if (method_summary.experiment=="A_stress").any() else "A"
    T=method_summary[(method_summary.experiment==trim_exp)&(method_summary.scenario=="informative_H")]
    if len(T):
        base=T[T.method=="DR_CF"][["n","rho","param_rmse_mean"]].rename(columns={"param_rmse_mean":"base_rmse"})
        tt=T[T.method.str.startswith("DR_TAU",na=False)].merge(base,on=["n","rho"],how="left")
        if len(tt):
            tt["delta_rmse"]=tt.param_rmse_mean-tt.base_rmse
            tt.to_csv(os.path.join(OUTDIR,"trimming_delta_rmse.csv"),index=False)

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
print("="*100)
print("PR-ZIB SIMULATION STUDY")
print("="*100)
print(f"MODE={MODE}, N_JOBS={N_JOBS}")
print(f"Experiment A (strict positivity): reps={MC_REPS_A}, n={N_GRID_A}, rho={RHO_GRID_A}, eps={POSITIVITY_EPS}")
print(f"Optional stress test enabled: {RUN_STRESS_TEST}")
print(f"Experiment B enabled: {RUN_EXPERIMENT_B}; reps={MC_REPS_B}, n={N_GRID_B}, q={Q_GRID_B}, rho={RHO_GRID_B}")
print("Primary DR: no pseudo-gate clipping; no statistical propensity truncation.")

# Calibrate NHANES-like truths once.
calibration_rows=[]
truth_B={}
for q in Q_GRID_B:
    a,b,py1,qchk=calibrate_truth_for_py1_q(NHANES_PY1_TARGET,q,seed=BASE_SEED+int(q*10000))
    truth_B[q]=(a,b)
    calibration_rows.append(dict(q_target=q,py1_target=NHANES_PY1_TARGET,
                                 alpha0=a[0],beta0=b[0],py1_check=py1,q_check=qchk))
calib=pd.DataFrame(calibration_rows)
calib.to_csv(os.path.join(OUTDIR,"nhanes_like_calibration.csv"),index=False)
print("\nNHANES-like calibration:")
print(calib.round(5).to_string(index=False))

all_rep=[];all_coef=[]

if RUN_EXPERIMENT_A:
    print("\nRunning Experiment A: positivity-respecting main DGP ...")
    jobs=[(n,rep) for n in N_GRID_A for rep in range(MC_REPS_A)]
    ans=Parallel(n_jobs=N_JOBS,backend="loky",verbose=10)(
        delayed(run_one_A)(n,rep,"A",POSITIVITY_EPS,False) for n,rep in jobs)
    for rr,cc in ans:
        all_rep.extend(rr);all_coef.extend(cc)
    del ans

if RUN_STRESS_TEST:
    print("\nRunning A_stress: original logistic-normal practical-positivity stress test ...")
    jobs=[(n,rep) for n in N_GRID_A for rep in range(MC_REPS_A)]
    ans=Parallel(n_jobs=N_JOBS,backend="loky",verbose=10)(
        delayed(run_one_A)(n,rep,"A_stress",0.0,True) for n,rep in jobs)
    for rr,cc in ans:
        all_rep.extend(rr);all_coef.extend(cc)
    del ans

if RUN_EXPERIMENT_B:
    print("\nRunning Experiment B ...")
    jobs=[]
    for q in Q_GRID_B:
        a,b=truth_B[q]
        for n in N_GRID_B:
            for rep in range(MC_REPS_B):
                jobs.append((n,q,a,b,rep))
    ans=Parallel(n_jobs=N_JOBS,backend="loky",verbose=10)(
        delayed(run_one_B)(n,q,a,b,rep) for n,q,a,b,rep in jobs)
    for rr,cc in ans:
        all_rep.extend(rr);all_coef.extend(cc)
    del ans

rep_df=pd.DataFrame(all_rep)
coef_df=pd.DataFrame(all_coef)
method_summary=summarize_method(rep_df)
coef_summary=summarize_coefficients(coef_df)

rep_path=os.path.join(OUTDIR,"replicate_results.csv")
coef_path=os.path.join(OUTDIR,"coefficient_results.csv")
ms_path=os.path.join(OUTDIR,"method_summary.csv")
cs_path=os.path.join(OUTDIR,"coefficient_summary.csv")
rep_df.to_csv(rep_path,index=False)
coef_df.to_csv(coef_path,index=False)
method_summary.to_csv(ms_path,index=False)
coef_summary.to_csv(cs_path,index=False)

save_plots(method_summary)

# Key printed summaries
pd.set_option("display.max_rows",120)
pd.set_option("display.width",180)

print("\n"+"="*100)
print("EXPERIMENT A: KEY METHOD SUMMARY")
print("="*100)
colsA=["scenario","n","rho","method","n_success","failure_rate","param_rmse_mean","coverage_mean_mean",
       "pred_rmse_mean","n_rev_mean","n_rev1_mean","n_rev0_mean","ess_mean",
       "true_e_min_zero_mean","true_e_q01_zero_mean"]
Ashow=method_summary[(method_summary.experiment=="A") & method_summary.method.isin(["STD_ZIB","PR_MLE","DR_CF"])]
print(Ashow[[c for c in colsA if c in Ashow.columns]].round(4).to_string(index=False))

print("\n"+"="*100)
print("EXPERIMENT B: NHANES-LIKE HIGH-rho / RARE-POSITIVE SUMMARY")
print("="*100)
colsB=["n","q_target","rho","method","n_success","failure_rate","param_rmse_mean","coverage_mean_mean",
       "pred_rmse_mean","rho_realized_mean","q_realized_mean","n_rev1_mean","n_rev0_mean","ess_mean"]
Bshow=method_summary[(method_summary.experiment=="B") & method_summary.method.isin(["STD_ZIB","PR_MLE","DR_CF"])]
print(Bshow[[c for c in colsB if c in Bshow.columns]].round(4).to_string(index=False))

print("\nCoefficient-wise results are in coefficient_summary.csv")
print("Each coefficient has bias, bias_MCSE, RMSE, RMSE_MCSE, empirical_SD, mean_SE, coverage, coverage_MCSE.")

# Save config for reproducibility.
config=dict(MODE=MODE,N_JOBS=N_JOBS,BASE_SEED=BASE_SEED,N_SPLITS=N_SPLITS,
            MC_REPS_A=MC_REPS_A,N_GRID_A=N_GRID_A,RHO_GRID_A=RHO_GRID_A,
            MC_REPS_B=MC_REPS_B,N_GRID_B=N_GRID_B,Q_GRID_B=Q_GRID_B,RHO_GRID_B=RHO_GRID_B,
            POSITIVITY_EPS=POSITIVITY_EPS,RUN_STRESS_TEST=RUN_STRESS_TEST,
            TRIM_TAUS=TRIM_TAUS,primary_e_floor=MACHINE_E_FLOOR,
            pseudo_gate_clipped=False,nhanes_py1_target=NHANES_PY1_TARGET,
            reported_analysis=True)
with open(os.path.join(OUTDIR,"config.json"),"w") as f: json.dump(config,f,indent=2)

zip_base=os.path.join(os.path.dirname(OUTDIR) or ".", "pr_zib_simulation_results")
if os.path.exists(zip_base+".zip"): os.remove(zip_base+".zip")
shutil.make_archive(zip_base,"zip",OUTDIR)
print("\nSaved output directory:",OUTDIR)
print("Saved archive:",zip_base+".zip")
print("\nDone.")
