"""Unified Empirical-Bayes Feasibility- & OOD-Aware Constrained Acquisition Core for MALIBO.

Addresses the fundamental failure mode of neural likelihood-free meta-BO (MALIBO) under
high-dimensional black-box constraints: ELU ResFFN lacks distance-to-data decay and
extrapolates monotonically past constraint cliffs into out-of-distribution (OOD)
infeasible regions.

Strictly zero-leakage & zero extra NFE: uses ONLY the already-evaluated black-box history
{(x_i, c_i)}_{i=1}^N where c_i in {0, 1} is the binary feasibility flag returned alongside y_i.

Core unified formulation (100% identical across 2D D=18 and 3D D=63, zero task tuning):
  1. Empirical-Bayes Beta(1, 3) OOD Feasibility Prior:
         p0(t) = (k_post + 1) / (n_post + 4)
     Starts cautiously at p0=0.25 at Step 16 (protecting against early OOD hallucinations)
     and automatically adapts to each problem's true post-init feasibility rate.
  2. Matérn-5/2 Kernel Feasibility Posterior Mean:
         p_feas(x) = clip(p0(t) + k(x, X)^T (K_XX + lambda I)^{-1} (c - p0(t) * 1), eps, 1)
  3. Zero-Exploration-Bias Soft-Capped Constrained-LFBO Acquisition:
         p_tilde(x) = min(p_feas(x) / tau_f, 1.0)    (tau_f = 0.55)
         alpha(x)   = sigmoid((u_GB(x; z_hat) - mu_u) / sigma_u) * p_tilde(x)^beta
     - Above tau_f (predicted feasible): p_tilde(x) = 1.0 -> zero bias pulling back to old points!
     - Below tau_f (OOD corners or observed constraint cliffs): quadratic suppression!
  4. Smooth Adaptive LFBO Quantile Annealing:
         gamma(t): 1/3 -> 0.15 over post-init budget.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def matern52_kernel(x1: np.ndarray, x2: np.ndarray, lengthscale: float = 0.35) -> np.ndarray:
    """Isotropic Matérn-5/2 kernel in normalized unit hypercube [0, 1]^D."""
    dist = cdist(x1, x2, metric="euclidean") / float(lengthscale)
    sqrt5 = 2.23606797749979
    return (1.0 + sqrt5 * dist + (5.0 / 3.0) * (dist ** 2)) * np.exp(-sqrt5 * dist)


def empirical_bayes_feas_prior(
    c_obs: np.ndarray,
    n_init: int = 15,
    alpha_prior: float = 1.0,
    beta_prior: float = 3.0,
) -> float:
    """Empirical-Bayes Beta(1, 3) OOD feasibility prior from post-initialization evaluations.

    At n_post = 0 (Step 16), yields cautious prior 1 / (1 + 3) = 0.25.
    On high-feasibility problems (e.g. 2D airfoil), adapts smoothly upward above tau_f so
    exploration is never choked; on thin constrained manifolds (e.g. 3D 63D blade), stays
    cautious whenever post-init feasibility is low.
    """
    post = c_obs[int(n_init) :] if len(c_obs) > int(n_init) else np.empty(0, dtype=np.float64)
    k_post = float(np.sum(post > 0.5))
    n_post = float(len(post))
    return float((k_post + alpha_prior) / (n_post + alpha_prior + beta_prior))


def predict_pessimistic_feasibility(
    X_obs: np.ndarray,
    c_obs: np.ndarray,
    X_cand: np.ndarray,
    lengthscale: float = 0.35,
    prior_mean: float = 0.25,
    noise_var: float = 1e-3,
) -> np.ndarray:
    """Closed-form posterior mean of a Matérn-5/2 kernel feasibility surrogate."""
    n_obs = len(X_obs)
    if n_obs == 0:
        return np.full(len(X_cand), float(prior_mean), dtype=np.float64)

    K_xx = matern52_kernel(X_obs, X_obs, lengthscale=lengthscale)
    K_xx += float(noise_var) * np.eye(n_obs, dtype=np.float64)

    try:
        c_fac = cho_factor(K_xx, lower=True)
        alpha = cho_solve(c_fac, c_obs.astype(np.float64) - float(prior_mean))
    except np.linalg.LinAlgError:
        K_xx += 1e-2 * np.eye(n_obs, dtype=np.float64)
        c_fac = cho_factor(K_xx, lower=True)
        alpha = cho_solve(c_fac, c_obs.astype(np.float64) - float(prior_mean))

    K_star = matern52_kernel(X_cand, X_obs, lengthscale=lengthscale)
    p_feas = float(prior_mean) + K_star @ alpha
    return np.clip(p_feas, 1e-4, 1.0)


def compute_constrained_lfbo_acquisition(
    raw_utilities: np.ndarray,
    X_obs: np.ndarray,
    c_obs: np.ndarray,
    X_cand: np.ndarray,
    n_init: int = 15,
    lengthscale: float = 0.35,
    tau_feas: float = 0.55,
    beta: float = 2.0,
    prior_mean: float = None,
) -> np.ndarray:
    """Scale-invariant Soft-Capped Empirical-Bayes Constrained-LFBO acquisition.

    Candidates with p_feas(x) >= tau_feas receive full factor 1.0 (preserving unrestricted
    MALIBO ranking and eliminating any magnet toward previously evaluated points), whereas
    OOD or constraint-cliff candidates with p_feas(x) < tau_feas are suppressed smoothly
    by (p_feas(x) / tau_feas)^beta.
    """
    u = np.asarray(raw_utilities, dtype=np.float64).reshape(-1)
    u_std = float(np.std(u))
    if u_std < 1e-8:
        u_prob = np.full_like(u, 0.5)
    else:
        u_z = np.clip((u - float(np.mean(u))) / u_std, -10.0, 10.0)
        u_prob = 1.0 / (1.0 + np.exp(-u_z))

    p0 = (
        empirical_bayes_feas_prior(c_obs, n_init=n_init)
        if prior_mean is None
        else float(prior_mean)
    )
    p_feas = predict_pessimistic_feasibility(
        X_obs=X_obs,
        c_obs=c_obs,
        X_cand=X_cand,
        lengthscale=lengthscale,
        prior_mean=p0,
    )
    p_capped = np.minimum(p_feas / float(tau_feas), 1.0)
    return u_prob * (p_capped ** float(beta))


def annealed_lfbo_gamma(
    step: int,
    n_init: int,
    max_evals: int,
    gamma_start: float = 1.0 / 3.0,
    gamma_end: float = 0.15,
) -> float:
    """Linearly anneal LFBO quantile threshold gamma from 1/3 to 0.15 over post-init budget."""
    denom = max(int(max_evals) - int(n_init), 1)
    progress = float(np.clip((int(step) - int(n_init)) / denom, 0.0, 1.0))
    return float(gamma_start - (gamma_start - gamma_end) * progress)

