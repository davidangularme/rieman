# operational_extras.py
# -*- coding: utf-8 -*-

"""
Operational extras for Harmonic Ontology numerical tests

WHAT THIS FILE ADDS (requested)
-------------------------------
(1) Explicit window P_T (projection) in the operational trace, not "implicit periodic window".
    - We implement P_T as a boolean mask on a *larger* discretized interval [0, L_total),
      and we trace P_T A P_T with heat-kernel damping, matching the "operational trace" spirit.

(2) Calibration on the free generator D (V=0) to fix spectral density:
    - We compute a calibration factor so that, for A = exp(-beta D^2),
      the discrete normalized trace matches the continuum model integral:
         τ_cont(exp(-beta D^2)) = ∫ exp(-beta λ^2) dλ/(2π) = (1/(2π)) * sqrt(pi/beta).
      This reduces spurious constant drifts due to discretization.

(3) Disk cache (.npz) for v_eps(t):
    - Computing v_eps(t) is the bottleneck (mpmath evaluations of xi'/xi).
      We store v(t) and the grid to disk keyed by parameter hash.

MATHEMATICAL STATUS / HYPOTHESES (from the PDFs)
------------------------------------------------
Unconditional (formal/operator level in PDFs):
  - H_ε = D + V_ε with V_ε bounded (via mollification) is self-adjoint by bounded perturbation.
  - D is the Stone generator; in t-coordinate D ~ -i d/dt.
Operational definitions (for computation):
  - τ_T(A) defined via window P_T in t and heat-kernel damping e^{-H^2/T^2}.
Conjectural:
  - HTI: τ(f(H_ε)) equals prime side + constant + remainder (continuous/trivial zeros).
  - Spectral atomicity and spectral identification (Spec_pp(H) equals zeta zeros ordinates).

NUMERICAL/IMPLEMENTATION HYPOTHESES (extra)
-------------------------------------------
  - 1D toy model in t only (fiber L^2(C_Q^1) truncated away).
  - Discretization uses periodic finite differences (global) on [0, L_total).
  - Window P_T is explicit as a mask selecting indices with t in [0, T_window].
  - Trace estimated via Hutchinson + expm_multiply on exp(-beta H^2).
  - Calibration uses the same discretization of D as used in H to reduce mismatch.

NOTE
----
This supports "test (A)" (C_hat stability) and improves its numerical robustness.
For "test (B)" (spectral identification), calibration isn't essential; windowing may still help.
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

# we reuse your existing construction/model
from harmonic_operator import HarmonicOperatorModel, build_D_periodic


# -------------------------
#   Disk cache for v_eps(t)
# -------------------------
def _hash_params(params: Dict[str, Any]) -> str:
    s = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def potential_cache_path(
    cache_dir: str,
    params: Dict[str, Any],
    prefix: str = "v_eps_cache",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = _hash_params(params)
    return os.path.join(cache_dir, f"{prefix}_{h}.npz")


def save_potential_npz(path: str, t_grid: np.ndarray, v: np.ndarray, params: Dict[str, Any]) -> None:
    np.savez_compressed(path, t_grid=t_grid.astype(np.float64), v=v.astype(np.float64),
                        params_json=np.array(json.dumps(params, sort_keys=True)))


def load_potential_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    t_grid = data["t_grid"].astype(np.float64)
    v = data["v"].astype(np.float64)
    params_json = str(data["params_json"])
    params = json.loads(params_json)
    return t_grid, v, params


# -------------------------
#   Explicit window P_T
# -------------------------
def window_mask(t_grid: np.ndarray, T_window: float) -> np.ndarray:
    """Boolean mask for P_T: keep t in [0, T_window]."""
    return (t_grid >= 0.0) & (t_grid <= float(T_window))


def apply_mask_vec(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(z)
    out[mask] = z[mask]
    return out


# -------------------------
#   Calibration on D (V=0)
# -------------------------
def tau_continuum_free_gauss(beta: float) -> float:
    """
    Continuum model:
      τ_cont(exp(-beta D^2)) = ∫ exp(-beta λ^2) dλ/(2π) = (1/(2π))*sqrt(pi/beta).
    """
    beta = float(beta)
    return (1.0 / (2.0 * np.pi)) * np.sqrt(np.pi / beta)


def tau_discrete_free_D_fd_periodic(beta: float, Nt_total: int, dt: float) -> float:
    """
    Discrete normalized trace for exp(-beta D^2) where D is the periodic FD operator.

    For the periodic central difference derivative:
      d/dt eigenvalues are i*(sin(k)/dt) with k=2π n/N
      => D=-i d/dt eigenvalues are (sin(k)/dt), real.
    """
    N = int(Nt_total)
    dt = float(dt)
    n = np.arange(N, dtype=np.float64)
    k = 2.0 * np.pi * n / N
    lam = np.sin(k) / dt
    return float(np.mean(np.exp(-float(beta) * lam * lam)))


def calibration_factor_on_D(beta: float, Nt_total: int, dt: float) -> float:
    """
    cal = τ_cont / τ_discrete_free
    Multiply discrete operational traces by cal to align density to continuum model.
    """
    tau_cont = tau_continuum_free_gauss(beta)
    tau_disc = tau_discrete_free_D_fd_periodic(beta, Nt_total=Nt_total, dt=dt)
    if tau_disc == 0.0:
        raise ZeroDivisionError("tau_disc == 0, beta too large or dt too small")
    return tau_cont / tau_disc


# -------------------------
#   Windowed operational trace estimator
# -------------------------
def tau_windowed_exp_neg_beta_H2(
    H: sparse.csr_matrix,
    beta: float,
    mask: np.ndarray,
    n_probe: int = 64,
    seed: int = 0,
    use_masked_probes: bool = True,
) -> float:
    """
    Estimate normalized trace:
      τ ≈ (1/|mask|) Tr( P exp(-beta H^2) P )
    via Hutchinson:
      Tr(P A P) = E[ (Pz)^T A (Pz) ] for Rademacher z.
    If use_masked_probes=True, we draw z on full grid then zero it outside mask.

    Returns: normalized by window size |mask|.
    """
    beta = float(beta)
    N = H.shape[0]
    mask = np.asarray(mask, dtype=bool)
    Nw = int(mask.sum())
    if Nw <= 0:
        raise ValueError("Empty window mask")

    H2 = (H @ H).tocsr()
    A = (-beta) * H2

    rng = np.random.default_rng(int(seed))
    acc = 0.0
    for _ in range(int(n_probe)):
        z = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        if use_masked_probes:
            z = apply_mask_vec(z, mask)  # z := P z
        y = expm_multiply(A, z)          # y = exp(-beta H^2) z
        acc += float(np.dot(z, y))       # z^T exp(.) z = (Pz)^T exp(.) (Pz)
    tr_est = acc / float(n_probe)

    return tr_est / float(Nw)


@dataclass
class WindowedOperatorBuilder:
    """
    Build H on a larger interval [0, L_total) and provide explicit window P_T mask on [0, T_window].

    Parameters:
      - T_window: window length for P_T
      - L_total: total simulation length (>= T_window). Choose L_total = T_window + padding.
      - Nt_total: grid size on [0, L_total)
    """
    T_window: float
    L_total: float
    Nt_total: int
    eps: float

    cutoff_width: float = 0.5
    n_gh: int = 24
    mp_dps: int = 80
    bc: str = "periodic"

    cache_dir: str = "./cache_potential"
    use_disk_cache: bool = True

    def build(self):
        """
        Returns:
          H (sparse), t_grid_total, dt, v_total, mask_window
        """
        if self.L_total < self.T_window:
            raise ValueError("L_total must be >= T_window")
        Nt = int(self.Nt_total)
        L = float(self.L_total)
        dt = L / Nt
        t_grid = np.linspace(0.0, L, Nt, endpoint=False)
        mask = window_mask(t_grid, self.T_window)

        # We build H via HarmonicOperatorModel by setting T_window = L_total
        # and cutoff T0 approximately T_window (so potential is localized to the window).
        model = HarmonicOperatorModel(
            T_window=L,
            Nt=Nt,
            eps=float(self.eps),
            T0_cutoff=float(self.T_window),   # cutoff near the window edge
            cutoff_width=float(self.cutoff_width),
            n_gh=int(self.n_gh),
            bc=self.bc,
            mp_dps=int(self.mp_dps),
        )

        # Disk cache on v(t)
        params = {
            "T_window": self.T_window,
            "L_total": self.L_total,
            "Nt_total": self.Nt_total,
            "eps": float(self.eps),
            "cutoff_width": float(self.cutoff_width),
            "n_gh": int(self.n_gh),
            "mp_dps": int(self.mp_dps),
            "bc": self.bc,
            "T0_cutoff": float(self.T_window),
        }

        if self.use_disk_cache:
            path = potential_cache_path(self.cache_dir, params)
            if os.path.exists(path):
                t0, v0, p0 = load_potential_npz(path)
                # sanity checks
                if len(t0) == len(t_grid) and np.allclose(t0, t_grid, atol=0.0, rtol=0.0):
                    # reuse cached v by reconstructing H = D + diag(v)
                    if self.bc != "periodic":
                        # For simplicity, calibration+windowing assume periodic D.
                        # If you really want dirichlet, you can still use it, but
                        # then D eigenvalues formula (calibration) must be updated.
                        pass
                    D = build_D_periodic(Nt, dt)
                    V = sparse.diags(v0.astype(np.float64), 0, format="csr")
                    H = (D + V).tocsr()
                    return H, t_grid, dt, v0, mask

        # compute fresh
        potential_cache_mem: Dict[Tuple[float, float], complex] = {}
        H, t_grid2, dt2, v = model.build(potential_cache=potential_cache_mem)
        # t_grid2 should match t_grid (both from [0,L))
        if self.use_disk_cache:
            path = potential_cache_path(self.cache_dir, params)
            save_potential_npz(path, t_grid2, v, params)

        return H, t_grid2, dt2, v, mask


def calibrated_windowed_tau_gauss(
    H: sparse.csr_matrix,
    mask: np.ndarray,
    T_gauss: float,
    heat_factor: float,
    Nt_total: int,
    dt: float,
    n_probe: int = 64,
    seed: int = 0,
    do_calibrate_on_D: bool = True,
) -> Dict[str, float]:
    """
    Compute windowed operational tau for Gaussian test combined with heat-kernel:
      f_T(λ)=exp(-λ^2/T^2)
      damping exp(-heat_factor * λ^2/T^2)
      => exp(-beta λ^2) with beta = (1+heat_factor)/T^2.

    Then:
      tau_raw = (1/|mask|) Tr( P exp(-beta H^2) P )  (Hutchinson)
      cal = tau_cont(exp(-beta D^2)) / tau_disc_free(exp(-beta D^2))
      tau_cal = cal * tau_raw
    """
    T = float(T_gauss)
    beta = (1.0 + float(heat_factor)) / (T * T)

    tau_raw = tau_windowed_exp_neg_beta_H2(H, beta=beta, mask=mask, n_probe=n_probe, seed=seed)

    cal = 1.0
    if do_calibrate_on_D:
        cal = calibration_factor_on_D(beta=beta, Nt_total=int(Nt_total), dt=float(dt))

    return {
        "beta": float(beta),
        "tau_raw": float(tau_raw),
        "cal": float(cal),
        "tau_cal": float(cal * tau_raw),
    }


# -------------------------
#   Fast prime sum up to N (segmented sieve)
# -------------------------
def primes_up_to_segmented(n: int, segment_size: int = 1_000_000) -> np.ndarray:
    """
    Generate primes <= n using a simple segmented sieve (numpy).
    Good enough for n ~ 1e6..1e7 without external deps.
    """
    n = int(n)
    if n < 2:
        return np.array([], dtype=np.int64)

    limit = int(np.sqrt(n)) + 1
    # base sieve up to sqrt(n)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[p]:
            is_prime[p*p:limit+1:p] = False
    base_primes = np.flatnonzero(is_prime).astype(np.int64)

    primes = []
    low = 2
    while low <= n:
        high = min(low + segment_size - 1, n)
        seg = np.ones(high - low + 1, dtype=bool)
        for p in base_primes:
            start = max(p*p, ((low + p - 1) // p) * p)
            if start > high:
                continue
            seg[start - low: high - low + 1: p] = False
        primes.append((np.flatnonzero(seg) + low).astype(np.int64))
        low = high + 1
    return np.concatenate(primes)


def fhat_gauss_vec(xi: np.ndarray, T: float) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    T = float(T)
    return T * np.sqrt(np.pi) * np.exp(-0.25 * (T * xi) ** 2)


def prime_sum_truncated_fast(T: float, N_max: int, K_max: int) -> float:
    """
    Vectorized prime side:
      Σ_{p<=N} Σ_{k<=K} (log p)/p^{k/2} * fhat(k log p)

    Designed for N up to 1e6 and K up to ~10 comfortably.
    """
    T = float(T)
    N_max = int(N_max)
    K_max = int(K_max)

    primes = primes_up_to_segmented(N_max)
    if primes.size == 0:
        return 0.0

    lp = np.log(primes.astype(np.float64))
    ks = np.arange(1, K_max + 1, dtype=np.float64)

    # weights: log p * exp(-(k/2) log p) = log p * p^{-k/2}
    # and fhat depends on k log p
    xi = lp[:, None] * ks[None, :]
    weights = lp[:, None] * np.exp(-0.5 * xi)
    return float(np.sum(weights * fhat_gauss_vec(xi, T)))
