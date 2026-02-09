# harmonic_operator.py
# -*- coding: utf-8 -*-

"""
Harmonic Ontology - Numerical toy implementation

HYPOTHESES / STATUT (d'après les PDFs)
-------------------------------------
(1) Construction de H_ε = D + V_ε sur H = L^2(C_Q) (réduit ici à un modèle 1D en t=log|x|)
    - D est le générateur de Stone de l'action de translation en t (donc ~ -i d/dt) [[12]].
    - V_ε est un potentiel multiplicatif réel borné construit à partir d'une mollification de (xi'/xi)(1/2+it)
      et d'un cutoff compact (ou numériquement un cutoff lisse) [[12]].
    - H_ε est auto-adjoint par perturbation bornée (Kato–Rellich) (INCONDITIONNEL) [[12]].

(2) Trace opérationnelle (NUMÉRIQUE)
    Les PDFs révisés définissent une trace réguliarisée "opérationnelle" via fenêtre P_T en t
    et amortissement heat-kernel e^{-H^2/T^2} [[12]].
    Ici on approxime τ_T(A) par une trace normalisée discrète (sur la grille), avec un estimateur stochastique.

(3) HTI (CONJECTURE)
    τ(f(H_ε)) = \hat f(0) C + sum_{p,k} (log p)/p^{k/2} \hat f(k log p) + R(f) [[11]].
    C et R(f) dépendent de conventions et du terme "continu/triviaux" [[11]].
    => Dans le test (A) on regarde surtout la stabilité d'une "constante effective" C_hat(T)
       quand on raffine (N,K,Nt, #probes), ce qui est un critère numérique utile mais pas un théorème.

(4) Spectral Atomicity + Spectral Identification (HYPOTHÈSES)
    - Atomicité : la partie discrète du spectre est purement atomique (multiplicités finies) [[11]].
    - Identification : Spec_pp(H) = { ordinées γ : ζ(1/2+iγ)=0 } (multiplicités) [[11]].
    => Le test (B) compare simplement des valeurs propres numériques d'un modèle discret à des zéros connus.
       Un mismatch ne falsifie que le modèle numérique / discrétisation, pas forcément l'énoncé abstrait.

HYPOTHESES NUMERIQUES ADDITIONNELLES (inévitables)
--------------------------------------------------
(A) Modélisation 1D : on ignore la fibre L^2(C_Q^1) (ou on la tronque à un facteur de dimension 1).
(B) Discrétisation : D est discrétisé sur [0,T] avec conditions périodiques (par défaut).
(C) Mollificateur : on utilise une mollification gaussienne (via Gauss-Hermite), pas un bump compact exact.
(D) Trace : on estime Tr(exp(-β H^2)) via Hutchinson + expm_multiply (stochastique).
"""

from __future__ import annotations

import numpy as np
import mpmath as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from sympy import primerange
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, eigsh


# -----------------------
#   High precision setup
# -----------------------
def set_mp_dps(dps: int = 80) -> None:
    mp.mp.dps = int(dps)


# -----------------------
#   xi'/xi
# -----------------------
def xi_log_derivative(s: mp.mpc) -> mp.mpc:
    """Compute xi'(s)/xi(s) using mpmath."""
    # Robust definition: d/ds log xi(s)
    # (avoid accidental extra division)
    return mp.diff(lambda z: mp.log(mp.xi(z)), s)


# -----------------------
#   Smooth cutoff phi(t)
# -----------------------
def smooth_cutoff(t: np.ndarray, T0: float, width: float = 0.5) -> np.ndarray:
    """
    Smooth step: ~1 for t << T0, ~0 for t >> T0.
    Numériquement stable; pas strictement compact.
    """
    t = np.asarray(t, dtype=np.float64)
    return 0.5 * (1.0 - np.tanh((t - T0) / width))


# -----------------------
#   Mollified potential
# -----------------------
def mollified_potential_grid(
    t_grid: np.ndarray,
    eps: float,
    T0: float,
    cutoff_width: float = 0.5,
    n_gh: int = 24,
    cache: Optional[Dict[Tuple[float, float], complex]] = None,
) -> np.ndarray:
    """
    v_eps(t) = -Re( g_eps(t) ) * phi(t)
    g_eps = mollification of g(t) = (xi'/xi)(1/2 + i t) on the critical line [[12]].

    Implementation detail: Gaussian mollifier using Gauss-Hermite quadrature:
        g_eps(t) ≈ (1/sqrt(pi)) * Σ w_j * g(t - eps*x_j),
    where (x_j, w_j) integrate ∫ e^{-x^2} f(x) dx.

    cache: optional dict to memoize xi_log_derivative evaluations at (t, eps-shift).
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    phi = smooth_cutoff(t_grid, T0=T0, width=cutoff_width)

    x, w = np.polynomial.hermite.hermgauss(n_gh)  # ∫ e^{-x^2} f(x) dx
    x = x.astype(np.float64)
    w = w.astype(np.float64)

    v = np.zeros_like(t_grid, dtype=np.float64)
    if cache is None:
        cache = {}

    for i, t in enumerate(t_grid):
        acc = mp.mpc(0)
        for xj, wj in zip(x, w):
            tj = float(t - eps * xj)
            key = (tj, eps)
            if key in cache:
                g = cache[key]
            else:
                s = mp.mpf("0.5") + 1j * mp.mpf(str(tj))
                g = complex(xi_log_derivative(s))
                cache[key] = g
            acc += mp.mpf(str(wj)) * (mp.mpc(g.real, g.imag))

        g_eps = acc / mp.sqrt(mp.pi)
        v[i] = -float(mp.re(g_eps)) * float(phi[i])

    return v


# -----------------------
#   Discretize D and H
# -----------------------
def build_D_periodic(Nt: int, dt: float) -> sparse.csr_matrix:
    """
    Central difference periodic derivative: d/dt.
    D = -i d/dt should be Hermitian on periodic grid.
    """
    Nt = int(Nt)
    dt = float(dt)

    Dd = sparse.lil_matrix((Nt, Nt), dtype=np.complex128)
    for i in range(Nt):
        ip = (i + 1) % Nt
        im = (i - 1) % Nt
        Dd[i, ip] = 1.0
        Dd[i, im] = -1.0
    Dd = (Dd.tocsr()) / (2.0 * dt)

    D = (-1j) * Dd
    return D.tocsr()


def build_D_dirichlet(Nt: int, dt: float) -> sparse.csr_matrix:
    """
    Dirichlet-like central differences (no wrap).
    Not perfectly skew-adjoint at boundary; we symmetrize to enforce Hermiticity of D=-i d/dt.
    """
    Nt = int(Nt)
    dt = float(dt)

    Dd = sparse.lil_matrix((Nt, Nt), dtype=np.complex128)
    for i in range(1, Nt - 1):
        Dd[i, i + 1] = 1.0
        Dd[i, i - 1] = -1.0
    Dd = (Dd.tocsr()) / (2.0 * dt)

    D = (-1j) * Dd
    # enforce Hermitian part (numerical convenience)
    D = 0.5 * (D + D.getH())
    return D.tocsr()


@dataclass
class HarmonicOperatorModel:
    T_window: float
    Nt: int
    eps: float
    T0_cutoff: Optional[float] = None
    cutoff_width: float = 0.5
    n_gh: int = 24
    bc: str = "periodic"  # "periodic" or "dirichlet"
    mp_dps: int = 80

    def build(self, potential_cache: Optional[Dict[Tuple[float, float], complex]] = None):
        """
        Build sparse H = D + V_eps on grid t in [0,T_window).

        Returns: (H, t_grid, dt, v_diag)
        """
        set_mp_dps(self.mp_dps)

        T = float(self.T_window)
        Nt = int(self.Nt)
        dt = T / Nt
        t_grid = np.linspace(0.0, T, Nt, endpoint=False)

        if self.T0_cutoff is None:
            T0 = T
        else:
            T0 = float(self.T0_cutoff)

        if self.bc == "periodic":
            D = build_D_periodic(Nt, dt)
        elif self.bc == "dirichlet":
            D = build_D_dirichlet(Nt, dt)
        else:
            raise ValueError("bc must be 'periodic' or 'dirichlet'")

        v = mollified_potential_grid(
            t_grid=t_grid,
            eps=float(self.eps),
            T0=T0,
            cutoff_width=float(self.cutoff_width),
            n_gh=int(self.n_gh),
            cache=potential_cache,
        )
        V = sparse.diags(v.astype(np.float64), 0, format="csr")

        H = (D + V).tocsr()

        # Hermiticity check
        herm_err = sparse.linalg.norm(H - H.getH())
        if herm_err > 1e-8:
            print(f"[warn] ||H - H*|| = {herm_err:g} (discretization may be inconsistent)")

        return H, t_grid, dt, v


# -----------------------
#   Gaussian test + trace
# -----------------------
def fhat_gauss(xi: np.ndarray | float, T: float) -> np.ndarray | float:
    """
    Fourier transform convention used in the original pseudo-code:
    \hat f_T(ξ) = T * sqrt(pi) * exp(-(T ξ)^2/4)
    (Convention mismatch changes constants; test (A) focuses on stability not absolute value.)
    """
    xi_arr = np.asarray(xi, dtype=np.float64)
    return T * np.sqrt(np.pi) * np.exp(-0.25 * (T * xi_arr) ** 2)


def prime_sum_truncated(T: float, N_max: int, K_max: int) -> float:
    """
    P_{N,K}(f_T) = Σ_{p<=N} Σ_{k<=K} (log p)/p^{k/2} * \hat f_T(k log p) [[11]].
    """
    primes = np.fromiter(primerange(2, int(N_max) + 1), dtype=np.int64)
    lp = np.log(primes.astype(np.float64))
    ks = np.arange(1, int(K_max) + 1, dtype=np.float64)

    xi = lp[:, None] * ks[None, :]
    weights = (lp[:, None] / (primes[:, None].astype(np.float64) ** (ks[None, :] / 2.0)))
    return float(np.sum(weights * fhat_gauss(xi, float(T))))


def tau_gauss_operational(
    H: sparse.csr_matrix,
    T_gauss: float,
    n_probe: int = 64,
    seed: int = 0,
    heat_factor: float = 1.0,
) -> float:
    """
    Operational trace proxy:
      τ_T(f_T(H)) ~ (1/Nt) Tr( f_T(H) e^{-heat_factor * H^2/T^2} )
    For f_T(λ)=exp(-λ^2/T^2), product becomes exp(-(1+heat_factor) H^2/T^2).

    Implementation: Hutchinson estimator of Tr(exp(-beta H^2)) using expm_multiply.

    NOTE: This is a numerical proxy for the windowed/heat-damped operational trace [[12]].
    """
    T = float(T_gauss)
    N = H.shape[0]
    beta = (1.0 + float(heat_factor)) / (T ** 2)

    H2 = (H @ H).tocsr()
    A = (-beta) * H2

    rng = np.random.default_rng(int(seed))
    acc = 0.0
    for _ in range(int(n_probe)):
        z = rng.choice([-1.0, 1.0], size=N).astype(np.float64)
        y = expm_multiply(A, z)  # exp(A) z
        acc += float(np.dot(z, y))
    tr_est = acc / float(n_probe)
    return tr_est / float(N)  # normalized


# -----------------------
#   Spectral extraction
# -----------------------
def eigvals_near(
    H: sparse.csr_matrix,
    sigma: float,
    k: int = 40,
    which: str = "LM",
) -> np.ndarray:
    """
    Extract k eigenvalues near sigma using shift-invert (eigsh with sigma).
    For Hermitian matrices, this returns eigenvalues close to sigma.
    """
    vals = eigsh(H, k=int(k), sigma=float(sigma), which=which, return_eigenvectors=False)
    vals = np.real_if_close(vals, tol=1e-9).astype(np.float64)
    return np.sort(vals)
