# run_hti_convergence.py
# -*- coding: utf-8 -*-

import numpy as np
from harmonic_operator import (
    HarmonicOperatorModel,
    tau_gauss_operational,
    prime_sum_truncated,
    fhat_gauss,
)

def run_A_hti_convergence(
    T: float = 20.0,
    eps: float = 0.15,
    bc: str = "periodic",
    Nt_list=(256, 512, 1024),
    N_list=(2_000, 10_000, 50_000),
    K_list=(2, 4, 6),
    n_probe_list=(32, 64),
    mp_dps: int = 80,
    n_gh: int = 24,
    cutoff_width: float = 0.5,
    T0_cutoff=None,
    seed: int = 0,
):
    """
    TEST (A): HTI numerical stability (C_hat(T) stability & convergence)

    Hypothèses mathématiques pertinentes:
      - H_ε auto-adjoint (vrai au niveau formel; ici discretisation) [[12]]
      - HTI (conjecture) [[11]]
      - Trace opérationnelle par fenêtre/heat kernel (ici proxy normalisé) [[12]]

    Sortie:
      - tableau de C_hat pour grilles de (Nt,N,K,n_probe)
      - critère simple: variation max-min de C_hat au raffinement
    """
    potential_cache = {}  # réutilisé entre constructions pour accélérer V_ε(t)

    fhat0 = float(fhat_gauss(0.0, T))

    results = []
    for Nt in Nt_list:
        model = HarmonicOperatorModel(
            T_window=T,
            Nt=int(Nt),
            eps=eps,
            T0_cutoff=T0_cutoff,
            cutoff_width=cutoff_width,
            n_gh=n_gh,
            bc=bc,
            mp_dps=mp_dps,
        )
        H, t_grid, dt, v = model.build(potential_cache=potential_cache)

        for n_probe in n_probe_list:
            tau = tau_gauss_operational(H, T_gauss=T, n_probe=int(n_probe), seed=seed, heat_factor=1.0)

            for N_max in N_list:
                for K_max in K_list:
                    P = prime_sum_truncated(T=T, N_max=int(N_max), K_max=int(K_max))
                    C_hat = (tau - P) / fhat0

                    results.append({
                        "T": T, "eps": eps, "bc": bc,
                        "Nt": int(Nt), "n_probe": int(n_probe),
                        "N_max": int(N_max), "K_max": int(K_max),
                        "tau": float(tau), "P": float(P),
                        "C_hat": float(C_hat),
                        "v_min": float(np.min(v)), "v_max": float(np.max(v)),
                    })

                    print(
                        f"Nt={Nt:5d} probes={n_probe:3d} N={N_max:7d} K={K_max:2d} | "
                        f"tau={tau:+.6e} P={P:+.6e}  C_hat={C_hat:+.6e}"
                    )

    # simple stability summary by refinement (group by Nt)
    by_Nt = {}
    for r in results:
        by_Nt.setdefault(r["Nt"], []).append(r["C_hat"])

    print("\n=== Stability summary (spread of C_hat by Nt) ===")
    for Nt, vals in sorted(by_Nt.items()):
        vals = np.array(vals, dtype=float)
        print(f"Nt={Nt:5d}: mean={vals.mean():+.6e}  spread(max-min)={(vals.max()-vals.min()):.6e}")

    return results


if __name__ == "__main__":
    run_A_hti_convergence(
        T=20.0,
        eps=0.15,
        bc="periodic",
        Nt_list=(256, 512),
        N_list=(2_000, 10_000),
        K_list=(2, 4),
        n_probe_list=(32, 64),
        mp_dps=80,
        n_gh=24,
        seed=1,
    )
