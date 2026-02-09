# experiments/hti_non_circular_sweep.py
# -*- coding: utf-8 -*-

import os, json, time
import numpy as np

from operational_extras import (
    WindowedOperatorBuilder,
    calibrated_windowed_tau_gauss,
    prime_sum_truncated_fast,
)

# ---- Fixed constant from HTI statement in the complete edition ----
C_CONST = float(np.log(2.0 * np.pi))  # C = log(2π) [[1 Conj. 6.2]]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

def build_fiber_lift(H_t, J=1, fiber_coupling=0.0, seed=0):
    """
    Fiber truncation parameter J as in the revised protocol [[2 §8.1]].
    Here: placeholder model (not the true C^1_Q characters). We replicate t-operator J times.

    Optionally add a small random Hermitian coupling between fiber blocks to test sensitivity.
    """
    from scipy import sparse

    J = int(J)
    if J <= 1:
        return H_t

    Ij = sparse.identity(J, format="csr", dtype=np.complex128)
    H = sparse.kron(Ij, H_t, format="csr")

    if fiber_coupling and fiber_coupling > 0:
        rng = np.random.default_rng(int(seed))
        A = rng.standard_normal((J, J))
        A = (A + A.T) / 2.0
        A = A / (np.linalg.norm(A) + 1e-12)
        A = sparse.csr_matrix(A.astype(np.float64))
        It = sparse.identity(H_t.shape[0], format="csr", dtype=np.complex128)
        H = (H + float(fiber_coupling) * sparse.kron(A, It, format="csr")).tocsr()

    return H

def lift_mask(mask_t, J=1):
    mask_t = np.asarray(mask_t, dtype=bool)
    if int(J) <= 1:
        return mask_t
    return np.tile(mask_t, int(J))

def run_sweep(
    out_dir="../results",
    tag="hti_sweep",
    # Operational trace params
    T_gauss_list=(10.0, 20.0),
    T_window=20.0,
    L_total=60.0,
    Nt_total_list=(2048, 4096),
    eps_list=(0.25, 0.15),
    J_list=(1, 4),
    n_probe_list=(32, 64),
    repeats=3,              # repeat with different seeds to estimate stochastic error
    heat_factor=1.0,        # as in operational definition with heat cutoff [[2 §5.2]]
    do_calibrate_on_D=True, # calibration on D suggested in revised edition [[2 §5.3]]
    # Prime side params
    N_max_list=(100_000, 1_000_000),
    K_max_list=(4, 10),
    # Potential build params
    mp_dps=80,
    n_gh=24,
    bc="periodic",
    cache_dir="../cache_potential",
    fiber_coupling=0.0,
):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{tag}.jsonl")

    run_meta = {
        "tag": tag,
        "created_unix": time.time(),
        "model": "toy-1D-t + window PT + heat cutoff + Hutchinson",
        "notes": "Non-circular HTI: spectral side computed from H_eps (no preloaded zeta zeros).",
    }
    # write meta header
    jsonl_append(out_path, {"_meta": run_meta})

    for Nt_total in Nt_total_list:
        for eps in eps_list:
            # Build H on [0,L_total) and explicit window mask on [0,T_window] [[2 §5.2]]
            builder = WindowedOperatorBuilder(
                T_window=T_window,
                L_total=L_total,
                Nt_total=int(Nt_total),
                eps=float(eps),
                cutoff_width=0.5,
                n_gh=int(n_gh),
                mp_dps=int(mp_dps),
                bc=bc,
                cache_dir=cache_dir,
                use_disk_cache=True,
            )
            t0 = time.time()
            H_t, t_grid, dt, v, mask_t = builder.build()
            build_sec = time.time() - t0

            for J in J_list:
                H = build_fiber_lift(H_t, J=J, fiber_coupling=fiber_coupling, seed=0)
                mask = lift_mask(mask_t, J=J)

                for T_gauss in T_gauss_list:
                    # fhat(0) for our Gaussian convention used elsewhere:
                    # fhat(0) = T * sqrt(pi)
                    fhat0 = float(T_gauss * np.sqrt(np.pi))

                    for N_max in N_max_list:
                        for K_max in K_max_list:
                            # prime-side truncated sum (fast sieve) [[1 §6.4–6.5]]
                            P = prime_sum_truncated_fast(T=float(T_gauss), N_max=int(N_max), K_max=int(K_max))

                            # repeated operational trace estimates -> mean/std
                            tau_vals = []
                            diag = None
                            for r in range(int(repeats)):
                                seed = 12345 + 1000 * r + int(Nt_total) + 17 * int(J)
                                trace_out = calibrated_windowed_tau_gauss(
                                    H, mask=mask,
                                    T_gauss=float(T_gauss),
                                    heat_factor=float(heat_factor),
                                    Nt_total=int(Nt_total) * int(J),
                                    dt=float(dt),
                                    n_probe=int(n_probe_list[min(r, len(n_probe_list)-1)]),
                                    seed=int(seed),
                                    do_calibrate_on_D=bool(do_calibrate_on_D),
                                )
                                tau_vals.append(trace_out["tau_cal"])
                                diag = trace_out  # keep last diagnostics

                            tau_mean = float(np.mean(tau_vals))
                            tau_std = float(np.std(tau_vals, ddof=1)) if len(tau_vals) > 1 else 0.0

                            # Non-circular discrepancy:
                            # Δ = S_T - fhat0*C - P_{N,K}(f_T)   [[1 Conj. 6.2]]
                            Delta = float(tau_mean - fhat0 * C_CONST - P)

                            record = {
                                "Nt_total": int(Nt_total),
                                "J": int(J),
                                "eps": float(eps),
                                "T_gauss": float(T_gauss),
                                "T_window": float(T_window),
                                "L_total": float(L_total),
                                "N_max": int(N_max),
                                "K_max": int(K_max),
                                "heat_factor": float(heat_factor),
                                "do_calibrate_on_D": bool(do_calibrate_on_D),
                                "prime_sum_P": float(P),
                                "fhat0": float(fhat0),
                                "C_const": float(C_CONST),
                                "tau_mean": float(tau_mean),
                                "tau_std": float(tau_std),
                                "Delta": float(Delta),
                                "beta": float(diag["beta"]),
                                "cal": float(diag["cal"]),
                                "tau_raw": float(diag["tau_raw"]),
                                "dt": float(dt),
                                "v_min": float(np.min(v)),
                                "v_max": float(np.max(v)),
                                "build_sec": float(build_sec),
                            }
                            print(record)
                            jsonl_append(out_path, record)

    print(f"\nDone. Results written to: {out_path}")

if __name__ == "__main__":
    run_sweep()
