# Fail modes / Diagnostics (HTI non-circular protocol)

This document lists typical reasons why the non-circular HTI discrepancy
    Delta = S_T - fhat(0)*C - P_{N,K}
does not stabilize under refinement.

## 1) Potential V_eps issues (dominant)
- epsilon too small: mollified xi'/xi becomes numerically stiff near critical zeros.
- insufficient mp.dps precision -> noisy V(t) -> non-Hermitian artifacts / unstable expm_multiply.
- cutoff/window mismatch: V is localized but the window P_T clips it too hard.

Symptoms:
- v_min/v_max explode or fluctuate with Nt.
- Delta changes wildly when only epsilon is slightly adjusted.

## 2) Window P_T and boundary effects
- Too small padding (L_total close to T_window) leads to wrap-around artifacts (periodic BC).
- Window mask too sharp; consider smoother projection to reduce Gibbs-like effects.

Symptoms:
- Strong dependence on L_total.
- Delta changes when varying only L_total.

## 3) Trace estimator noise (Hutchinson)
- Too few probes -> tau_std is large; Delta looks unstable.

Fix:
- increase n_probe; use repeats with different seeds and report mean/std.

## 4) Calibration mismatch (free D model)
- If boundary conditions differ from the calibration model, the "density fix" is wrong.
- For non-periodic BC, you need a new calibration formula for D.

Symptoms:
- Delta drift with Nt even for V=0 sanity tests.

## 5) Prime-side truncation
- N_max or K_max too small; prime tail dominates.

Fix:
- increase N_max and K_max; compare P(N,K) vs P(2N,K) empirically.

## 6) Fundamental model limitation (toy truncation)
- The true framework lives on L^2(C_Q) with a norm-one fiber L^2(C_Q^1).
  Our toy model uses a placeholder fiber truncation J (block replication).
  Convergence in J may not reflect the true C_Q^1 harmonic analysis.

Symptoms:
- Delta stabilizes in Nt,N,K but is sensitive / meaningless in J.
