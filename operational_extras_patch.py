from operational_extras import (
    WindowedOperatorBuilder,
    calibrated_windowed_tau_gauss,
    prime_sum_truncated_fast,
    fhat_gauss_vec,
)

# paramètres cibles
T = 20.0
T_window = 20.0
L_total = 60.0          # padding important pour que P_T soit "dans" le domaine
Nt_total = 8192
eps = 0.15

builder = WindowedOperatorBuilder(
    T_window=T_window,
    L_total=L_total,
    Nt_total=Nt_total,
    eps=eps,
    bc="periodic",
    mp_dps=80,
    n_gh=24,
    cache_dir="./cache_potential",
    use_disk_cache=True,
)

H, t_grid, dt, v, mask = builder.build()

trace_out = calibrated_windowed_tau_gauss(
    H, mask=mask,
    T_gauss=T,
    heat_factor=1.0,
    Nt_total=Nt_total,
    dt=dt,
    n_probe=64,
    seed=1,
    do_calibrate_on_D=True,
)

tau = trace_out["tau_cal"]       # trace calibrée sur D
P = prime_sum_truncated_fast(T=T, N_max=10**6, K_max=10)

fhat0 = float(T * np.sqrt(np.pi))  # fhat_gauss(0)
C_hat = (tau - P) / fhat0
print("C_hat =", C_hat, trace_out)
