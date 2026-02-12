#!/usr/bin/env python3
"""
Jost-Zeta FINAL: reproduire le succes de v1 avec plus de points.
Cle: dt doit satisfaire h^2 * max(k^2 + |v|) / 12 < 1 (stabilite Numerov)
"""
import numpy as np
from mpmath import mp, mpf, mpc, zetazero, zeta, gamma as mgamma, pi as mpi
mp.dps = 15

N_Z = 5
zeta_g = [float(zetazero(n).imag) for n in range(1, N_Z+1)]
print("Zeros zeta:", [f"{g:.4f}" for g in zeta_g])

def xi_line(t):
    s = mpf('0.5') + mpc('0','1')*mpf(str(t))
    return float((mpf('0.5')*s*(s-1)*mpi**(-s/2)*mgamma(s/2)*zeta(s)).real)

def g_func(t, h=1e-5):
    xi = xi_line(t)
    if abs(xi) < 1e-20: return 0.0
    return -(xi_line(t+h) - xi_line(t-h)) / (2*h*xi)

# T_MAX=30 marchait pour 3 zeros. Essayons T_MAX=30 avec N=800 (dt plus petit)
# pour que la resolution supporte k jusqu'a 35.
configs = [
    ("Config A: T=30 N=400 (v1 original)", 30.0, 400, 0.35),
    ("Config B: T=30 N=800 (dt plus fin)",  30.0, 800, 0.35),
    ("Config C: T=35 N=1000 (etendu)",      35.0, 1000, 0.40),
]

# Pre-calc g
print("Pre-calcul g(t)...")
T_G = 40.0; N_G = 900
tgg = np.linspace(-T_G, T_G, N_G)
g_all = np.array([g_func(float(t)) for t in tgg])
g_all = np.clip(g_all, -80, 80)
print("Done.\n")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(configs), 2, figsize=(15, 4*len(configs)))

for ci, (name, T_MAX, N, EPS) in enumerate(configs):
    dt = 2*T_MAX/(N-1)
    tg = np.linspace(-T_MAX, T_MAX, N)
    
    # Stabilite Numerov: h^2 * (k_max^2 + max|v|) / 12 < 1
    # => k_max < sqrt(12/h^2 - max|v|)
    
    g_interp = np.interp(tg, tgg, g_all)
    kh = int(4*EPS/dt)+1
    kt = np.arange(-kh, kh+1)*dt
    kern = np.exp(-kt**2/(2*EPS**2)) / (EPS*np.sqrt(2*np.pi)) * dt
    g_e = np.convolve(g_interp, kern, mode='same')
    gp = np.gradient(g_e, dt)
    v = g_e**2 + gp
    maxv = np.max(np.abs(v))
    
    k_max_stable = np.sqrt(max(0, 12/dt**2 - maxv))
    print(f"{name}")
    print(f"  dt={dt:.5f}, max|v|={maxv:.1f}, k_max_stable~{k_max_stable:.1f}")
    
    # Scan
    k_hi = min(36.0, k_max_stable * 0.9)
    ks = np.linspace(5.0, k_hi, 600)
    Fk = np.zeros(len(ks))
    
    for i, k in enumerate(ks):
        q = k**2 - v; h2 = dt**2
        psi = np.zeros(N, dtype=complex)
        psi[-1] = np.exp(1j*k*tg[-1])
        psi[-2] = np.exp(1j*k*tg[-2])
        ok = True
        for n in range(N-3, -1, -1):
            w2 = 1+h2*q[n+2]/12; w1 = 1-5*h2*q[n+1]/12; w0 = 1+h2*q[n]/12
            psi[n] = (2*w1*psi[n+1] - w2*psi[n+2]) / w0
            if not np.isfinite(psi[n]): ok = False; break
        if not ok: Fk[i] = float('nan'); continue
        i1, i2 = 10, 30
        e1p = np.exp(1j*k*tg[i1]); e1m = np.exp(-1j*k*tg[i1])
        e2p = np.exp(1j*k*tg[i2]); e2m = np.exp(-1j*k*tg[i2])
        det = e1p*e2m - e1m*e2p
        A = (psi[i2]*e1m - psi[i1]*e2m) / det
        Fk[i] = abs(A)
    
    valid = np.isfinite(Fk) & (Fk > 0)
    ks_v, Fk_v = ks[valid], Fk[valid]
    Fn = Fk_v / np.max(Fk_v)
    
    # Minima
    mins = []
    for i in range(1, len(Fn)-1):
        if Fn[i]<Fn[i-1] and Fn[i]<Fn[i+1] and Fn[i]<0.7:
            mins.append((ks_v[i], Fn[i]))
    mins.sort(key=lambda x: x[1])
    
    print(f"  Plage k: [{ks_v[0]:.1f}, {ks_v[-1]:.1f}], {len(mins)} minima")
    print(f"  {'n':>3} {'gamma':>10} {'k_Jost':>10} {'Dk':>8}")
    for n, gn in enumerate(zeta_g):
        if gn > ks_v[-1]: 
            print(f"  {n+1:3d} {gn:10.4f}   hors plage")
            continue
        if mins:
            b = min(mins[:20], key=lambda x: abs(x[0]-gn))
            d = abs(b[0]-gn)
            tag = " MATCH" if d<0.3 else (" ~" if d<1.0 else "")
            print(f"  {n+1:3d} {gn:10.4f} {b[0]:10.4f} {d:8.4f}{tag}")
    print()
    
    # Plot
    ax = axes[ci, 0]
    ax.plot(tg, v, 'b-', lw=0.6)
    for g in zeta_g: 
        if g < T_MAX: ax.axvline(g, c='r', alpha=0.3, ls='--')
    ax.set_title(f'{name}\nmax|v|={maxv:.1f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[ci, 1]
    if len(ks_v) > 0:
        ax.plot(ks_v, np.log(Fn+1e-300), 'b-', lw=1.0)
        for i,g in enumerate(zeta_g):
            if g < ks_v[-1]:
                ax.axvline(g, c='r', alpha=0.5, ls='--', label=f'γ_{i+1}' if i<5 else '')
        for km, fm in mins[:10]:
            ax.plot(km, np.log(fm+1e-300), 'gv', ms=7)
    ax.set_title(f'log|F/Fmax|, k_max_stable={k_max_stable:.0f}', fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

plt.suptitle('Conjecture Jost-Zeta — Etude de stabilite Numerov', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/jost_final2.png', dpi=150, bbox_inches='tight')
print("Graphique: jost_final2.png")
