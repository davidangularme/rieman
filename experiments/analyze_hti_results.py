# experiments/analyze_hti_results.py
# -*- coding: utf-8 -*-

import json
import numpy as np
from collections import defaultdict

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            rows.append(obj)
    return rows

def group_key(r, keys):
    return tuple(r[k] for k in keys)

def analyze(path):
    rows = load_jsonl(path)
    print(f"Loaded {len(rows)} rows.")

    # Group by (T_gauss, eps, T_window, L_total, heat_factor, calibrate)
    gkeys = ["T_gauss", "eps", "T_window", "L_total", "heat_factor", "do_calibrate_on_D"]
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r, gkeys)].append(r)

    for g, items in sorted(groups.items()):
        items = sorted(items, key=lambda x: (x["Nt_total"], x["J"], x["N_max"], x["K_max"]))
        deltas = np.array([it["Delta"] for it in items], dtype=float)
        taus = np.array([it["tau_mean"] for it in items], dtype=float)

        print("\n=== GROUP", dict(zip(gkeys, g)), "===")
        print(f"count={len(items)}  Delta_mean={deltas.mean():+.3e}  Delta_spread={deltas.max()-deltas.min():.3e}")

        # crude convergence check in Nt_total: compare average Delta at each Nt_total
        byNt = defaultdict(list)
        for it in items:
            byNt[it["Nt_total"]].append(it["Delta"])
        for Nt in sorted(byNt):
            arr = np.array(byNt[Nt], dtype=float)
            print(f"  Nt={Nt:6d}: Delta mean={arr.mean():+.3e} spread={arr.max()-arr.min():.3e}")

        # show a few worst instability points (large tau_std or large Delta magnitude)
        items_sorted = sorted(items, key=lambda x: (abs(x["Delta"]) + 10.0*x["tau_std"]), reverse=True)
        print("  Worst 5 (|Delta| + 10*tau_std):")
        for it in items_sorted[:5]:
            print(
                f"    Nt={it['Nt_total']:6d} J={it['J']:2d} N={it['N_max']:7d} K={it['K_max']:2d} "
                f"Delta={it['Delta']:+.3e} tau_std={it['tau_std']:.2e} v=[{it['v_min']:.2e},{it['v_max']:.2e}]"
            )

if __name__ == "__main__":
    analyze("../results/hti_sweep.jsonl")
