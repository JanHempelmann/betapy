"""
Diagnostic script for multicenter visualization vs detection consistency.

Checks three things:
  1. Do the worker's visualization fits match what detection uses?
  2. For each flagged pair, is phi_iso^{-1/3} actually outside the viz band?
  3. Which flagged pairs are missing from badger_data (highlight bug)?

Run from betapy root:
  python3 examples/GeTe/GeTe_ph/debug_gete.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from collections import defaultdict
from scipy.stats import theilslopes, median_abs_deviation

from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS
from betapy.core.structure import Supercell
from betapy.core.projection import compute_bulk_pfcs
from betapy.core.badger import compute_badger_quantities
from betapy.core.multicenter import suggest_cobi_directives

BASE    = 'examples/GeTe/GeTe_ph'
POSCAR  = f'{BASE}/POSCAR'
N_SIGMA = 2.5


def compute_viz_fits(results, sc):
    """Reproduce the worker's Theil-Sen fit exactly."""
    rl       = min(np.linalg.norm(v) for v in sc.lattice) / 2.0
    reliable = [r for r in results if r['distance'] <= rl]
    reliable = compute_badger_quantities(reliable)

    seen, deduped = set(), []
    for r in reliable:
        k = (min(r['atom1_idx'], r['atom2_idx']), max(r['atom1_idx'], r['atom2_idx']))
        if k not in seen:
            seen.add(k)
            deduped.append(r)

    by_pair = defaultdict(list)
    for r in deduped:
        by_pair[tuple(sorted([r['species1'], r['species2']]))].append(r)

    fits    = {}
    rec_map = {}
    for sp_key, recs in by_pair.items():
        pfcs  = np.array([r.get('phi_iso', np.nan) for r in recs])
        dists = np.array([r['distance'] for r in recs])
        valid = np.isfinite(pfcs) & (pfcs > 0)
        v_pfcs, v_dists = pfcs[valid], dists[valid]
        v_recs = [r for r, v in zip(recs, valid) if v]

        slope = intercept = std = std_raw = np.nan
        if valid.sum() >= 4 and float(v_dists.max() - v_dists.min()) >= 0.05:
            ic = v_pfcs ** (-1.0 / 3.0)
            _rd = np.round(v_dists, 3)
            _, _ux = np.unique(_rd, return_index=True)
            if len(_ux) >= 4:
                slope, intercept, *_ = theilslopes(ic[_ux], v_dists[_ux])
                pred_uniq = slope * v_dists[_ux] + intercept
                log_ratio_uniq = 3.0 * np.log(
                    np.maximum(pred_uniq, 1e-12) / ic[_ux])
                std_raw = float(median_abs_deviation(log_ratio_uniq) * 1.4826)
                std     = max(std_raw, 1e-6)

        fits[sp_key] = {'slope': slope, 'intercept': intercept,
                        'std': std, 'std_raw': std_raw, 'n': int(valid.sum())}
        rec_map[sp_key] = {
            (min(r['atom1_idx'], r['atom2_idx']),
             max(r['atom1_idx'], r['atom2_idx'])): r
            for r in v_recs
        }
    return fits, rec_map, rl


def check_pair(fp, fits, n_sigma):
    """Return (viz_nsig, in_band) for a flagged pair."""
    sp   = tuple(sorted([fp['species1'], fp['species2']]))
    phi  = fp.get('phi_iso', np.nan)
    dist = fp['distance']
    fit  = fits.get(sp, {})
    sl   = fit.get('slope',     np.nan)
    ic   = fit.get('intercept', np.nan)
    std  = fit.get('std',       np.nan)
    if not (np.isfinite(sl) and np.isfinite(phi) and phi > 0):
        return np.nan, None
    pred     = sl * dist + ic
    phi_cbrt = phi ** (-1.0 / 3.0)
    if pred <= 0 or phi_cbrt <= 0:
        return np.nan, None
    log_ratio = 3.0 * np.log(pred / phi_cbrt)   # positive = stronger than predicted
    viz_nsig  = log_ratio / std
    in_band   = log_ratio < n_sigma * std
    return viz_nsig, in_band


# ── Main ─────────────────────────────────────────────────────────────────────
sc      = Supercell(read_SPOSCAR(f'{BASE}/SPOSCAR'))
fc      = read_FORCE_CONSTANTS(f'{BASE}/FORCE_CONSTANTS')
results, _, _ = compute_bulk_pfcs(sc, fc['atomic_pairs'], fc['force_matrices'])

viz_fits, viz_rec_map, rl = compute_viz_fits(results, sc)

print(f'Reliability limit : {rl:.4f} Å')
print(f'max_detect_dist   : {rl * 0.75:.4f} Å\n')

print('=== WORKER VISUALIZATION FITS ===')
for sp_key in sorted(viz_fits):
    f = viz_fits[sp_key]
    print(f'  {sp_key}  n={f["n"]}'
          f'  slope={f["slope"]:.5f}  intercept={f["intercept"]:.5f}'
          f'  std_raw={f["std_raw"]:.3e}  std={f["std"]:.3e}')

print('\n=== DETECTION PATH ===')
det_result = suggest_cobi_directives(
    results, sc, POSCAR, n_sigma=N_SIGMA, detect_cutoff_frac=0.75)
flagged = det_result['flagged_pairs']
print(f'Total flagged pairs: {len(flagged)}')

print('\n=== PER-FLAGGED-PAIR ANALYSIS ===')
missing_from_plot = []
n_inside = 0

for fp in flagged:
    sp       = tuple(sorted([fp['species1'], fp['species2']]))
    a1, a2   = fp['atom1_idx'], fp['atom2_idx']
    pair_key = (min(a1, a2), max(a1, a2))
    dist     = fp['distance']
    phi      = fp.get('phi_iso', np.nan)
    phi_cbrt = phi ** (-1.0 / 3.0) if (np.isfinite(phi) and phi > 0) else np.nan
    det_nsig = fp.get('n_sigma', np.nan)
    viz_nsig, in_band = check_pair(fp, viz_fits, N_SIGMA)

    in_plot = pair_key in viz_rec_map.get(sp, {})
    if not in_plot:
        missing_from_plot.append((sp, pair_key, dist))
    if in_band:
        n_inside += 1

    band_tag = '*** INSIDE BAND ***' if in_band else 'outside band ✓'
    plot_tag = 'in plot ✓' if in_plot else '*** MISSING FROM PLOT ***'
    print(f'  {sp} ({a1},{a2})  d={dist:.3f} Å  phi_iso={phi:.5f}'
          f'  phi^(-1/3)={phi_cbrt:.5f}')
    print(f'    det_nsig={det_nsig:.2f}  viz_nsig={viz_nsig:.2f}'
          f'  [{band_tag}]  [{plot_tag}]')

print('\n=== SUMMARY ===')
print(f'Flagged pairs total          : {len(flagged)}')
print(f'Inside viz band (BUG)        : {n_inside}')
print(f'Missing from plot (BUG)      : {len(missing_from_plot)}')
for sp, key, d in missing_from_plot:
    print(f'  {sp} atoms{key}  d={d:.3f} Å')
