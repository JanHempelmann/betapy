#!/usr/bin/env python3
"""
Stiffness-shift cutoff convergence sweep.

Runs the intra-structure two-site (or bi-structure) stiffness-shift analysis
at multiple cutoff distances and reports how the total ΔpFC and pair count
converge.  Useful for choosing a cutoff and for checking stability of results.

Intra-structure mode (default)
-------------------------------
One directory with SPOSCAR + FORCE_CONSTANTS, one REFPOS with at least two
entries.  Site A (vacant/deintercalated) and site B (occupied/intercalated)
are compared within the same supercell using a rotation-invariant fingerprint.

Usage examples
--------------
# Intra-site sweep with default cutoffs (4.0–5.0 in 0.25 steps):
python scripts/cutoff_sweep.py \\
    path/to/phonon_and_forceconstnts_directory \\
    path/to/REFPOS

# Custom cutoff range, save CSV:
python scripts/cutoff_sweep.py <dir> <refpos> \\
    --cutoffs 3.5 4.0 4.5 5.0 5.5 6.0 --out sweep.csv

# Print in eV/Å² instead of N/m:
python scripts/cutoff_sweep.py <dir> <refpos> --ev
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importable without installation when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betapy.core.structure import Supercell
from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS, read_refpos
from betapy.core.projection import (
    find_refsite_pairs,
    match_fc_pairs_direct,
    stiffness_shift_from_pairs,
    structural_disturbance,
)
from betapy.core.constants import EV_ANG2_TO_N_M

_OCCUPANT_THRESHOLD = 1.0  # Å — atom closer than this to refsite B is the occupant


def _wide_shell(sc, fc, frac, cutoff, excl_sp, core_pairs):
    """Return pairs in the shell (cutoff, cutoff * 1.5] around frac."""
    wide, _ = find_refsite_pairs(
        sc, fc['atomic_pairs'], fc['force_matrices'],
        frac, cutoff=cutoff * 1.5, min_distance=0.0,
        exclude_species=excl_sp, show_progress=False,
    )
    core_keys = {(r['atom1_idx'], r['atom2_idx']) for r in core_pairs}
    return [r for r in wide if (r['atom1_idx'], r['atom2_idx']) not in core_keys]


def _two_phase_match(sc, fc, frac_a, frac_b, excl_sp, cutoff, tol):
    """
    Find and match pairs for both sites using a symmetric two-phase strategy.

    Phase 1: both sides at `cutoff`.
    Phase 2A: any A pairs still unmatched after phase 1 are tried against B
              pairs in the shell [cutoff, cutoff * 1.5].
    Phase 2B: any B pairs still unmatched after phase 1 are tried against A
              pairs in the shell [cutoff, cutoff * 1.5].

    Both phase-2 searches are needed because the asymmetry between the two
    sites (due to vacancy-induced relaxation) can leave edge pairs on either
    side just outside the base cutoff of the other site.

    Returns (matched, unmatched_a, unmatched_b, offsite_a, offsite_b).
    offsite_a and offsite_b contain only the base-cutoff pairs (for display).
    """
    offsite_a, _ = find_refsite_pairs(
        sc, fc['atomic_pairs'], fc['force_matrices'],
        frac_a, cutoff=cutoff, min_distance=0.0,
        exclude_species=excl_sp, show_progress=False,
    )
    offsite_b, _ = find_refsite_pairs(
        sc, fc['atomic_pairs'], fc['force_matrices'],
        frac_b, cutoff=cutoff, min_distance=0.0,
        exclude_species=excl_sp, show_progress=False,
    )

    matched, ua, ub = match_fc_pairs_direct(
        offsite_a, offsite_b, sc, sc, frac_a, frac_b,
        tol=tol, directional=False,
    )

    # Phase 2A: unmatched A → wider B shell
    if ua:
        ob_extra = _wide_shell(sc, fc, frac_b, cutoff, excl_sp, offsite_b)
        m2, ua, _ = match_fc_pairs_direct(
            ua, ob_extra, sc, sc, frac_a, frac_b,
            tol=tol, directional=False,
        )
        matched = matched + m2

    # Phase 2B: unmatched B → wider A shell
    if ub:
        oa_extra = _wide_shell(sc, fc, frac_a, cutoff, excl_sp, offsite_a)
        m2, _, ub = match_fc_pairs_direct(
            oa_extra, ub, sc, sc, frac_a, frac_b,
            tol=tol, directional=False,
        )
        matched = matched + m2

    return matched, ua, ub, offsite_a, offsite_b


def run_intra_sweep(dir_path, refpos_path, cutoffs, site_idx_a, site_idx_b,
                    tol, no_excl) -> tuple:
    """
    Run the intra-site sweep and return a DataFrame of results.

    At each requested cutoff, the full two-phase matching is run independently
    (same as the GUI).  Phase 2A/2B extend the search to cutoff×1.5 so that
    edge pairs whose counterpart sits just outside the strict sphere are found
    correctly without wrong assignments.  Running fresh at each cutoff ensures
    the Hungarian-optimal assignment is always computed for the actual pair pool
    at that cutoff, giving values consistent with the GUI.
    """
    sc = Supercell(read_SPOSCAR(dir_path / 'SPOSCAR'))
    fc = read_FORCE_CONSTANTS(dir_path / 'FORCE_CONSTANTS')
    all_pos = read_refpos(refpos_path)['positions']

    if site_idx_a >= len(all_pos) or site_idx_b >= len(all_pos):
        sys.exit(f'Site index out of range: REFPOS has {len(all_pos)} entries')

    frac_a = np.asarray(all_pos[site_idx_a])
    frac_b = np.asarray(all_pos[site_idx_b])

    excl_sp = None
    if not no_excl:
        dists = [sc.distance_to_point(k + 1, frac_b) for k in range(sc.n_atoms)]
        ni = min(range(sc.n_atoms), key=lambda k: dists[k])
        if dists[ni] < _OCCUPANT_THRESHOLD:
            excl_sp = {sc.species(ni + 1)}

    print(f'  Supercell : {sc}')
    print(f'  Site A    : {frac_a}  (index {site_idx_a})')
    print(f'  Site B    : {frac_b}  (index {site_idx_b})')
    print(f'  Excl. sp. : {excl_sp or "none"}')
    print(f'  Tolerance : {tol} Å')
    print()

    rows = []
    for cutoff in sorted(cutoffs):
        matched, ua, ub, _, _ = _two_phase_match(
            sc, fc, frac_a, frac_b, excl_sp, cutoff, tol,
        )
        n     = len(matched)
        total = sum(m['delta_pfc'] for m in matched)
        dist_met = structural_disturbance(matched)

        by_sp: dict = {}
        for m in matched:
            key = (m['species1'], m['species2'])
            by_sp[key] = by_sp.get(key, 0.0) + m['delta_pfc']

        if ua or ub:
            print(f'  cutoff = {cutoff:.2f} Å ...  {n} pairs  '
                  f'ΔpFC = {total:+.4f} eV/Å²  '
                  f'[{len(ua)} unmatched A, {len(ub)} unmatched B]')
        else:
            print(f'  cutoff = {cutoff:.2f} Å ...  {n} pairs  ΔpFC = {total:+.4f} eV/Å²')

        row = {
            'cutoff_ang':          cutoff,
            'n_matched':           n,
            'n_unmatched_a':       len(ua),
            'n_unmatched_b':       len(ub),
            'delta_pfc_eV_ang2':   total,
            'delta_pfc_N_m':       total * EV_ANG2_TO_N_M,
            'min_delta_eV_ang2':   dist_met.get('min_delta',   0.0),
            'min_delta_N_m':       dist_met.get('min_delta',   0.0) * EV_ANG2_TO_N_M,
            'min_species':         dist_met.get('min_species', ''),
            'total_abs_eV_ang2':   dist_met.get('total_abs',   0.0),
            'total_abs_N_m':       dist_met.get('total_abs',   0.0) * EV_ANG2_TO_N_M,
            'mean_abs_eV_ang2':    dist_met.get('mean_abs',    0.0),
            'mean_abs_N_m':        dist_met.get('mean_abs',    0.0) * EV_ANG2_TO_N_M,
        }
        for (sp1, sp2), val in by_sp.items():
            row[f'd_{sp1}-{sp2}_eV_ang2'] = val
            row[f'd_{sp1}-{sp2}_N_m']     = val * EV_ANG2_TO_N_M
        rows.append(row)

    return pd.DataFrame(rows), excl_sp


def print_summary(df, unit, excl_sp=None):
    factor = EV_ANG2_TO_N_M if unit == 'N/m' else 1.0
    col    = 'delta_pfc_N_m'    if unit == 'N/m' else 'delta_pfc_eV_ang2'
    col_mn = 'min_delta_N_m'    if unit == 'N/m' else 'min_delta_eV_ang2'
    col_ta = 'total_abs_N_m'    if unit == 'N/m' else 'total_abs_eV_ang2'
    col_ma = 'mean_abs_N_m'     if unit == 'N/m' else 'mean_abs_eV_ang2'

    sep = '─' * 52

    for _, row in df.iterrows():
        c = row['cutoff_ang']
        um_note = (f'  [!] {int(row["n_unmatched_a"])} unmatched A, '
                   f'{int(row["n_unmatched_b"])} unmatched B'
                   if row['n_unmatched_a'] or row['n_unmatched_b'] else '')
        print(f'\n── cutoff = {c:.2f} Å {sep[:36]}')
        print(f'  Matched pairs   : {int(row["n_matched"])}{um_note}')
        if excl_sp:
            print(f'  Excl. species   : {", ".join(sorted(excl_sp))}')
        print(f'  Σ ΔpFC          : {row[col]:+.5f}  {unit}')
        print(f'  Min ΔpFC        : {row[col_mn]:+.5f}  {unit}'
              + (f'  ({row["min_species"]})' if row.get('min_species') else ''))
        print(f'  Total |ΔpFC|    : {row[col_ta]:.5f}  {unit}  over {int(row["n_matched"])} bonds')
        print(f'  Mean  |ΔpFC|    : {row[col_ma]:.5f}  {unit}')

        sp_cols = [(k, v) for k, v in row.items()
                   if k.endswith('_N_m' if unit == 'N/m' else '_eV_ang2')
                   and k.startswith('d_')
                   and not pd.isna(v)]
        if sp_cols:
            print(f'  Per-species ΔpFC:')
            for k, v in sorted(sp_cols, key=lambda x: abs(x[1]), reverse=True):
                sp_label = k[2:].replace('_N_m', '').replace('_eV_ang2', '')
                print(f'    {sp_label:<12} {v:+.5f}  {unit}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('dir',
                   help='Directory containing SPOSCAR and FORCE_CONSTANTS')
    p.add_argument('refpos',
                   help='REFPOS file (two or more fractional-coordinate entries)')
    p.add_argument('--cutoffs', nargs='+', type=float,
                   default=[4.0, 4.25, 4.5, 4.75, 5.0],
                   metavar='Å',
                   help='Cutoff values in Å (default: 4.0 4.25 4.5 4.75 5.0)')
    p.add_argument('--site-a', type=int, default=0, metavar='IDX',
                   help='REFPOS index for site A / vacant (default: 0)')
    p.add_argument('--site-b', type=int, default=1, metavar='IDX',
                   help='REFPOS index for site B / occupied (default: 1)')
    p.add_argument('--tol', type=float, default=1.5, metavar='Å',
                   help='Fingerprint matching tolerance in Å (default: 1.5)')
    p.add_argument('--no-excl', action='store_true',
                   help='Do not exclude the site-occupying species from pairs')
    p.add_argument('--ev', action='store_true',
                   help='Display values in eV/Å² instead of N/m')
    p.add_argument('--out', metavar='FILE',
                   help='Save full results DataFrame to CSV')
    args = p.parse_args()

    print(f'betapy cutoff sweep — intra-structure mode')
    print(f'  Dir    : {args.dir}')
    print(f'  REFPOS : {args.refpos}')
    print()

    df, excl_sp = run_intra_sweep(
        dir_path   = Path(args.dir),
        refpos_path= Path(args.refpos),
        cutoffs    = args.cutoffs,
        site_idx_a = args.site_a,
        site_idx_b = args.site_b,
        tol        = args.tol,
        no_excl    = args.no_excl,
    )

    unit = 'eV/Å²' if args.ev else 'N/m'
    print_summary(df, unit, excl_sp=excl_sp)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f'\nWritten: {args.out}')


if __name__ == '__main__':
    main()
