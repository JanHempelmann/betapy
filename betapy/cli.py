"""
Command-line interface for betapy.

Thin wrapper: parse settings, call core functions, report results.
All scientific logic lives in betapy.core.
"""

import timeit
from pathlib import Path

from betapy.core.settings import Settings
from betapy.core.io import (
    read_SPOSCAR, read_FORCE_CONSTANTS, read_refpos,
    write_unique_pfcs, write_refsite_pfcs, write_refsite_onsite_pfcs,
)
from betapy.core.structure import Supercell
from betapy.core.projection import (
    compute_bulk_pfcs, unique_pfcs,
    find_refsite_pairs, refsite_results_to_dataframes,
    match_atoms_across_structures, match_fc_pairs,
    stiffness_shift_from_pairs, fallback_equal_count_shift,
)


def _load_structure(sposcar_path, fc_path):
    """Load and return (Supercell, fc_data)."""
    supercell = Supercell(read_SPOSCAR(sposcar_path))
    fc_data   = read_FORCE_CONSTANTS(fc_path)
    return supercell, fc_data


def run_bulk_analysis(supercell, fc_data, settings):
    t0 = timeit.default_timer()
    results, onsite, _ = compute_bulk_pfcs(
        supercell,
        fc_data['atomic_pairs'],
        fc_data['force_matrices'],
    )
    df_unique = unique_pfcs(results)
    print(f'  Off-site pairs : {len(results)}')
    print(f'  On-site terms  : {len(onsite)}')
    print(f'  Unique pFCs    : {len(df_unique)}')
    if settings.store:
        write_unique_pfcs(df_unique)
        print('  Written: unique_pFCs.csv')
    print(f'  Time           : {timeit.default_timer()-t0:.3f} s')
    return results, df_unique


def run_refsite_analysis(supercell, fc_data, settings):
    rs = settings.refsite
    try:
        refpos_data = read_refpos(rs.file)
    except FileNotFoundError:
        print(f'  Error: REFPOS file not found at {rs.file}')
        return None, None

    all_offsite, all_onsite = [], []
    for idx, frac_pos in enumerate(refpos_data['positions']):
        exclude_sp = None
        if rs.exclude_refsite_species:
            dists     = [supercell.distance_to_point(k + 1, frac_pos)
                         for k in range(supercell.n_atoms)]
            near_idx  = min(range(supercell.n_atoms), key=lambda k: dists[k])
            exclude_sp = {supercell.species(near_idx + 1)}
        offsite, onsite = find_refsite_pairs(
            supercell,
            fc_data['atomic_pairs'],
            fc_data['force_matrices'],
            frac_pos,
            rs.cutoff,
            exclude_species=exclude_sp,
        )
        all_offsite.extend(offsite)
        all_onsite.extend(onsite)
        excl_note = f', excl. {next(iter(exclude_sp))} pairs' if exclude_sp else ''
        print(f'  Site {idx}: {len(offsite)} off-site, {len(onsite)} on-site{excl_note}')

    df_off, df_on = refsite_results_to_dataframes(
        all_offsite, all_onsite, refpos_data['label']
    )
    if settings.store:
        write_refsite_pfcs(df_off)
        write_refsite_onsite_pfcs(df_on)
        print('  Written: refsite_pFCs.csv, refsite_onsite_pFCs.csv')
    return df_off, df_on


def _resolve_refpos(structure_settings, fallback_path):
    """Return REFPOS path: per-structure override if set, else shared fallback."""
    if structure_settings.refpos is not None:
        return structure_settings.refpos
    return fallback_path


def run_stiffness_shift(settings):
    """
    Load two structures, run refsite projection on each, match equivalent
    force-constant pairs by atom position, and compute the stiffness shift.

    Falls back to equal-count truncation (with a warning) if position
    matching fails for any species pair.
    """
    ss = settings.stiffness_shift

    print('  Loading structure A ...')
    sc_a, fc_a = _load_structure(
        ss.structure_a.sposcar, ss.structure_a.force_constants
    )
    print(f'    {sc_a}  |  {len(fc_a["atomic_pairs"])} pairs')

    print('  Loading structure B ...')
    sc_b, fc_b = _load_structure(
        ss.structure_b.sposcar, ss.structure_b.force_constants
    )
    print(f'    {sc_b}  |  {len(fc_b["atomic_pairs"])} pairs')

    # Resolve REFPOS paths
    refpos_path_a = _resolve_refpos(ss.structure_a, ss.refpos)
    refpos_path_b = _resolve_refpos(ss.structure_b, ss.refpos)
    try:
        refpos_a = read_refpos(refpos_path_a)
    except FileNotFoundError:
        print(f'  Error: REFPOS for structure A not found at {refpos_path_a}')
        return None
    try:
        refpos_b = read_refpos(refpos_path_b)
    except FileNotFoundError:
        print(f'  Error: REFPOS for structure B not found at {refpos_path_b}')
        return None

    # --- Determine intercalated species from structure B (for exclusion filter) ---
    # Find the atom in B that sits within min_site_dist of each refsite position.
    # That species is then excluded from both A and B so the analysis reflects
    # only the host-framework contributions.
    intercalated_species = set()
    if ss.exclude_refsite_species:
        for frac_pos in refpos_b['positions']:
            dists    = [sc_b.distance_to_point(k + 1, frac_pos)
                        for k in range(sc_b.n_atoms)]
            near_idx = min(range(sc_b.n_atoms), key=lambda k: dists[k])
            if dists[near_idx] < ss.min_site_dist:
                intercalated_species.add(sc_b.species(near_idx + 1))
    excl_arg = intercalated_species if intercalated_species else None

    # --- Refsite projection ---
    print(f'  Projecting in structure A (cutoff {ss.cutoff} Å) ...')
    offsite_a = []
    for frac_pos in refpos_a['positions']:
        offsite, _ = find_refsite_pairs(
            sc_a, fc_a['atomic_pairs'], fc_a['force_matrices'],
            frac_pos, cutoff=ss.cutoff, min_distance=0.0,
            exclude_species=excl_arg,
        )
        offsite_a.extend(offsite)
    excl_note = f', excl. {"/".join(sorted(intercalated_species))} pairs' if excl_arg else ''
    print(f'    {len(offsite_a)} off-site pairs{excl_note}')

    print(f'  Projecting in structure B (cutoff {ss.cutoff} Å) ...')
    offsite_b = []
    for frac_pos in refpos_b['positions']:
        offsite, _ = find_refsite_pairs(
            sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
            frac_pos, cutoff=ss.cutoff, min_distance=ss.min_site_dist,
            exclude_species=excl_arg,
        )
        offsite_b.extend(offsite)
    print(f'    {len(offsite_b)} off-site pairs (site-occupying atom + {"/".join(sorted(intercalated_species or {"none"}))} pairs excluded)')

    # --- Atom position matching ---
    print('  Matching atoms across structures by fractional position ...')
    all_species = sorted(set(sc_a.chem_symbols) & set(sc_b.chem_symbols))
    atom_matches  = {}   # idx_a -> idx_b, covering all matchable species
    matching_failed = []

    for sp in all_species:
        if sp not in sc_b.chem_symbols:
            print(f'    {sp}: absent in structure B — pairs skipped')
            continue
        matches, unmatched = match_atoms_across_structures(
            sc_a, sc_b, sp, tolerance=ss.match_tolerance
        )
        atom_matches.update(matches)
        if unmatched:
            print(f'    WARNING: {sp}: {len(unmatched)} atoms unmatched '
                  f'(tolerance {ss.match_tolerance} Å) — '
                  f'falling back to equal-count for this species')
            matching_failed.append(sp)
        else:
            print(f'    {sp}: {len(matches)} atoms matched')

    # --- Pair matching and shift computation ---
    if not matching_failed:
        # Primary path: explicit pair matching
        print('  Matching force-constant pairs ...')
        matched, unmatched_pairs, _ = match_fc_pairs(offsite_a, offsite_b, atom_matches, sc_a)
        print(f'    {len(matched)} pairs matched, '
              f'{len(unmatched_pairs)} unmatched (species absent in B or missing FC)')

        df, total = stiffness_shift_from_pairs(matched)
        method = 'position-matched pairs'

    else:
        # Fallback: equal-count truncation by distance
        print()
        print('  WARNING: Atom matching failed for one or more species.')
        print('  WARNING: Falling back to equal-count truncation ordered by distance.')
        print('  WARNING: Results may be less reliable — check your structures.')
        print()
        df, total, n = fallback_equal_count_shift(offsite_a, offsite_b)
        method = f'equal-count fallback ({n} pairs)'

    print(f'\n Method: {method}')
    print(f'  Total stiffness shift (B − A): {total:+.6f}')

    if settings.store:
        out = Path('stiffness_shift.csv')
        df.to_csv(out, index=False)
        print(f'  Written: {out}')

    return df, total


def main():
    settings, args = Settings.from_cli()

    # Special one-shot commands
    if args.write_template:
        path = Settings.write_template()
        print(f'Template written to {path}')
        return

    if args.gui:
        from betapy.gui.app import main as gui_main
        gui_main()
        return

    t_total = timeit.default_timer()
    print('betapy — projected force constant analysis')
    print('=' * 45)

    # --- Stiffness-shift mode (two-structure, self-contained) ---
    if settings.stiffness_shift is not None:
        print('\n[Stiffness-shift analysis]')
        run_stiffness_shift(settings)
        print(f'\nTotal time: {timeit.default_timer()-t_total:.3f} s')
        return

    # --- Single-structure mode ---
    print(f'\nReading {settings.sposcar} ...', end=' ', flush=True)
    supercell = Supercell(read_SPOSCAR(settings.sposcar))
    print(f'done  ({supercell})')

    print(f'Reading {settings.force_constants} ...', end=' ', flush=True)
    fc_data = read_FORCE_CONSTANTS(settings.force_constants)
    print(f'done  ({len(fc_data["atomic_pairs"])} pairs, '
          f'FC shape {fc_data["nats"]})')

    print('\n[Bulk pFC analysis]')
    run_bulk_analysis(supercell, fc_data, settings)

    if settings.refsite is not None:
        print(f'\n[Reference-site analysis — cutoff {settings.refsite.cutoff} Å]')
        run_refsite_analysis(supercell, fc_data, settings)

    print(f'\nTotal time: {timeit.default_timer()-t_total:.3f} s')


if __name__ == '__main__':
    main()
