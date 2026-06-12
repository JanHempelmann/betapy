"""
Command-line interface for betapy.

Thin wrapper: parse settings, call core functions, report results.
All scientific logic lives in betapy.core.
"""

import timeit
from pathlib import Path

from betapy.core.settings import Settings
from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL
from betapy.core.io import (
    read_SPOSCAR, read_FORCE_CONSTANTS, read_refpos,
    write_unique_pfcs, write_bulk_pfcs,
    write_refsite_pfcs, write_refsite_onsite_pfcs,
)
from betapy.core.structure import Supercell
from betapy.core.projection import (
    compute_bulk_pfcs, unique_pfcs,
    find_refsite_pairs, refsite_results_to_dataframes,
    compare_refsite_projections,
    match_fc_pairs_direct, stiffness_shift_from_pairs,
    structural_disturbance, sum_intercalant_pfcs,
)


def _load_structure(sposcar_path, fc_path):
    """Load and return (Supercell, fc_data)."""
    supercell = Supercell(read_SPOSCAR(sposcar_path))
    fc_data   = read_FORCE_CONSTANTS(fc_path)
    return supercell, fc_data


def _annotate_lobster(df_unique, lobster_pairs):
    """Add ICOBI / ICOHP / ICOOP columns to df_unique from LOBSTER pair data."""
    from betapy.core.lobster import lookup
    available = {k for row in lobster_pairs for k in ('icobi', 'icohp', 'icoop') if k in row}
    for col_key, col_name in [('icobi', 'ICOBI'), ('icohp', 'ICOHP'), ('icoop', 'ICOOP')]:
        if col_key not in available:
            continue
        def _lob_val(row):
            v = lookup(lobster_pairs, row['Atom 1'], row['Atom 2'],
                       row['Distance (Angstr.)'], key=col_key)
            return round(v, 5) if v is not None else None
        df_unique[col_name] = [_lob_val(row) for _, row in df_unique.iterrows()]


def run_multicenter(supercell, bulk_results, lobster_dir, args):
    """
    Detect anomalous pFCs, trace multicenter chains, and print cobiBetween
    directives.  Called after run_bulk_analysis so bulk_results are available.
    """
    import math
    from betapy.core.multicenter import suggest_cobi_directives, append_cobi_directives

    poscar_path = Path(lobster_dir) / 'POSCAR'
    if not poscar_path.exists():
        print(f'  Error: {poscar_path} not found.  Provide a POSCAR in the '
              'LOBSTER directory for coordinate mapping.')
        return

    result = suggest_cobi_directives(
        bulk_results, supercell, poscar_path,
        n_sigma=args.mc_sigma,
        max_order=args.mc_max_order,
        min_angle_deg=args.mc_angle,
        max_nn_ratio=args.mc_ratio if args.mc_ratio > 0 else None,
    )

    flagged    = result['flagged_pairs']
    chains     = result['chains']
    directives = result['directives']

    print(f'\n  Flagged pairs   : {len(flagged)}')
    for f in flagged:
        sig = f'{f["n_sigma"]:.1f}σ' if not math.isnan(f['n_sigma']) else 'monotone'
        print(f'    {f["species1"]}-{f["species2"]}  '
              f'd={f["distance"]:.3f} Å  '
              f'pFC={f["mean_pfc"]:.4f}  [{f["method"]}, {sig}]')

    if not directives:
        if not flagged:
            print('  No anomalous pairs detected at this σ threshold.')
        else:
            print(f'\n  {len(flagged)} anomalous pair(s) detected but no multicenter '
                  f'chains could form.')
            if args.mc_ratio > 0:
                print(f'  Chain steps exceeding {args.mc_ratio:.1f}× NN were blocked '
                      f'(--mc-ratio).  If genuine multicenter bonds are expected here, '
                      f'try a higher value.')
            else:
                print('  No linear chain geometry was found.')
        return

    print(f'\n  Chains found    : {len(chains)}')
    for chain in chains:
        sp = '-'.join(chain['species_chain'])
        print(f'\n  Chain  {sp}  ({chain["total_distance"]:.2f} Å end-to-end)')
        for sub in chain['sub_chains']:
            d = sub["directive"] or '(mapping failed)'
            print(f'    {sub["order"]}-center:  {d}')

    print(f'\n  Unique directives ({len(directives)}):')
    for d in directives:
        print(f'    {d}')

    if args.store:
        out = Path('multicenter_directives.txt')
        out.write_text('\n'.join(directives) + '\n')
        print(f'\n  Written: {out}')

        lobsterin = Path(lobster_dir) / 'lobsterin'
        if lobsterin.exists():
            n = append_cobi_directives(lobsterin, directives)
            if n:
                print(f'  Appended {n} directive(s) to {lobsterin}')
            else:
                print(f'  lobsterin already contains all directives — nothing added')


def run_bulk_analysis(supercell, fc_data, settings, lobster_pairs=None):
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
    if lobster_pairs:
        _annotate_lobster(df_unique, lobster_pairs)
        cols = [c for c in ('ICOBI', 'ICOHP', 'ICOOP') if c in df_unique.columns]
        print(f'  LOBSTER cols   : {", ".join(cols)}')
    if settings.store:
        write_unique_pfcs(df_unique)
        write_bulk_pfcs(results)
        print('  Written: unique_pFCs.csv, bulk_pFCs.csv')
    print(f'  Time           : {timeit.default_timer()-t0:.3f} s')
    return results, df_unique


def run_refsite_analysis(supercell, fc_data, settings):
    rs = settings.refsite
    try:
        refpos_data = read_refpos(rs.file)
    except FileNotFoundError:
        print(f'  Error: REFPOS file not found at {rs.file}')
        return None, None

    factor     = EV_ANG2_TO_N_M if settings.unit == 'N/m' else 1.0
    unit_label = UNIT_LABEL.get(settings.unit, settings.unit)

    all_offsite, all_onsite = [], []
    site_offsite = []   # per-site, used for pairwise comparison below
    for idx, frac_pos in enumerate(refpos_data['positions']):
        exclude_sp = None
        if rs.exclude_refsite_species:
            dists    = [supercell.distance_to_point(k + 1, frac_pos)
                        for k in range(supercell.n_atoms)]
            near_idx = min(range(supercell.n_atoms), key=lambda k: dists[k])
            # Only exclude if an atom actually sits at the site (< 1 Å).
            # Vacant sites have no atom nearby, so nothing is excluded there.
            if dists[near_idx] < 1.0:
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
        site_offsite.append(offsite)
        pfc_sum   = sum(r['mean_pfc'] for r in offsite) * factor
        excl_note = f', excl. {next(iter(exclude_sp))} pairs' if exclude_sp else ''
        print(f'  Site {idx}: {len(offsite)} off-site, {len(onsite)} on-site'
              f'  Σ pFC = {pfc_sum:+.5f} {unit_label}{excl_note}')

    if len(site_offsite) >= 2:
        print(f'\n  ── Intra-structure comparison {"─" * 20}')
        positions = refpos_data['positions']

        # Determine the shared exclusion set for comparison: union of species
        # found within 1 Å of any occupied site. Applied to both sides so that
        # only framework bonds are matched — mirrors _StiffnessWorker sp_set logic.
        comparison_excl: set = set()
        if rs.exclude_refsite_species:
            for frac_pos in positions:
                dists    = [supercell.distance_to_point(k + 1, frac_pos)
                            for k in range(supercell.n_atoms)]
                near_idx = min(range(supercell.n_atoms), key=lambda k: dists[k])
                if dists[near_idx] < 1.0:
                    comparison_excl.add(supercell.species(near_idx + 1))

        for i in range(len(site_offsite)):
            for j in range(i + 1, len(site_offsite)):
                sub_i = [r for r in site_offsite[i]
                         if r['species1'] not in comparison_excl
                         and r['species2'] not in comparison_excl]
                sub_j = [r for r in site_offsite[j]
                         if r['species1'] not in comparison_excl
                         and r['species2'] not in comparison_excl]
                matched, ua, ub = match_fc_pairs_direct(
                    sub_i, sub_j,
                    supercell, supercell,
                    positions[i], positions[j],
                    tol=1.5, directional=False,
                )
                _, delta = stiffness_shift_from_pairs(matched)
                print(f'  Site {i} → Site {j}:  ΔΣ pFC = {delta * factor:+.5f} {unit_label}'
                      f'  ({len(matched)} matched, {len(ua)} unmatched A, {len(ub)} unmatched B)')

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
    force-constant pairs by fractional-coordinate fingerprint, and compute
    the stiffness shift.

    The cutoff is enforced only on structure A; structure B uses twice the
    cutoff to ensure all equivalent pairs are found even when intercalation
    expands the cell significantly.  Matching uses a purely fractional
    fingerprint so it works even when A and B have different crystallographic
    origins (e.g. pnnm intercalation pairs).
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

    # Determine intercalated species from structure B (for exclusion filter)
    intercalated_species = set()
    if ss.exclude_refsite_species:
        for frac_pos in refpos_b['positions']:
            dists    = [sc_b.distance_to_point(k + 1, frac_pos)
                        for k in range(sc_b.n_atoms)]
            near_idx = min(range(sc_b.n_atoms), key=lambda k: dists[k])
            if dists[near_idx] < ss.min_site_dist:
                intercalated_species.add(sc_b.species(near_idx + 1))
    excl_arg = intercalated_species if intercalated_species else None
    excl_note = f', excl. {"/".join(sorted(intercalated_species))} pairs' if excl_arg else ''

    # Species present in both structures
    sp_set = set(sc_a.chem_symbols) & set(sc_b.chem_symbols)

    # B uses a generous cutoff so equivalent pairs are found even after
    # significant cell expansion on intercalation.
    cutoff_b = ss.cutoff * 1.5

    all_matched     = []
    all_unmatched_a = []
    all_unmatched_b = []

    for idx, (ref_a, ref_b) in enumerate(zip(refpos_a['positions'], refpos_b['positions'])):
        print(f'  Site {idx}: projecting A (cutoff {ss.cutoff} Å) ...')
        res_a, _ = find_refsite_pairs(
            sc_a, fc_a['atomic_pairs'], fc_a['force_matrices'],
            ref_a, cutoff=ss.cutoff, min_distance=0.0,
            exclude_species=excl_arg,
        )
        print(f'  Site {idx}: projecting B (cutoff {cutoff_b:.1f} Å) ...')
        res_b, _ = find_refsite_pairs(
            sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
            ref_b, cutoff=cutoff_b, min_distance=ss.min_site_dist,
            exclude_species=excl_arg,
        )

        sub_a = [r for r in res_a if r['species1'] in sp_set and r['species2'] in sp_set]
        sub_b = [r for r in res_b if r['species1'] in sp_set and r['species2'] in sp_set]
        print(f'    {len(sub_a)} A pairs, {len(sub_b)} B pairs{excl_note}')

        m, ua, ub = match_fc_pairs_direct(
            sub_a, sub_b, sc_a, sc_b, ref_a, ref_b, tol=ss.match_tolerance
        )
        all_matched.extend(m)
        all_unmatched_a.extend(ua)
        all_unmatched_b.extend(ub)
        print(f'    {len(m)} matched, {len(ua)} unmatched A, {len(ub)} unmatched B')

    df, total = stiffness_shift_from_pairs(all_matched)
    dist      = structural_disturbance(all_matched)

    # Intercalant contribution: framework → intercalant bonds in B only.
    # Use the standard cutoff (not 1.5×) and min_distance=0 so the intercalant
    # atom sitting at the refsite is also counted as atom2.
    intercalant_species = set(sc_b.chem_symbols) - set(sc_a.chem_symbols)
    intercalant_total = 0.0
    if intercalant_species:
        for ref_b in refpos_b['positions']:
            res_b_ic, _ = find_refsite_pairs(
                sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
                ref_b, cutoff=ss.cutoff, min_distance=0.0,
                exclude_species=None, show_progress=False,
            )
            ic_sum, _ = sum_intercalant_pfcs(res_b_ic, intercalant_species)
            intercalant_total += ic_sum

    factor     = EV_ANG2_TO_N_M if settings.unit == 'N/m' else 1.0
    unit_label = UNIT_LABEL.get(settings.unit, settings.unit)
    u          = unit_label

    print(f'\n Method: fractional-fingerprint matching')
    print(f'  Unmatched A: {len(all_unmatched_a)}   '
          f'Unmatched B: {len(all_unmatched_b)}')
    print(f'\n  ── Stiffness shift (B − A) {"─" * 24}')
    print(f'  Matched pairs   : {dist["n_pairs"]}')
    print(f'  Σ ΔpFC          : {total * factor:+.6f}  {u}')
    print(f'  Min ΔpFC        : {dist["min_delta"] * factor:+.6f}  {u}  ({dist["min_species"]})')
    print(f'\n  ── Structural disturbance {"─" * 26}')
    print(f'  Total |ΔpFC|    : {dist["total_abs"] * factor:.6f}  {u}  over {dist["n_pairs"]} bonds')
    print(f'  Mean  |ΔpFC|    : {dist["mean_abs"]  * factor:.6f}  {u}')
    if intercalant_species:
        sp_str = '/'.join(sorted(intercalant_species))
        print(f'\n  ── Intercalant contribution ({sp_str}) {"─" * 14}')
        print(f'  Σ pFC (B only)  : {intercalant_total * factor:+.6f}  {u}')

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
        gui_main(cli_args=args)
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

    # --- LOBSTER integration ---
    lobster_pairs = None
    from betapy.core.lobster import find_lobster_dir, load_pairs as _lob_load
    if settings.lobster_dir:
        _ldir = Path(settings.lobster_dir)
        lobster_pairs = _lob_load(_ldir)
        print(f'\nLOBSTER dir     : {_ldir}  ({len(lobster_pairs)} pair shells)')
    else:
        _ldir = find_lobster_dir(Path(settings.sposcar).parent)
        if _ldir is not None:
            lobster_pairs = _lob_load(_ldir)
            print(f'\nLOBSTER dir     : {_ldir.name} (auto-discovered, '
                  f'{len(lobster_pairs)} pair shells)')

    print('\n[Bulk pFC analysis]')
    bulk_results, _ = run_bulk_analysis(supercell, fc_data, settings, lobster_pairs=lobster_pairs)

    if args.multicenter:
        if _ldir is None:
            print('\n[Multicenter analysis]')
            print('  Error: no LOBSTER directory found.  Use --lobster-dir to specify one.')
        else:
            print(f'\n[Multicenter bonding analysis]')
            run_multicenter(supercell, bulk_results, _ldir, args)

    if settings.refsite is not None:
        print(f'\n[Reference-site analysis — cutoff {settings.refsite.cutoff} Å]')
        run_refsite_analysis(supercell, fc_data, settings)

    print(f'\nTotal time: {timeit.default_timer()-t_total:.3f} s')


if __name__ == '__main__':
    main()
