"""
Command-line interface for betapy.

Thin wrapper: parse settings, call core functions, report results.
All scientific logic lives in betapy.core.
"""

import timeit
from pathlib import Path

from betapy.core.settings import Settings
from betapy.core.io import (
    read_SPOSCAR, read_FORCE_CONSTANTS,
    write_unique_pfcs, write_bulk_pfcs,
    write_refsite_pfcs, write_refsite_onsite_pfcs,
)
from betapy.core.structure import Supercell
from betapy.core.analysis import (
    compute_bulk_analysis, compute_refsite_analysis, compute_stiffness_shift,
)


def _check_stability(directory, label=''):
    """
    Look for Phonopy mesh/band/qpoints output next to the loaded structure and
    warn about imaginary modes, if any. Silent if no such file is present.
    """
    from betapy.core.stability import check_stability, format_warning
    report = check_stability(directory)
    if report is None:
        return
    prefix = f'{label}: ' if label else ''
    if report.is_stable:
        print(f'  {prefix}Phonon stability check ({report.source.name}): OK '
              f'({report.n_modes_total} modes, {report.n_qpoints} q-point(s), '
              f'no imaginary modes)')
    else:
        print(f'  {prefix}Warning: {format_warning(report)}')


def run_multicenter(supercell, bulk_results, lobster_dir, args):
    """
    Detect anomalous pFCs, trace multicenter chains, and print cobiBetween
    directives.  Called after run_bulk_analysis so bulk_results are available.
    lobster_dir may be None; in that case chain detection still runs but
    cobiBetween directives cannot be generated.
    """
    import math
    from betapy.core.multicenter import suggest_cobi_directives, append_cobi_directives

    poscar_path = None
    if lobster_dir is not None:
        _p = Path(lobster_dir) / 'POSCAR'
        if _p.exists():
            poscar_path = _p
        else:
            print(f'  Warning: {_p} not found — chains will be detected '
                  'but cobiBetween directives cannot be generated.')

    import numpy as np
    L = supercell.lattice
    a, b, c = L[0], L[1], L[2]
    V = abs(float(np.dot(a, np.cross(b, c))))
    rc = min(
        V / np.linalg.norm(np.cross(b, c)),
        V / np.linalg.norm(np.cross(a, c)),
        V / np.linalg.norm(np.cross(a, b)),
    ) / 2.0

    result = suggest_cobi_directives(
        bulk_results, supercell, poscar_path,
        n_sigma=args.mc_sigma,
        max_order=args.mc_max_order,
        min_angle_deg=args.mc_angle,
        max_nn_ratio=args.mc_ratio if args.mc_ratio > 0 else None,
        bond_ratio_tol=args.mc_bond_tol if args.mc_bond_tol > 0 else None,
        reliability_cutoff=rc,
        pos_tol=args.mc_pos_tol,
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

    if not flagged:
        print('  No anomalous pairs detected at this σ threshold.')
        return

    if not chains:
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
            if sub["directive"] is None:
                d = '(no POSCAR — provide --lobster-dir to generate directive)'
            else:
                d = sub["directive"] or '(mapping failed)'
            print(f'    {sub["order"]}-center:  {d}')

    if poscar_path is None:
        print('\n  Note: cobiBetween directives require a LOBSTER directory with '
              'a POSCAR.  Use --lobster-dir to enable.')
        return

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
    data = compute_bulk_analysis(supercell, fc_data, lobster_pairs=lobster_pairs)
    results, onsite, df_unique = data['results'], data['onsite'], data['df_unique']

    print(f'  Off-site pairs : {len(results)}')
    print(f'  On-site terms  : {len(onsite)}')
    print(f'  Unique pFCs    : {len(df_unique)}')
    if lobster_pairs:
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
        data = compute_refsite_analysis(supercell, fc_data, rs, unit=settings.unit)
    except FileNotFoundError:
        print(f'  Error: REFPOS file not found at {rs.file}')
        return None, None

    unit_label = data['unit_label']
    for site in data['sites']:
        excl_note = f', excl. {site["exclude_species"][0]} pairs' if site['exclude_species'] else ''
        print(f'  Site {site["index"]}: {site["n_offsite"]} off-site, {site["n_onsite"]} on-site'
              f'  Σ pFC = {site["pfc_sum_converted"]:+.5f} {unit_label}{excl_note}')

    if data['comparisons']:
        print(f'\n  ── Intra-structure comparison {"─" * 20}')
        for c in data['comparisons']:
            print(f'  Site {c["site_i"]} → Site {c["site_j"]}:  '
                  f'ΔΣ pFC = {c["delta_converted"]:+.5f} {unit_label}'
                  f'  ({c["n_matched"]} matched, {c["n_unmatched_a"]} unmatched A, '
                  f'{c["n_unmatched_b"]} unmatched B)')

    df_off, df_on = data['df_offsite'], data['df_onsite']
    if settings.store:
        write_refsite_pfcs(df_off)
        write_refsite_onsite_pfcs(df_on)
        print('  Written: refsite_pFCs.csv, refsite_onsite_pFCs.csv')
    return df_off, df_on


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
    try:
        data = compute_stiffness_shift(settings)
    except FileNotFoundError as e:
        print(f'  Error: {e}')
        return None

    u = data['unit_label']
    from betapy.core.stability import format_warning

    for label, struct in (('Structure A', data['structure_a']),
                           ('Structure B', data['structure_b'])):
        print(f'  {label}: {struct["n_atoms"]} atoms  |  {struct["n_pairs"]} pairs')
        report = struct['stability']
        if report is not None and not report.is_stable:
            print(f'  {label}: Warning: {format_warning(report)}')

    dist = data['disturbance']
    print(f'\n Method: fractional-fingerprint matching')
    print(f'  Unmatched A: {data["n_unmatched_a"]}   '
          f'Unmatched B: {data["n_unmatched_b"]}')
    print(f'\n  ── Stiffness shift (B − A) {"─" * 24}')
    print(f'  Matched pairs   : {dist["n_pairs"]}')
    print(f'  Σ ΔpFC          : {data["total_shift_converted"]:+.6f}  {u}')
    print(f'  Min ΔpFC        : {dist["min_delta_converted"]:+.6f}  {u}  ({dist["min_species"]})')
    print(f'\n  ── Structural disturbance {"─" * 26}')
    print(f'  Total |ΔpFC|    : {dist["total_abs_converted"]:.6f}  {u}  over {dist["n_pairs"]} bonds')
    print(f'  Mean  |ΔpFC|    : {dist["mean_abs_converted"]:.6f}  {u}')
    if data['intercalant_species']:
        sp_str = '/'.join(data['intercalant_species'])
        print(f'\n  ── Intercalant contribution ({sp_str}) {"─" * 14}')
        print(f'  Σ pFC (B only)  : {data["intercalant_total_converted"]:+.6f}  {u}')

    df = data['df_matched']
    if settings.store:
        out = Path('stiffness_shift.csv')
        df.to_csv(out, index=False)
        print(f'  Written: {out}')

    return df, data['total_shift']


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

    # --- NAC correction (optional) ---
    if settings.born_dir:
        from betapy.core.nac import (
            compute_short_range_fc, apply_to_betapy_fc_data, find_phonopy_yaml
        )
        ph_yaml = settings.phonopy_yaml
        if ph_yaml is None:
            ph_yaml = find_phonopy_yaml(Path(settings.force_constants).parent)
        if ph_yaml is None:
            raise FileNotFoundError(
                'NAC correction requires phonopy_disp.yaml next to '
                'FORCE_CONSTANTS, or specify phonopy_yaml in settings.'
            )
        print(f'Applying NAC correction (born_dir: {settings.born_dir}) ...',
              end=' ', flush=True)
        phi_short = compute_short_range_fc(ph_yaml, settings.force_constants,
                                           settings.born_dir)
        apply_to_betapy_fc_data(fc_data, phi_short)
        print('done')

    _check_stability(Path(settings.sposcar).parent)

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
        print(f'\n[Multicenter bonding analysis]')
        run_multicenter(supercell, bulk_results, _ldir, args)

    if settings.refsite is not None:
        print(f'\n[Reference-site analysis — cutoff {settings.refsite.cutoff} Å]')
        run_refsite_analysis(supercell, fc_data, settings)

    print(f'\nTotal time: {timeit.default_timer()-t_total:.3f} s')


if __name__ == '__main__':
    main()
