"""
Orchestration layer for the three top-level betapy analyses: bulk pFC,
reference-site pFC, and stiffness shift.

These functions compute and return structured results only — no printing,
no file writes. They exist so that betapy.cli (text output) and
betapy.mcp_server (JSON tool results) can share exactly one implementation
of each analysis instead of drifting apart.
"""

from pathlib import Path

import numpy as np

from betapy.core.constants import EV_ANG2_TO_N_M, UNIT_LABEL
from betapy.core.io import read_refpos
from betapy.core.lobster import annotate_lobster_columns
from betapy.core.stability import check_stability
from betapy.core.projection import (
    compute_bulk_pfcs, unique_pfcs,
    find_refsite_pairs, refsite_results_to_dataframes,
    match_fc_pairs_direct, stiffness_shift_from_pairs,
    structural_disturbance, sum_intercalant_pfcs,
)


def _unit_factor(unit):
    factor = EV_ANG2_TO_N_M if unit == 'N/m' else 1.0
    return factor, UNIT_LABEL.get(unit, unit)


def compute_bulk_analysis(supercell, fc_data, lobster_pairs=None):
    """
    Off-site + on-site bulk pFC projection over an entire supercell.

    Returns
    -------
    dict with keys:
        results    : list of dicts, one per off-site pair (see compute_bulk_pfcs)
        onsite     : list of dicts, one per on-site term
        df_unique  : DataFrame of unique pFC values (+ ICOBI/ICOHP/ICOOP columns
                     if lobster_pairs is given)
    """
    results, onsite, _ = compute_bulk_pfcs(
        supercell, fc_data['atomic_pairs'], fc_data['force_matrices'],
    )
    df_unique = unique_pfcs(results)
    if lobster_pairs:
        annotate_lobster_columns(df_unique, lobster_pairs)
    return {'results': results, 'onsite': onsite, 'df_unique': df_unique}


def compute_refsite_analysis(supercell, fc_data, refsite_settings, unit='eV/Ang2'):
    """
    Reference-site pFC projection for every site in a REFPOS file, plus
    pairwise Sigma-pFC comparison across sites when there are 2 or more.

    Raises FileNotFoundError if refsite_settings.file does not exist.

    Returns
    -------
    dict with keys:
        unit, unit_label
        sites       : list of dicts {index, frac_pos, n_offsite, n_onsite,
                      pfc_sum, pfc_sum_converted, exclude_species}
        comparisons : list of dicts {site_i, site_j, delta, delta_converted,
                      n_matched, n_unmatched_a, n_unmatched_b}
        df_offsite, df_onsite : DataFrames (raw eV/Ang2 values)
    """
    rs = refsite_settings
    refpos_data = read_refpos(rs.file)
    factor, unit_label = _unit_factor(unit)

    all_offsite, all_onsite = [], []
    site_offsite = []
    sites = []

    for idx, frac_pos in enumerate(refpos_data['positions']):
        exclude_sp = None
        if rs.exclude_refsite_species:
            dists    = [supercell.distance_to_point(k + 1, frac_pos)
                        for k in range(supercell.n_atoms)]
            near_idx = min(range(supercell.n_atoms), key=lambda k: dists[k])
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

        pfc_sum = sum(r['mean_pfc'] for r in offsite)
        sites.append({
            'index':             idx,
            'frac_pos':          list(np.asarray(frac_pos).tolist()),
            'n_offsite':         len(offsite),
            'n_onsite':          len(onsite),
            'pfc_sum':           pfc_sum,
            'pfc_sum_converted': pfc_sum * factor,
            'exclude_species':   sorted(exclude_sp) if exclude_sp else [],
        })

    comparisons = []
    if len(site_offsite) >= 2:
        positions = refpos_data['positions']

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
                comparisons.append({
                    'site_i':         i,
                    'site_j':         j,
                    'delta':          delta,
                    'delta_converted': delta * factor,
                    'n_matched':      len(matched),
                    'n_unmatched_a':  len(ua),
                    'n_unmatched_b':  len(ub),
                })

    df_off, df_on = refsite_results_to_dataframes(
        all_offsite, all_onsite, refpos_data['label']
    )

    return {
        'unit':        unit,
        'unit_label':  unit_label,
        'sites':       sites,
        'comparisons': comparisons,
        'df_offsite':  df_off,
        'df_onsite':   df_on,
    }


def _resolve_refpos(structure_settings, fallback_path):
    if structure_settings.refpos is not None:
        return structure_settings.refpos
    return fallback_path


def compute_stiffness_shift(settings):
    """
    Load two structures (A, B), run refsite projection on each, match
    equivalent force-constant pairs by fractional-coordinate fingerprint,
    and compute the stiffness shift B - A.

    See betapy.cli.run_stiffness_shift for the full method description;
    this function is the same computation with printing removed.

    Raises FileNotFoundError if either REFPOS file is missing.

    Returns
    -------
    dict with keys:
        unit, unit_label
        structure_a, structure_b : dict {n_atoms, n_pairs, stability}
            stability is a StabilityReport or None (see betapy.core.stability)
        n_unmatched_a, n_unmatched_b : int
        df_matched   : DataFrame, one row per matched pair (see stiffness_shift_from_pairs)
        total_shift, total_shift_converted : float, Sigma delta_pFC (B - A)
        disturbance  : dict from structural_disturbance(), plus *_converted variants
        intercalant_species : sorted list of str
        intercalant_total, intercalant_total_converted : float
    """
    from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS
    from betapy.core.structure import Supercell

    ss = settings.stiffness_shift
    factor, unit_label = _unit_factor(settings.unit)

    def _load(struct_settings):
        supercell = Supercell(read_SPOSCAR(struct_settings.sposcar))
        fc_data   = read_FORCE_CONSTANTS(struct_settings.force_constants)
        stability = check_stability(Path(struct_settings.sposcar).parent)
        return supercell, fc_data, stability

    sc_a, fc_a, stab_a = _load(ss.structure_a)
    sc_b, fc_b, stab_b = _load(ss.structure_b)

    refpos_path_a = _resolve_refpos(ss.structure_a, ss.refpos)
    refpos_path_b = _resolve_refpos(ss.structure_b, ss.refpos)
    refpos_a = read_refpos(refpos_path_a)
    refpos_b = read_refpos(refpos_path_b)

    intercalated_species = set()
    if ss.exclude_refsite_species:
        for frac_pos in refpos_b['positions']:
            dists    = [sc_b.distance_to_point(k + 1, frac_pos)
                        for k in range(sc_b.n_atoms)]
            near_idx = min(range(sc_b.n_atoms), key=lambda k: dists[k])
            if dists[near_idx] < ss.min_site_dist:
                intercalated_species.add(sc_b.species(near_idx + 1))
    excl_arg = intercalated_species if intercalated_species else None

    sp_set = set(sc_a.chem_symbols) & set(sc_b.chem_symbols)
    cutoff_b = ss.cutoff * 1.5

    all_matched, all_unmatched_a, all_unmatched_b = [], [], []

    for ref_a, ref_b in zip(refpos_a['positions'], refpos_b['positions']):
        res_a, _ = find_refsite_pairs(
            sc_a, fc_a['atomic_pairs'], fc_a['force_matrices'],
            ref_a, cutoff=ss.cutoff, min_distance=0.0,
            exclude_species=excl_arg,
        )
        res_b, _ = find_refsite_pairs(
            sc_b, fc_b['atomic_pairs'], fc_b['force_matrices'],
            ref_b, cutoff=cutoff_b, min_distance=ss.min_site_dist,
            exclude_species=excl_arg,
        )

        sub_a = [r for r in res_a if r['species1'] in sp_set and r['species2'] in sp_set]
        sub_b = [r for r in res_b if r['species1'] in sp_set and r['species2'] in sp_set]

        m, ua, ub = match_fc_pairs_direct(
            sub_a, sub_b, sc_a, sc_b, ref_a, ref_b, tol=ss.match_tolerance
        )
        all_matched.extend(m)
        all_unmatched_a.extend(ua)
        all_unmatched_b.extend(ub)

    df, total = stiffness_shift_from_pairs(all_matched)
    dist = structural_disturbance(all_matched)

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

    return {
        'unit':       settings.unit,
        'unit_label': unit_label,
        'structure_a': {
            'n_atoms': sc_a.n_atoms, 'n_pairs': len(fc_a['atomic_pairs']),
            'stability': stab_a,
        },
        'structure_b': {
            'n_atoms': sc_b.n_atoms, 'n_pairs': len(fc_b['atomic_pairs']),
            'stability': stab_b,
        },
        'n_unmatched_a': len(all_unmatched_a),
        'n_unmatched_b': len(all_unmatched_b),
        'df_matched':    df,
        'total_shift':            total,
        'total_shift_converted':  total * factor,
        'disturbance': {
            **dist,
            'total_abs_converted': dist['total_abs'] * factor,
            'mean_abs_converted':  dist['mean_abs'] * factor,
            'min_delta_converted': dist['min_delta'] * factor,
        },
        'intercalant_species':  sorted(intercalant_species),
        'intercalant_total':           intercalant_total,
        'intercalant_total_converted': intercalant_total * factor,
    }
