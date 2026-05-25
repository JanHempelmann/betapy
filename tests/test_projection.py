"""
Tests for betapy.core.projection and betapy.core.structure

Run with:  python -m pytest tests/
"""

import pytest
import numpy as np

from betapy.core.structure import Supercell
from betapy.core.projection import (
    compute_bulk_pfcs, find_refsite_pairs, unique_pfcs,
    match_atoms_across_structures, match_fc_pairs,
    stiffness_shift_from_pairs, fallback_equal_count_shift,
)


def make_simple_supercell():
    """2-atom cubic cell: Li at origin, O at body centre."""
    return Supercell({
        'skal': 1.0,
        'lattice': [[4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0]],
        'chem_symbols': ['Li', 'O'],
        'chem_atoms': [1, 1],
        'positions': [[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.5]],
    })


def test_supercell_species():
    sc = make_simple_supercell()
    assert sc.species(1) == 'Li'
    assert sc.species(2) == 'O'


def test_supercell_distance():
    sc = make_simple_supercell()
    # Body diagonal of a 4 Å cube
    expected = 4.0 * np.sqrt(3) / 2
    assert sc.atom_distance(1, 2) == pytest.approx(expected, rel=1e-6)


def test_supercell_pbc_wrapping():
    sc = make_simple_supercell()
    # Distance from atom 2 (0.5,0.5,0.5) to a point just outside the cell
    # should wrap correctly
    d = sc.distance([0.5, 0.5, 0.5], [1.1, 0.5, 0.5])
    d_wrapped = sc.distance([0.5, 0.5, 0.5], [0.1, 0.5, 0.5])
    assert d == pytest.approx(d_wrapped, rel=1e-6)


def test_compute_bulk_pfcs_onsite():
    sc = make_simple_supercell()
    # On-site pair (i == j)
    pairs = [[1, 1]]
    fc_mat = [[[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]]
    results, onsite, distances = compute_bulk_pfcs(sc, pairs, fc_mat)
    assert len(results) == 0
    assert len(onsite) == 1
    assert onsite[0]['species'] == 'Li'


def test_compute_bulk_pfcs_offsite():
    sc = make_simple_supercell()
    pairs = [[1, 2]]
    # Identity FC matrix
    fc_mat = [[[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]]
    results, onsite, distances = compute_bulk_pfcs(sc, pairs, fc_mat)
    assert len(results) == 1
    assert len(onsite) == 0
    assert results[0]['species1'] == 'Li'
    assert results[0]['species2'] == 'O'
    assert results[0]['distance'] == pytest.approx(4.0 * np.sqrt(3) / 2, rel=1e-5)


def test_find_refsite_pairs_cutoff():
    sc = make_simple_supercell()
    pairs = [[1, 2]]
    fc_mat = [[[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]]
    # Reference site at the origin (where Li is)
    ref = [0.0, 0.0, 0.0]

    # With generous cutoff — both atoms found
    offsite, onsite = find_refsite_pairs(sc, pairs, fc_mat, ref, cutoff=10.0)
    assert len(offsite) == 1

    # With tiny cutoff — O is too far
    offsite, onsite = find_refsite_pairs(sc, pairs, fc_mat, ref, cutoff=0.1)
    assert len(offsite) == 0


# ---------------------------------------------------------------------------
# find_refsite_pairs — min_distance and exclude_species
# ---------------------------------------------------------------------------

def test_find_refsite_min_distance_excludes_site_atom():
    sc = make_simple_supercell()
    pairs = [[1, 2]]
    fc_mat = [np.eye(3).tolist()]
    ref = [0.0, 0.0, 0.0]  # coincides with Li (atom 1) → dist_i = 0
    offsite, _ = find_refsite_pairs(sc, pairs, fc_mat, ref,
                                     cutoff=10.0, min_distance=0.1)
    assert len(offsite) == 0


def test_find_refsite_exclude_species():
    sc = make_simple_supercell()
    pairs = [[1, 2]]
    fc_mat = [np.eye(3).tolist()]
    ref = [0.25, 0.25, 0.25]  # midpoint, both atoms well within cutoff
    offsite, _ = find_refsite_pairs(sc, pairs, fc_mat, ref,
                                     cutoff=10.0, exclude_species=['Li'])
    assert len(offsite) == 0


# ---------------------------------------------------------------------------
# unique_pfcs
# ---------------------------------------------------------------------------

def test_unique_pfcs_deduplicates_at_5dp():
    # Difference at 6th decimal → same when rounded to 5dp
    results = [
        {'atom1_idx': 1, 'atom2_idx': 2, 'species1': 'Li', 'species2': 'O',
         'distance': 3.0, 'mean_pfc': 1.000001, 'rms_pfc': 1.0},
        {'atom1_idx': 1, 'atom2_idx': 3, 'species1': 'Li', 'species2': 'O',
         'distance': 3.1, 'mean_pfc': 1.000002, 'rms_pfc': 1.0},
    ]
    df = unique_pfcs(results)
    assert len(df) == 1


def test_unique_pfcs_distinct_values():
    results = [
        {'atom1_idx': 1, 'atom2_idx': 2, 'species1': 'Li', 'species2': 'O',
         'distance': 3.0, 'mean_pfc': 1.0, 'rms_pfc': 1.0},
        {'atom1_idx': 1, 'atom2_idx': 3, 'species1': 'Li', 'species2': 'O',
         'distance': 4.0, 'mean_pfc': 2.0, 'rms_pfc': 2.0},
    ]
    df = unique_pfcs(results)
    assert len(df) == 2


def test_unique_pfcs_empty():
    assert unique_pfcs([]).empty


def test_unique_pfcs_same_value_different_species_not_merged():
    # Ge-Te and Ge-Ge bonds with identical pFC must NOT be collapsed into one row
    results = [
        {'atom1_idx': 1, 'atom2_idx': 2, 'species1': 'Ge', 'species2': 'Te',
         'distance': 3.0, 'mean_pfc': 1.0, 'rms_pfc': 1.0},
        {'atom1_idx': 2, 'atom2_idx': 3, 'species1': 'Ge', 'species2': 'Ge',
         'distance': 4.0, 'mean_pfc': 1.0, 'rms_pfc': 1.0},
    ]
    df = unique_pfcs(results)
    assert len(df) == 2
    assert set(zip(df['Atom 1'], df['Atom 2'])) == {('Ge', 'Te'), ('Ge', 'Ge')}


# ---------------------------------------------------------------------------
# match_atoms_across_structures
# ---------------------------------------------------------------------------

def _make_sc(species, positions, a=8.0):
    n = len(positions)
    return Supercell({
        'skal': 1.0,
        'lattice': [[a, 0, 0], [0, a, 0], [0, 0, a]],
        'chem_symbols': [species],
        'chem_atoms': [n],
        'positions': positions,
    })


def test_match_atoms_exact():
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    sc_a = _make_sc('Li', positions)
    sc_b = _make_sc('Li', positions)
    matches, unmatched = match_atoms_across_structures(sc_a, sc_b, 'Li')
    assert matches == {1: 1, 2: 2}
    assert unmatched == []


def test_match_atoms_small_shift_within_tolerance():
    sc_a = _make_sc('Li', [[0.0, 0.0, 0.0]])
    sc_b = _make_sc('Li', [[0.005, 0.0, 0.0]])  # 0.04 Å shift in 8 Å cell
    matches, unmatched = match_atoms_across_structures(sc_a, sc_b, 'Li', tolerance=0.3)
    assert 1 in matches
    assert unmatched == []


def test_match_atoms_large_shift_beyond_tolerance():
    sc_a = _make_sc('Li', [[0.0, 0.0, 0.0]])
    sc_b = _make_sc('Li', [[0.1, 0.0, 0.0]])   # 0.8 Å shift in 8 Å cell
    matches, unmatched = match_atoms_across_structures(sc_a, sc_b, 'Li', tolerance=0.3)
    assert matches == {}
    assert 1 in unmatched


def test_match_atoms_absent_species_in_b():
    sc_a = _make_sc('Li', [[0.0, 0.0, 0.0]])
    sc_b = _make_sc('O',  [[0.0, 0.0, 0.0]])  # no Li in sc_b
    matches, unmatched = match_atoms_across_structures(sc_a, sc_b, 'Li')
    assert matches == {}
    assert 1 in unmatched


def test_match_atoms_no_double_match():
    # Two sc_a atoms near the same sc_b atom — greedy: only one can match
    sc_a = _make_sc('Li', [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    sc_b = _make_sc('Li', [[0.0, 0.0, 0.0]])
    matches, unmatched = match_atoms_across_structures(sc_a, sc_b, 'Li', tolerance=1.0)
    assert len(matches) == 1
    assert len(unmatched) == 1


# ---------------------------------------------------------------------------
# match_fc_pairs
# ---------------------------------------------------------------------------

def _refsite_result(a1, a2, sp1, sp2, dist, pfc, ref_dist=1.0):
    return {
        'atom1_idx': a1, 'atom2_idx': a2,
        'species1': sp1, 'species2': sp2,
        'distance': dist, 'mean_pfc': pfc,
        'atom1_ref_dist': ref_dist,
    }


def test_match_fc_pairs_basic_delta():
    sc = make_simple_supercell()
    results_a = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    results_b = [_refsite_result(1, 2, 'Li', 'O', 3.0, 2.0)]
    matched, _, _ = match_fc_pairs(results_a, results_b, {1: 1, 2: 2}, sc)
    assert len(matched) == 1
    assert matched[0]['delta_pfc'] == pytest.approx(1.0)


def test_match_fc_pairs_atom_not_in_matches_silently_skipped():
    sc = make_simple_supercell()
    results_a = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    results_b = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    # atom 2 has no entry in atom_matches
    matched, unmatched_a, _ = match_fc_pairs(results_a, results_b, {1: 1}, sc)
    assert len(matched) == 0
    assert len(unmatched_a) == 0  # silently skipped, not added to unmatched_a


def test_match_fc_pairs_no_counterpart_in_b():
    sc = make_simple_supercell()
    results_a = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    results_b = []  # empty — no counterpart
    matched, unmatched_a, _ = match_fc_pairs(results_a, results_b, {1: 1, 2: 2}, sc)
    assert len(matched) == 0
    assert len(unmatched_a) == 1


def test_match_fc_pairs_unmatched_b():
    sc = make_simple_supercell()
    results_a = []
    results_b = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    _, _, unmatched_b = match_fc_pairs(results_a, results_b, {}, sc)
    assert len(unmatched_b) == 1


# ---------------------------------------------------------------------------
# stiffness_shift_from_pairs
# ---------------------------------------------------------------------------

def _matched_pair(a1a, a2a, a1b, a2b, pfc_a, pfc_b):
    return {
        'atom1_idx_a': a1a, 'atom2_idx_a': a2a,
        'atom1_idx_b': a1b, 'atom2_idx_b': a2b,
        'species1': 'Li', 'species2': 'O',
        'distance_a': 3.0, 'distance_b': 3.0,
        'atom1_ref_dist_a': 1.0, 'atom1_ref_dist_b': 1.0,
        'mean_pfc_a': pfc_a, 'mean_pfc_b': pfc_b,
        'delta_pfc': pfc_b - pfc_a,
    }


def test_stiffness_shift_total():
    pairs = [_matched_pair(1, 2, 1, 2, 1.0, 2.0),
             _matched_pair(1, 3, 1, 3, 0.5, 1.0)]
    df, total = stiffness_shift_from_pairs(pairs)
    assert total == pytest.approx(1.5)
    assert len(df) == 2


def test_stiffness_shift_empty():
    df, total = stiffness_shift_from_pairs([])
    assert total == pytest.approx(0.0)
    assert df.empty


# ---------------------------------------------------------------------------
# fallback_equal_count_shift
# ---------------------------------------------------------------------------

def test_fallback_equal_count_delta():
    results_a = [_refsite_result(1, 2, 'Li', 'O', 3.0, 1.0)]
    results_b = [_refsite_result(1, 2, 'Li', 'O', 3.0, 2.0)]
    df, total, n = fallback_equal_count_shift(results_a, results_b)
    assert n == 1
    assert total == pytest.approx(1.0)


def test_fallback_truncates_to_shorter():
    results_a = [
        _refsite_result(1, 2, 'Li', 'O', 3.0, 1.0),
        _refsite_result(1, 3, 'Li', 'O', 4.0, 1.0),
    ]
    results_b = [_refsite_result(1, 2, 'Li', 'O', 3.0, 2.0)]
    df, total, n = fallback_equal_count_shift(results_a, results_b)
    assert n == 1
    assert len(df) == 1
