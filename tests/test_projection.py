"""
Tests for betapy.core.projection and betapy.core.structure

Run with:  python -m pytest tests/
"""

import pytest
import numpy as np

from betapy.core.structure import Supercell
from betapy.core.projection import (
    compute_bulk_pfcs, find_refsite_pairs, unique_pfcs,
    match_fc_pairs_direct, stiffness_shift_from_pairs,
    _project_fc_lt,
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
# match_fc_pairs_direct
# ---------------------------------------------------------------------------

def _make_refsite_result(sc, atom1, atom2, refsite, pfc):
    """Build a result dict as find_refsite_pairs would produce."""
    return {
        'atom1_idx':      atom1,
        'atom2_idx':      atom2,
        'species1':       sc.species(atom1),
        'species2':       sc.species(atom2),
        'atom1_ref_dist': sc.distance_to_point(atom1, refsite),
        'distance':       sc.atom_distance(atom1, atom2),
        'mean_pfc':       pfc,
        'rms_pfc':        pfc,
    }


def test_match_fc_pairs_direct_basic():
    """Identical pair in A and B matches; delta_pfc is the difference."""
    sc = make_simple_supercell()
    ref = [0.25, 0.25, 0.25]
    results_a = [_make_refsite_result(sc, 1, 2, ref, 1.0)]
    results_b = [_make_refsite_result(sc, 1, 2, ref, 2.5)]

    matched, ua, ub = match_fc_pairs_direct(
        results_a, results_b, sc, sc, ref, ref, tol=0.3
    )
    assert len(matched) == 1
    assert len(ua) == 0
    assert len(ub) == 0
    assert matched[0]['delta_pfc'] == pytest.approx(1.5)


def test_match_fc_pairs_direct_empty_inputs():
    sc = make_simple_supercell()
    ref = [0.0, 0.0, 0.0]
    matched, ua, ub = match_fc_pairs_direct([], [], sc, sc, ref, ref)
    assert matched == []
    assert ua == []
    assert ub == []


def test_match_fc_pairs_direct_no_counterpart_in_b():
    """A pair with no equivalent in B goes to unmatched_a."""
    sc = make_simple_supercell()
    ref = [0.25, 0.25, 0.25]
    results_a = [_make_refsite_result(sc, 1, 2, ref, 1.0)]
    matched, ua, ub = match_fc_pairs_direct(
        results_a, [], sc, sc, ref, ref, tol=0.3
    )
    assert len(matched) == 0
    assert len(ua) == 1


def test_match_fc_pairs_direct_origin_shifted():
    """
    Pairs should match even when refsite positions differ by (0.5, 0.5, 0.5)
    in fractional space — this is the pnnm case where A and B use different
    crystallographic origins.
    """
    # Structure A: Li at [0.1, 0, 0], O at [0.4, 0, 0]; refsite near origin
    sc_a = Supercell({
        'skal': 1.0,
        'lattice': [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        'chem_symbols': ['Li', 'O'],
        'chem_atoms': [1, 1],
        'positions': [[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]],
    })
    refsite_a = [0.0, 0.0, 0.0]

    # Structure B: same geometry but with origin shifted by 0.5 along x.
    # Li at [0.6, 0, 0], O at [0.9, 0, 0]; refsite at [0.5, 0, 0].
    # Local displacements from refsite are identical to A.
    sc_b = Supercell({
        'skal': 1.0,
        'lattice': [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        'chem_symbols': ['Li', 'O'],
        'chem_atoms': [1, 1],
        'positions': [[0.6, 0.0, 0.0], [0.9, 0.0, 0.0]],
    })
    refsite_b = [0.5, 0.0, 0.0]

    results_a = [_make_refsite_result(sc_a, 1, 2, refsite_a, 1.5)]
    results_b = [_make_refsite_result(sc_b, 1, 2, refsite_b, 2.0)]

    matched, ua, ub = match_fc_pairs_direct(
        results_a, results_b, sc_a, sc_b, refsite_a, refsite_b, tol=0.5
    )
    assert len(matched) == 1
    assert len(ua) == 0
    assert matched[0]['delta_pfc'] == pytest.approx(0.5)


def test_match_fc_pairs_direct_tol_rejection():
    """Pairs with a fingerprint distance above tol are not matched."""
    # Two structures: same lattice but atom positions differ significantly
    sc_a = Supercell({
        'skal': 1.0,
        'lattice': [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        'chem_symbols': ['Li', 'O'],
        'chem_atoms': [1, 1],
        'positions': [[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]],
    })
    sc_b = Supercell({
        'skal': 1.0,
        'lattice': [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        'chem_symbols': ['Li', 'O'],
        'chem_atoms': [1, 1],
        'positions': [[0.3, 0.0, 0.0], [0.45, 0.0, 0.0]],  # very different positions
    })
    ref_a = [0.0, 0.0, 0.0]
    ref_b = [0.0, 0.0, 0.0]

    results_a = [_make_refsite_result(sc_a, 1, 2, ref_a, 1.0)]
    results_b = [_make_refsite_result(sc_b, 1, 2, ref_b, 2.0)]

    matched, ua, ub = match_fc_pairs_direct(
        results_a, results_b, sc_a, sc_b, ref_a, ref_b, tol=0.05
    )
    assert len(matched) == 0
    assert len(ua) == 1
    assert len(ub) == 1


# ---------------------------------------------------------------------------
# _project_fc_lt: longitudinal / transverse decomposition
# ---------------------------------------------------------------------------

def test_project_fc_lt_diagonal_along_x():
    # Bond along x; diagonal FC matrix with different eigenvalues
    fc = np.diag([3.0, 1.0, 1.0])
    e  = np.array([1.0, 0.0, 0.0])
    phi_l, phi_t = _project_fc_lt(fc, e)
    assert phi_l == pytest.approx(3.0)
    assert phi_t == pytest.approx(1.0)   # (Tr=5 - 3) / 2


def test_project_fc_lt_coulomb_ratio():
    # Pure Coulomb pair along x: Φ = diag(-2, 1, 1) * (C/d³), C/d³ = 1
    # Expected: A = phi_t / phi_l = 1 / (-2) = -0.5
    fc = np.diag([-2.0, 1.0, 1.0])
    e  = np.array([1.0, 0.0, 0.0])
    phi_l, phi_t = _project_fc_lt(fc, e)
    assert phi_l == pytest.approx(-2.0)
    assert phi_t == pytest.approx(1.0)   # (Tr=0 - (-2)) / 2
    assert phi_t / phi_l == pytest.approx(-0.5)


def test_project_fc_lt_isotropic():
    # Isotropic FC: phi_l == phi_t (A = 1), independent of bond direction
    k  = 5.0
    fc = k * np.eye(3)
    for e in [np.array([1, 0, 0]), np.array([0, 1, 0]),
              np.array([1, 1, 0]) / np.sqrt(2)]:
        phi_l, phi_t = _project_fc_lt(fc, e)
        assert phi_l == pytest.approx(k)
        assert phi_t == pytest.approx(k)


def test_project_fc_lt_transpose_invariance():
    # phi_l and phi_t must be identical for Φ and Φᵀ (no symmetrization needed)
    rng = np.random.default_rng(42)
    fc  = rng.standard_normal((3, 3))
    e   = np.array([1.0, 0.0, 0.0])
    phi_l,   phi_t   = _project_fc_lt(fc,   e)
    phi_l_T, phi_t_T = _project_fc_lt(fc.T, e)
    assert phi_l == pytest.approx(phi_l_T)
    assert phi_t == pytest.approx(phi_t_T)


def test_compute_bulk_pfcs_offsite_has_phi_lt():
    sc = make_simple_supercell()
    pairs  = [[1, 2]]
    fc_mat = [np.diag([-2.0, 1.0, 1.0]).tolist()]
    results, _, _ = compute_bulk_pfcs(sc, pairs, fc_mat, show_progress=False)
    assert 'phi_l' in results[0]
    assert 'phi_t' in results[0]
    # Existing keys must still be present
    assert 'mean_pfc' in results[0]
    assert 'rms_pfc'  in results[0]


# ---------------------------------------------------------------------------
# stiffness_shift_from_pairs
# ---------------------------------------------------------------------------

def _matched_pair(pfc_a, pfc_b):
    return {
        'atom1_idx_a': 1, 'atom2_idx_a': 2,
        'atom1_idx_b': 1, 'atom2_idx_b': 2,
        'species1': 'Li', 'species2': 'O',
        'distance_a': 3.0, 'distance_b': 3.0,
        'atom1_ref_dist_a': 1.0, 'atom1_ref_dist_b': 1.0,
        'mean_pfc_a': pfc_a, 'mean_pfc_b': pfc_b,
        'delta_pfc': pfc_b - pfc_a,
    }


def test_stiffness_shift_total():
    pairs = [_matched_pair(1.0, 2.0), _matched_pair(0.5, 1.0)]
    df, total = stiffness_shift_from_pairs(pairs)
    assert total == pytest.approx(1.5)
    assert len(df) == 2


def test_stiffness_shift_empty():
    df, total = stiffness_shift_from_pairs([])
    assert total == pytest.approx(0.0)
    assert df.empty
