"""
Tests for betapy.core.projection and betapy.core.structure

Run with:  python -m pytest tests/
"""

import pytest
import numpy as np

from betapy.core.structure import Supercell
from betapy.core.projection import compute_bulk_pfcs, find_refsite_pairs


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
