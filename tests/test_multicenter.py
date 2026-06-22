"""
Tests for betapy.core.multicenter

Run with:  python -m pytest tests/
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from betapy.core.multicenter import (
    detect_anomalous_pairs,
    _map_sc_atom_to_poscar,
    _build_neighbor_lookup,
    _grow_chain,
    find_chains,
    format_cobi_directive,
    append_cobi_directives,
    _bonded_nn_distances,
    suggest_cobi_directives,
)
from betapy.core.structure import Supercell
from betapy.core.lobster import _parse_poscar_lobster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pair(sp1, sp2, dist, pfc, atom1_idx=1, atom2_idx=2, direction=None):
    return {
        'atom1_idx': atom1_idx,
        'atom2_idx': atom2_idx,
        'species1':  sp1,
        'species2':  sp2,
        'distance':  dist,
        'mean_pfc':  pfc,
        'direction': direction or [1., 0., 0.],
    }


def write_poscar(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def _supercell_from_dict(d):
    return Supercell(d)


# Minimal 2-atom POSCAR (A at 0,0,0; B at 0.5,0,0) in 4 Å cubic cell
POSCAR_2ATOM = """\
2-atom cell
1.0
  4.0  0.0  0.0
  0.0  4.0  0.0
  0.0  0.0  4.0
A B
1 1
Direct
  0.00000  0.00000  0.00000
  0.50000  0.00000  0.00000
"""

# 2×1×1 supercell of the 2-atom POSCAR (8 Å along x).
# PHONOPY groups species together, so all A's come first, then all B's:
#   atom 1: A at frac (0.000, 0, 0)  → POSCAR atom1 (A), cell [0,0,0]
#   atom 2: A at frac (0.500, 0, 0)  → POSCAR atom1 (A), cell [1,0,0]
#   atom 3: B at frac (0.250, 0, 0)  → POSCAR atom2 (B), cell [0,0,0]
#   atom 4: B at frac (0.750, 0, 0)  → POSCAR atom2 (B), cell [1,0,0]
SPOSCAR_2X1X1_DICT = {
    'skal':         1.0,
    'lattice':      [[8.0, 0.0, 0.0],
                     [0.0, 4.0, 0.0],
                     [0.0, 0.0, 4.0]],
    'chem_symbols': ['A', 'B'],
    'chem_atoms':   [2, 2],
    'positions':    [
        [0.000, 0.0, 0.0],   # A, sc_idx=1 → atom1 (A), cell [0,0,0]
        [0.500, 0.0, 0.0],   # A, sc_idx=2 → atom1 (A), cell [1,0,0]
        [0.250, 0.0, 0.0],   # B, sc_idx=3 → atom2 (B), cell [0,0,0]
        [0.750, 0.0, 0.0],   # B, sc_idx=4 → atom2 (B), cell [1,0,0]
    ],
}


# ---------------------------------------------------------------------------
# detect_anomalous_pairs
# ---------------------------------------------------------------------------

class TestDetectAnomalousPairs:

    def _badger_pairs(self, dists, anomaly_idx=None, anomaly_factor=10.0):
        """
        Build synthetic pair records following a perfect Badger decay
        Phi^{-1/3} = 0.5*r + 0.1  →  Phi = (0.5*r + 0.1)^{-3}.
        Inject an outlier at anomaly_idx if given.
        """
        pairs = []
        for i, d in enumerate(dists):
            pfc = (0.5 * d + 0.1) ** (-3)
            if i == anomaly_idx:
                pfc *= anomaly_factor
            pairs.append(make_pair('Te', 'Te', d, pfc,
                                   atom1_idx=i + 1, atom2_idx=i + 2))
        return pairs

    def test_flags_regression_outlier(self):
        dists = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Inject anomaly at index 3 (d=4.0): pFC × 100 → clear regression outlier.
        # max_nn_ratio=None: this test exercises regression detection, not the NN filter.
        pairs = self._badger_pairs(dists, anomaly_idx=3, anomaly_factor=100.0)
        flagged = detect_anomalous_pairs(pairs, min_pairs=4, n_sigma=2.0,
                                         value_key='mean_pfc', max_nn_ratio=None)
        assert len(flagged) == 1
        assert flagged[0]['distance'] == pytest.approx(4.0)
        assert flagged[0]['method'] == 'regression'
        assert flagged[0]['n_sigma'] > 2.0

    def test_no_flag_on_clean_decay(self):
        dists = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pairs = self._badger_pairs(dists)
        flagged = detect_anomalous_pairs(pairs, min_pairs=4, n_sigma=2.0,
                                         value_key='mean_pfc')
        assert flagged == []

    def test_monotone_fallback_flags_increase(self):
        # Only 2 pairs — falls back to monotonicity check
        pairs = [
            make_pair('I', 'I', 2.8, 10.0, atom1_idx=1, atom2_idx=2),
            make_pair('I', 'I', 5.6, 20.0, atom1_idx=1, atom2_idx=3),
        ]
        flagged = detect_anomalous_pairs(pairs, min_pairs=4, value_key='mean_pfc')
        assert len(flagged) == 1
        assert flagged[0]['method'] == 'monotone'
        assert math.isnan(flagged[0]['n_sigma'])

    def test_monotone_fallback_no_flag_on_decay(self):
        pairs = [
            make_pair('I', 'I', 2.8, 20.0, atom1_idx=1, atom2_idx=2),
            make_pair('I', 'I', 5.6,  5.0, atom1_idx=1, atom2_idx=3),
        ]
        assert detect_anomalous_pairs(pairs, min_pairs=4, value_key='mean_pfc') == []

    def test_different_species_pairs_independent(self):
        # Ge-Te clean, Te-Te anomalous — only Te-Te should be flagged.
        # max_nn_ratio=None: tests species independence, not the NN ratio filter.
        dists = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        gete = [make_pair('Ge', 'Te', d, (0.5*d + 0.1)**(-3),
                          atom1_idx=i+1, atom2_idx=i+10)
                for i, d in enumerate(dists)]
        tete = self._badger_pairs(dists, anomaly_idx=4, anomaly_factor=200.0)
        flagged = detect_anomalous_pairs(gete + tete, n_sigma=2.0, value_key='mean_pfc',
                                         max_nn_ratio=None)
        species_pairs = {(f['species1'], f['species2']) for f in flagged}
        # (Te, Te) may appear as (Te, Te) since both species are Te
        assert all(sp in {('Te', 'Te')} for sp in species_pairs)
        assert len(flagged) >= 1

    def test_skips_zero_pfc_pairs(self):
        pairs = [
            make_pair('X', 'Y', 1.0, 0.0, atom1_idx=1, atom2_idx=2),  # zero — skip
            make_pair('X', 'Y', 2.0, 5.0, atom1_idx=1, atom2_idx=3),
            make_pair('X', 'Y', 3.0, 2.0, atom1_idx=1, atom2_idx=4),
            make_pair('X', 'Y', 4.0, 1.0, atom1_idx=1, atom2_idx=5),
            make_pair('X', 'Y', 5.0, 0.5, atom1_idx=1, atom2_idx=6),
        ]
        # Should not crash; 4 valid pairs available for regression
        detect_anomalous_pairs(pairs, min_pairs=4, value_key='mean_pfc')

    def test_robust_to_multiple_outliers(self):
        # Theil-Sen baseline is unaffected by anomalous pairs — both outliers
        # should be flagged even when they represent 25 % of the data.
        # method='joint' intercept (median of y_i - slope*x_i) is used so
        # anomalous pairs cannot bias the intercept via median(y).
        # max_nn_ratio=None: tests regression robustness, not the NN ratio filter.
        dists = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        pairs = self._badger_pairs(dists)
        pairs[2]['mean_pfc'] *= 100.0   # d=3.0 — anomalous
        pairs[6]['mean_pfc'] *= 100.0   # d=7.0 — anomalous
        flagged = detect_anomalous_pairs(pairs, min_pairs=4, n_sigma=2.0,
                                         value_key='mean_pfc', max_nn_ratio=None)
        flagged_dists = {f['distance'] for f in flagged}
        assert 3.0 in flagged_dists
        assert 7.0 in flagged_dists

    def test_max_nn_ratio_filters_far_pairs(self):
        # Anomaly at d=3.0 is 3.0× the NN (d=1.0).
        # With max_nn_ratio=2.5, it must be suppressed; with None it must be flagged.
        dists = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pairs = self._badger_pairs(dists, anomaly_idx=2, anomaly_factor=100.0)
        flagged_no_filter = detect_anomalous_pairs(pairs, min_pairs=4, n_sigma=2.0,
                                                    value_key='mean_pfc',
                                                    max_nn_ratio=None)
        flagged_filtered  = detect_anomalous_pairs(pairs, min_pairs=4, n_sigma=2.0,
                                                    value_key='mean_pfc',
                                                    max_nn_ratio=2.5)
        assert any(f['distance'] == pytest.approx(3.0) for f in flagged_no_filter), \
            "anomaly at 3×NN must be flagged when filter is disabled"
        assert not any(f['distance'] == pytest.approx(3.0) for f in flagged_filtered), \
            "anomaly at 3×NN must be suppressed by max_nn_ratio=2.5"


# ---------------------------------------------------------------------------
# _map_sc_atom_to_poscar
# ---------------------------------------------------------------------------

class TestMapScAtomToPostcar:

    @pytest.fixture
    def poscar_path(self):
        return write_poscar(POSCAR_2ATOM)

    @pytest.fixture
    def supercell(self):
        return _supercell_from_dict(SPOSCAR_2X1X1_DICT)

    @pytest.fixture
    def lob_poscar(self, poscar_path):
        return _parse_poscar_lobster(poscar_path)

    def test_atom1_maps_to_atom1_cell_000(self, supercell, lob_poscar):
        # A at frac (0.0) → POSCAR atom1 (A), cell [0,0,0]
        label, cell = _map_sc_atom_to_poscar(1, supercell, lob_poscar)
        assert label == 'atom1'
        assert cell == [0, 0, 0]

    def test_atom2_maps_to_atom1_cell_100(self, supercell, lob_poscar):
        # A at frac (0.5) → POSCAR atom1 (A), cell [1,0,0]
        label, cell = _map_sc_atom_to_poscar(2, supercell, lob_poscar)
        assert label == 'atom1'
        assert cell == [1, 0, 0]

    def test_atom3_maps_to_atom2_cell_000(self, supercell, lob_poscar):
        # B at frac (0.25) → POSCAR atom2 (B), cell [0,0,0]
        label, cell = _map_sc_atom_to_poscar(3, supercell, lob_poscar)
        assert label == 'atom2'
        assert cell == [0, 0, 0]

    def test_atom4_maps_to_atom2_cell_100(self, supercell, lob_poscar):
        # B at frac (0.75) → POSCAR atom2 (B), cell [1,0,0]
        label, cell = _map_sc_atom_to_poscar(4, supercell, lob_poscar)
        assert label == 'atom2'
        assert cell == [1, 0, 0]

    def test_raises_on_species_mismatch(self, supercell):
        # POSCAR with A and B swapped in position — geometry matches but
        # species are wrong; the check should catch this before LOBSTER
        # silently computes the wrong multicenter interaction.
        swapped = write_poscar("""\
swapped
1.0
  4.0  0.0  0.0
  0.0  4.0  0.0
  0.0  0.0  4.0
B A
1 1
Direct
  0.00000  0.00000  0.00000
  0.50000  0.00000  0.00000
""")
        lob = _parse_poscar_lobster(swapped)
        with pytest.raises(ValueError, match='(?i)species'):
            _map_sc_atom_to_poscar(1, supercell, lob)

    def test_raises_on_incommensurate_poscar(self, supercell):
        # 3 Å POSCAR — not commensurate with the 8 Å supercell.
        # Atom 2 (B) in SPOSCAR is at Cartesian (2, 0, 0); in the 3 Å POSCAR
        # that is fractional 0.667, which is 2.0 Å from the only atom at (0,0,0)
        # — well beyond the default tolerance.
        bad = write_poscar("""\
bad
1.0
  3.0  0.0  0.0
  0.0  3.0  0.0
  0.0  0.0  3.0
A
1
Direct
  0.0  0.0  0.0
""")
        lob = _parse_poscar_lobster(bad)
        with pytest.raises(ValueError, match='POSCAR'):
            _map_sc_atom_to_poscar(2, supercell, lob)


# ---------------------------------------------------------------------------
# _build_neighbor_lookup / _grow_chain
# ---------------------------------------------------------------------------

class TestGrowChain:
    """
    Synthetic linear chain: atoms 1,2,3 at x=0,1,2 in a 10 Å box.
    """

    @pytest.fixture
    def sc(self):
        return _supercell_from_dict({
            'skal':         1.0,
            'lattice':      [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]],
            'chem_symbols': ['A'],
            'chem_atoms':   [3],
            'positions':    [[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0],
                             [0.2, 0.0, 0.0]],
        })

    @pytest.fixture
    def bulk_results(self):
        # Pairs in the synthetic chain: 1-2 at 1 Å and 2-3 at 1 Å, direction [1,0,0]
        return [
            {'atom1_idx': 1, 'atom2_idx': 2, 'distance': 1.0,
             'direction': [1., 0., 0.],
             'species1': 'A', 'species2': 'A', 'mean_pfc': 1.0},
            {'atom1_idx': 2, 'atom2_idx': 3, 'distance': 1.0,
             'direction': [1., 0., 0.],
             'species1': 'A', 'species2': 'A', 'mean_pfc': 0.5},
        ]

    def test_neighbor_lookup_both_directions(self, bulk_results):
        nb = _build_neighbor_lookup(bulk_results, bond_cutoff=1.5)
        # atom 1 has neighbour 2 in [1,0,0]
        assert any(n['idx'] == 2 for n in nb.get(1, []))
        # atom 2 has neighbour 1 in [-1,0,0]
        nb_2 = {n['idx']: n for n in nb.get(2, [])}
        assert 1 in nb_2
        assert nb_2[1]['dir'] == pytest.approx([-1., 0., 0.])

    def test_grow_chain_full_length(self, sc, bulk_results):
        nb = _build_neighbor_lookup(bulk_results, bond_cutoff=1.5)
        # Start at atom 1, direction [1,0,0], reliability_limit 5 Å, max_order 5
        chain = _grow_chain(1, [1., 0., 0.], nb,
                            min_cos=np.cos(np.radians(30.)),
                            max_order=5, reliability_limit=5.0)
        assert chain == [1, 2, 3]

    def test_grow_chain_stops_at_reliability_limit(self, sc, bulk_results):
        nb = _build_neighbor_lookup(bulk_results, bond_cutoff=1.5)
        # Cumulative limit 1.5 Å: 1→2 costs 1 Å (ok), 2→3 would cost 2 Å total (>1.5)
        chain = _grow_chain(1, [1., 0., 0.], nb,
                            min_cos=np.cos(np.radians(30.)),
                            max_order=5, reliability_limit=1.5)
        assert chain == [1, 2]

    def test_grow_chain_respects_angle(self, sc, bulk_results):
        # Perpendicular neighbour — should be rejected at min_angle=150
        bulk = [
            {'atom1_idx': 1, 'atom2_idx': 2, 'distance': 1.0,
             'direction': [1., 0., 0.], 'species1': 'A', 'species2': 'A',
             'mean_pfc': 1.0},
            # atom 2's only forward neighbour is perpendicular
            {'atom1_idx': 2, 'atom2_idx': 3, 'distance': 1.0,
             'direction': [0., 1., 0.], 'species1': 'A', 'species2': 'A',
             'mean_pfc': 0.5},
        ]
        nb = _build_neighbor_lookup(bulk, bond_cutoff=1.5)
        chain = _grow_chain(1, [1., 0., 0.], nb,
                            min_cos=np.cos(np.radians(30.)),  # cos(30°) ≈ 0.866
                            max_order=5, reliability_limit=5.0)
        # direction from atom2 to atom3 is [0,1,0]; dot with [1,0,0] = 0 < 0.866
        assert chain == [1, 2]


# ---------------------------------------------------------------------------
# find_chains
# ---------------------------------------------------------------------------

class TestFindChains:

    @pytest.fixture
    def linear_sc(self):
        """5 atoms in a line at x=0,1,2,3,4 in a 20 Å box."""
        return _supercell_from_dict({
            'skal':         1.0,
            'lattice':      [[20., 0., 0.], [0., 20., 0.], [0., 0., 20.]],
            'chem_symbols': ['A'],
            'chem_atoms':   [5],
            'positions':    [[0.00, 0., 0.],
                             [0.05, 0., 0.],
                             [0.10, 0., 0.],
                             [0.15, 0., 0.],
                             [0.20, 0., 0.]],
        })

    @pytest.fixture
    def linear_bulk(self):
        """All consecutive pairs at 1 Å in [1,0,0]."""
        pairs = []
        for i in range(1, 5):
            pairs.append({'atom1_idx': i, 'atom2_idx': i + 1, 'distance': 1.0,
                          'direction': [1., 0., 0.],
                          'species1': 'A', 'species2': 'A', 'mean_pfc': 1.0})
        return pairs

    def test_finds_three_center_chain(self, linear_sc, linear_bulk):
        # Flag pair 1-3 (distance 2, direction [1,0,0]) as trigger
        flagged = [{'atom1_idx': 1, 'atom2_idx': 3, 'distance': 2.0,
                    'direction': [1., 0., 0.],
                    'species1': 'A', 'species2': 'A', 'mean_pfc': 5.0}]
        chains = find_chains(flagged, linear_sc, linear_bulk,
                             min_angle_deg=150., max_order=3, bond_cutoff=1.5)
        assert len(chains) >= 1
        assert chains[0]['full_chain'] == [1, 2, 3]

    def test_sub_chains_all_orders(self, linear_sc, linear_bulk):
        flagged = [{'atom1_idx': 1, 'atom2_idx': 5, 'distance': 4.0,
                    'direction': [1., 0., 0.],
                    'species1': 'A', 'species2': 'A', 'mean_pfc': 5.0}]
        chains = find_chains(flagged, linear_sc, linear_bulk,
                             min_angle_deg=150., max_order=5, bond_cutoff=1.5)
        assert len(chains) == 1
        chain = chains[0]
        assert chain['full_chain'] == [1, 2, 3, 4, 5]
        orders = [s['order'] for s in chain['sub_chains']]
        # Must include 3-center, 4-center, and 5-center entries
        assert 3 in orders
        assert 4 in orders
        assert 5 in orders
        # Exactly one 5-center (the full chain)
        assert orders.count(5) == 1
        # Three 3-center sub-sequences: [1,2,3], [2,3,4], [3,4,5]
        assert orders.count(3) == 3

    def test_reliability_limit_caps_chain(self, linear_sc, linear_bulk):
        # With reliability_limit < 3 Å, chain cannot grow past 3 atoms
        flagged = [{'atom1_idx': 1, 'atom2_idx': 5, 'distance': 4.0,
                    'direction': [1., 0., 0.],
                    'species1': 'A', 'species2': 'A', 'mean_pfc': 5.0}]
        # Force short reliability by shrinking supercell lattice via max_order
        # Instead: rely on the actual reliability_limit from the supercell.
        # The supercell is 20 Å, so reliability_limit = 10 Å — no cap here.
        # Explicitly test via a small supercell.
        small_sc = _supercell_from_dict({
            'skal':         1.0,
            'lattice':      [[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
            'chem_symbols': ['A'],
            'chem_atoms':   [5],
            'positions':    [[0.00, 0., 0.],
                             [0.25, 0., 0.],
                             [0.50, 0., 0.],
                             [0.75, 0., 0.],
                             [1.00, 0., 0.]],   # same as 0.0 by PBC
        })
        # reliability_limit = 4/2 = 2 Å → max 3 atoms (2×1 Å steps)
        chains = find_chains(flagged, small_sc, linear_bulk,
                             min_angle_deg=150., max_order=5, bond_cutoff=1.5)
        if chains:
            assert chains[0]['total_distance'] <= 2.0 + 1e-6

    def test_no_chain_shorter_than_3(self, linear_sc, linear_bulk):
        # Trigger pair that can only reach 2 atoms (direction away from chain)
        flagged = [{'atom1_idx': 1, 'atom2_idx': 2, 'distance': 1.0,
                    'direction': [-1., 0., 0.],   # away from chain
                    'species1': 'A', 'species2': 'A', 'mean_pfc': 5.0}]
        chains = find_chains(flagged, linear_sc, linear_bulk,
                             min_angle_deg=150., max_order=5, bond_cutoff=1.5)
        assert chains == []


# ---------------------------------------------------------------------------
# format_cobi_directive
# ---------------------------------------------------------------------------

class TestFormatCobiDirective:

    @pytest.fixture
    def poscar_path(self):
        return write_poscar(POSCAR_2ATOM)

    @pytest.fixture
    def sc(self):
        return _supercell_from_dict(SPOSCAR_2X1X1_DICT)

    @pytest.fixture
    def lob_poscar(self, poscar_path):
        return _parse_poscar_lobster(poscar_path)

    def test_cross_cell_chain_has_cell_tag(self, sc, lob_poscar):
        # Chain: atom1(A,cell000) → atom3(B,cell000) → atom2(A,cell100)
        # atom2 is a periodic image — cell tag must appear for it.
        directive = format_cobi_directive([1, 3, 2], sc, lob_poscar)
        assert directive.startswith('cobiBetween')
        assert 'atom1' in directive
        assert 'atom2' in directive
        assert tokens[1] == 'atom1' if (tokens := directive.split()) else True
        assert 'cell' in directive
        # 2×1×1 supercell: minimum-image of [1,0,0] is [-1,0,0]
        assert '-1 0 0' in directive or '1 0 0' in directive

    def test_first_atom_has_no_cell_tag(self, sc, lob_poscar):
        # atom1(A) and atom3(B) are both in cell [0,0,0] — reference atom gets no cell tag
        directive = format_cobi_directive([1, 3], sc, lob_poscar)
        tokens = directive.split()
        assert tokens[1] == 'atom1'
        assert tokens[2] != 'cell'

    def test_same_cell_atoms_no_cell_tag(self, sc, lob_poscar):
        # atom1(A) and atom3(B) are both in POSCAR cell [0,0,0] → no cell tag at all
        directive = format_cobi_directive([1, 3], sc, lob_poscar)
        assert 'cell' not in directive


# ---------------------------------------------------------------------------
# append_cobi_directives
# ---------------------------------------------------------------------------

class TestAppendCobiDirectives:

    def test_appends_new_directives(self, tmp_path):
        f = tmp_path / 'lobsterin'
        f.write_text('COHPstartEnergy -20\n')
        n = append_cobi_directives(f, ['cobiBetween Ge1 Te5', 'cobiBetween Te5 Ge2 Te8'])
        assert n == 2
        content = f.read_text()
        assert 'cobiBetween Ge1 Te5' in content
        assert 'cobiBetween Te5 Ge2 Te8' in content

    def test_skips_existing_directives(self, tmp_path):
        f = tmp_path / 'lobsterin'
        f.write_text('cobiBetween Ge1 Te5\n')
        n = append_cobi_directives(f, ['cobiBetween Ge1 Te5'])
        assert n == 0
        assert f.read_text().count('cobiBetween Ge1 Te5') == 1

    def test_partial_skip(self, tmp_path):
        f = tmp_path / 'lobsterin'
        f.write_text('cobiBetween Ge1 Te5\n')
        n = append_cobi_directives(f, ['cobiBetween Ge1 Te5', 'cobiBetween Te5 Ge2'])
        assert n == 1
        assert 'cobiBetween Te5 Ge2' in f.read_text()


# ---------------------------------------------------------------------------
# _bonded_nn_distances — covalent-radius screen for the species-pair NN
# reference used by max_nn_ratio
# ---------------------------------------------------------------------------

class TestBondedNnDistances:

    def _records(self, sp1, sp2, dists):
        return [
            {'atom1_idx': i + 1, 'atom2_idx': i + 2, 'species1': sp1,
             'species2': sp2, 'distance': d, 'phi_iso': 1.0}
            for i, d in enumerate(dists)
        ]

    def test_real_bond_kept_at_face_value(self):
        # S-Zn in ZnS: 2.32 A vs covalent sum 2.27 A (ratio 1.02) — a real bond.
        recs = self._records('S', 'Zn', [2.32, 4.6])
        nn = _bonded_nn_distances(recs, bond_ratio_tol=1.4)
        assert nn[('S', 'Zn')] == pytest.approx(2.32)

    def test_non_bonded_same_species_pair_disqualified(self):
        # S-S in zincblende ZnS: the only shell (3.79 A) is the 2nd-coordination
        # -shell distance mediated through Zn, not a direct bond (covalent sum
        # 2.10 A, ratio 1.81). Must be disqualified (0.0), not merely recorded.
        recs = self._records('S', 'S', [3.79, 7.58])
        nn = _bonded_nn_distances(recs, bond_ratio_tol=1.4)
        assert nn[('S', 'S')] == 0.0

    def test_weak_secondary_bond_within_tolerance_is_kept(self):
        # Te-Te across the Sb2Te3 van der Waals gap: 3.60 A vs covalent sum
        # 2.76 A (ratio 1.30) — weaker than a primary bond but genuinely used
        # as a chain-hop in the real detection pipeline; must survive the screen.
        recs = self._records('Te', 'Te', [3.60, 7.20])
        nn = _bonded_nn_distances(recs, bond_ratio_tol=1.4)
        assert nn[('Te', 'Te')] == pytest.approx(3.60)

    def test_bond_ratio_tol_none_disables_the_screen(self):
        recs = self._records('S', 'S', [3.79])
        nn = _bonded_nn_distances(recs, bond_ratio_tol=None)
        assert nn[('S', 'S')] == pytest.approx(3.79)

    def test_bond_ratio_tol_zero_disables_the_screen(self):
        recs = self._records('S', 'S', [3.79])
        nn = _bonded_nn_distances(recs, bond_ratio_tol=0)
        assert nn[('S', 'S')] == pytest.approx(3.79)


# ---------------------------------------------------------------------------
# suggest_cobi_directives — end-to-end regression test for the ZnS-style
# single-sublattice false positive
# ---------------------------------------------------------------------------

class TestSuggestCobiDirectivesBondRatioTol:
    """
    Reproduces the ZnS false positive in miniature: a same-species pair (here
    'S') with no real direct bond at any observed distance must not seed a
    "chain" that just walks the lattice's own periodic translation vector.

    In real ZnS the only S-S shell (3.79 A) is the 2nd-coordination-shell
    distance mediated through Zn; here 7 'S' atoms spaced 4 A apart on a line
    play the same role, with the first-shell pFC boosted so it gets flagged —
    mirroring the real data, where the shortest S-S shell is flagged at 3.7
    sigma yet is not a bond.
    """

    @pytest.fixture
    def marching_sc(self):
        n, spacing = 7, 4.0
        return _supercell_from_dict({
            'skal':         1.0,
            'lattice':      [[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]],
            'chem_symbols': ['S'],
            'chem_atoms':   [n],
            'positions':    [[i * spacing / 100.0, 0., 0.] for i in range(n)],
        })

    def _marching_bulk(self, n=7, spacing=4.0, anomaly_factor=4.0):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d   = (j - i) * spacing
                pfc = (0.3 * d + 0.2) ** (-3)
                if j - i == 1:
                    pfc *= anomaly_factor
                pairs.append({
                    'atom1_idx': i + 1, 'atom2_idx': j + 1,
                    'distance': d, 'direction': [1., 0., 0.],
                    'species1': 'S', 'species2': 'S',
                    'mean_pfc': pfc, 'phi_l': pfc, 'phi_t': pfc,
                })
        return pairs

    def test_bond_ratio_tol_blocks_single_sublattice_chain(self, marching_sc):
        bulk = self._marching_bulk()
        result = suggest_cobi_directives(
            bulk, marching_sc, poscar_lobster_path=None,
            n_sigma=1.5, max_order=4, min_angle_deg=150.0, bond_cutoff=4.5,
            _skip_symmetry_expand=True)
        assert result['flagged_pairs'], 'fixture must actually trigger a flag'
        assert result['chains'] == [], (
            "S-S has no real bond at any observed distance (covalent radius "
            "sum ~2.1 A vs the 4.0 A shell) — must not chain by walking the "
            "periodic translation vector")

    def test_disabling_bond_ratio_tol_reproduces_the_bug(self, marching_sc):
        # Confirms the fixture is meaningful: without the covalent-radius
        # screen, the same data does produce the spurious marching chain.
        bulk = self._marching_bulk()
        result = suggest_cobi_directives(
            bulk, marching_sc, poscar_lobster_path=None,
            n_sigma=1.5, max_order=4, min_angle_deg=150.0, bond_cutoff=4.5,
            bond_ratio_tol=0, _skip_symmetry_expand=True)
        assert len(result['chains']) > 0
        assert set(result['chains'][0]['species_chain']) == {'S'}
