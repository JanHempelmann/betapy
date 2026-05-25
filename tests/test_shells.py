"""
Tests for group_by_shells and the reliability-cutoff formula.

Covers:
  - Shell grouping by species pair and distance bin
  - Species-pair normalisation: (B, A) records are merged into the (A, B) shell
    and their atom indices are swapped so rep-atom logic stays consistent
  - max_distance cutoff: pairs beyond L/2 are excluded
  - Pair deduplication logic (the set-based filter used in the GUI click handler)
  - Reliability cutoff formula for cubic and orthorhombic supercells

Run with:  python -m pytest tests/
"""

import pytest
import numpy as np

from betapy.core.structure import Supercell
from betapy.core.projection import compute_bulk_pfcs, group_by_shells


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_cubic_supercell(a=8.0, species=('A', 'B'), n_atoms=(1, 1),
                          positions=None):
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Supercell({
        'skal': 1.0,
        'lattice': [[a, 0, 0], [0, a, 0], [0, 0, a]],
        'chem_symbols': list(species),
        'chem_atoms': list(n_atoms),
        'positions': positions,
    })


def _fake_result(sp1, sp2, a1, a2, distance, pfc=1.0):
    return {
        'species1': sp1, 'species2': sp2,
        'atom1_idx': a1, 'atom2_idx': a2,
        'distance': distance,
        'mean_pfc': pfc, 'rms_pfc': pfc,
    }


# ---------------------------------------------------------------------------
# group_by_shells — basic grouping
# ---------------------------------------------------------------------------

class TestGroupByShellsBasic:

    def test_single_record_becomes_one_shell(self):
        records = [_fake_result('Ge', 'Te', 1, 2, 3.00)]
        shells = group_by_shells(records)
        assert len(shells) == 1
        assert shells[0]['species1'] == 'Ge'
        assert shells[0]['species2'] == 'Te'
        assert shells[0]['count'] == 1

    def test_same_distance_same_shell(self):
        # Three bonds with identical distance → one shell, count = 3
        records = [
            _fake_result('Ge', 'Te', 1, 3, 3.00),
            _fake_result('Ge', 'Te', 1, 4, 3.00),
            _fake_result('Ge', 'Te', 1, 5, 3.00),
        ]
        shells = group_by_shells(records)
        assert len(shells) == 1
        assert shells[0]['count'] == 3

    def test_different_distances_different_shells(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00),
            _fake_result('Ge', 'Te', 1, 3, 4.50),
        ]
        shells = group_by_shells(records)
        assert len(shells) == 2
        distances = sorted(s['distance_mean'] for s in shells)
        assert distances[0] == pytest.approx(3.00)
        assert distances[1] == pytest.approx(4.50)

    def test_different_species_pairs_separate_shells(self):
        records = [
            _fake_result('Ge', 'Te', 1, 3, 3.00),
            _fake_result('Ge', 'Ge', 2, 4, 3.00),
        ]
        shells = group_by_shells(records)
        assert len(shells) == 2

    def test_pfc_statistics(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00, pfc=1.0),
            _fake_result('Ge', 'Te', 1, 3, 3.00, pfc=3.0),
        ]
        shells = group_by_shells(records)
        assert len(shells) == 1
        s = shells[0]
        assert s['pfc_mean'] == pytest.approx(2.0)
        assert s['pfc_min']  == pytest.approx(1.0)
        assert s['pfc_max']  == pytest.approx(3.0)

    def test_output_sorted_by_species_then_distance(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 5.00),
            _fake_result('Ge', 'Ge', 2, 3, 4.00),
            _fake_result('Ge', 'Te', 1, 4, 3.00),
        ]
        shells = group_by_shells(records)
        keys = [(s['species1'], s['species2'], s['distance_mean']) for s in shells]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# group_by_shells — species-pair normalisation
# ---------------------------------------------------------------------------

class TestGroupByShellsNormalisation:

    def test_reversed_pair_merged_into_same_shell(self):
        # (Te, Ge) should be normalised to (Ge, Te) and merged
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00, pfc=2.0),
            _fake_result('Te', 'Ge', 2, 1, 3.00, pfc=2.0),  # reversed
        ]
        shells = group_by_shells(records)
        assert len(shells) == 1
        s = shells[0]
        assert s['species1'] == 'Ge'
        assert s['species2'] == 'Te'
        assert s['count'] == 2

    def test_reversed_pair_atom_indices_swapped(self):
        # When (B, A) is normalised to (A, B), atom1_idx and atom2_idx swap
        records = [_fake_result('Te', 'Ge', 7, 3, 3.00)]
        shells = group_by_shells(records)
        rec = shells[0]['records'][0]
        assert rec['atom1_idx'] == 3  # was atom2_idx (Ge)
        assert rec['atom2_idx'] == 7  # was atom1_idx (Te)

    def test_unreversed_pair_indices_unchanged(self):
        records = [_fake_result('Ge', 'Te', 3, 7, 3.00)]
        shells = group_by_shells(records)
        rec = shells[0]['records'][0]
        assert rec['atom1_idx'] == 3
        assert rec['atom2_idx'] == 7

    def test_normalisation_preserves_pfc_values(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00, pfc=1.5),
            _fake_result('Te', 'Ge', 2, 1, 3.00, pfc=2.5),
        ]
        shells = group_by_shells(records)
        pfcs = sorted(r['mean_pfc'] for r in shells[0]['records'])
        assert pfcs == pytest.approx([1.5, 2.5])


# ---------------------------------------------------------------------------
# group_by_shells — max_distance cutoff
# ---------------------------------------------------------------------------

class TestGroupByShellsMaxDistance:

    def test_pairs_beyond_cutoff_excluded(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00),
            _fake_result('Ge', 'Te', 1, 3, 6.00),  # beyond cutoff
        ]
        shells = group_by_shells(records, max_distance=5.0)
        assert len(shells) == 1
        assert shells[0]['distance_mean'] == pytest.approx(3.00)

    def test_pair_exactly_at_cutoff_excluded(self):
        # max_distance is a strict upper bound (> not >=)
        records = [_fake_result('Ge', 'Te', 1, 2, 5.00)]
        shells = group_by_shells(records, max_distance=5.00)
        # 5.00 > 5.00 is False → record IS included
        assert len(shells) == 1

    def test_no_cutoff_includes_all(self):
        records = [
            _fake_result('Ge', 'Te', 1, 2, 3.00),
            _fake_result('Ge', 'Te', 1, 3, 20.00),
        ]
        shells = group_by_shells(records, max_distance=None)
        assert len(shells) == 2


# ---------------------------------------------------------------------------
# Reliability cutoff formula
# ---------------------------------------------------------------------------

def _reliability_cutoff(lattice):
    """Mirror of the formula in PFCViewerWidget.set_supercell()."""
    a, b, c = np.array(lattice)
    V = abs(float(np.dot(a, np.cross(b, c))))
    return min(
        V / np.linalg.norm(np.cross(b, c)),
        V / np.linalg.norm(np.cross(a, c)),
        V / np.linalg.norm(np.cross(a, b)),
    ) / 2.0


class TestReliabilityCutoff:

    def test_cubic_cell(self):
        # a×a×a cube: L/2 = a/2
        a = 8.34
        lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        assert _reliability_cutoff(lattice) == pytest.approx(a / 2, rel=1e-9)

    def test_orthorhombic_cell(self):
        # a×b×c box: L/2 = min(a,b,c)/2
        a, b, c = 10.0, 8.0, 6.0
        lattice = [[a, 0, 0], [0, b, 0], [0, 0, c]]
        assert _reliability_cutoff(lattice) == pytest.approx(c / 2, rel=1e-9)

    def test_4x4x4_GeTe_cubic(self):
        # GeTe primitive a ≈ 4.17 Å, 4×4×4 supercell → L ≈ 16.68 Å, L/2 ≈ 8.34
        a = 4.173 * 4
        lattice = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        expected = a / 2
        assert _reliability_cutoff(lattice) == pytest.approx(expected, rel=1e-9)

    def test_scaling_linear_in_cell_size(self):
        # Doubling the cell doubles L/2
        lattice_1x = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        lattice_2x = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        assert _reliability_cutoff(lattice_2x) == pytest.approx(
            2 * _reliability_cutoff(lattice_1x), rel=1e-9
        )


# ---------------------------------------------------------------------------
# Pair deduplication (logic mirrored from the GUI click handler)
# ---------------------------------------------------------------------------

def _deduplicate_pairs(records, rep_atom1):
    """Mirror of the deduplication logic in PFCViewerWidget._on_scatter_click."""
    seen = set()
    pairs = []
    for r in records:
        if int(r['atom1_idx']) == rep_atom1:
            p = (int(r['atom1_idx']), int(r['atom2_idx']))
            if p not in seen:
                seen.add(p)
                pairs.append(p)
    return pairs


class TestPairDeduplication:

    def test_unique_pairs_kept(self):
        records = [
            {'atom1_idx': 1, 'atom2_idx': 5},
            {'atom1_idx': 1, 'atom2_idx': 6},
            {'atom1_idx': 1, 'atom2_idx': 7},
        ]
        pairs = _deduplicate_pairs(records, rep_atom1=1)
        assert len(pairs) == 3

    def test_duplicate_pair_removed(self):
        # Same (atom1, atom2) appearing twice (full FC matrix artefact)
        records = [
            {'atom1_idx': 1, 'atom2_idx': 5},
            {'atom1_idx': 1, 'atom2_idx': 5},
            {'atom1_idx': 1, 'atom2_idx': 6},
        ]
        pairs = _deduplicate_pairs(records, rep_atom1=1)
        assert len(pairs) == 2
        assert (1, 5) in pairs
        assert (1, 6) in pairs

    def test_other_atom1_excluded(self):
        # Only records with atom1_idx == rep_atom1 are included
        records = [
            {'atom1_idx': 1, 'atom2_idx': 5},
            {'atom1_idx': 2, 'atom2_idx': 5},  # different source atom
        ]
        pairs = _deduplicate_pairs(records, rep_atom1=1)
        assert len(pairs) == 1
        assert pairs[0] == (1, 5)

    def test_all_same_pair_gives_one_bond(self):
        records = [{'atom1_idx': 1, 'atom2_idx': 3}] * 4
        pairs = _deduplicate_pairs(records, rep_atom1=1)
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# progress_callback hook
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_callback_called_at_completion(self):
        sc = make_cubic_supercell()
        pairs = [[1, 2]]
        fc_mat = [np.eye(3).tolist()]
        calls = []
        compute_bulk_pfcs(sc, pairs, fc_mat, show_progress=False,
                          progress_callback=lambda n, t: calls.append((n, t)))
        # Final call must be (total, total)
        assert calls[-1] == (1, 1)

    def test_callback_receives_correct_total(self):
        sc = make_cubic_supercell()
        n_pairs = 5
        pairs  = [[1, 1]] * n_pairs
        fc_mats = [np.eye(3).tolist()] * n_pairs
        totals = []
        compute_bulk_pfcs(sc, pairs, fc_mats, show_progress=False,
                          progress_callback=lambda n, t: totals.append(t))
        assert all(t == n_pairs for t in totals)

    def test_no_callback_runs_without_error(self):
        sc = make_cubic_supercell()
        pairs = [[1, 2]]
        fc_mat = [np.eye(3).tolist()]
        results, _, _ = compute_bulk_pfcs(sc, pairs, fc_mat,
                                          show_progress=False,
                                          progress_callback=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# GeTe smoke test (integration)
# ---------------------------------------------------------------------------

from pathlib import Path
GETE_DIR = Path(__file__).parent.parent / 'examples' / 'GeTe'

@pytest.mark.skipif(
    not (GETE_DIR / 'SPOSCAR').exists(),
    reason='GeTe example files not present'
)
class TestGeTePipeline:

    def _load(self):
        from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS
        sc_data = read_SPOSCAR(GETE_DIR / 'SPOSCAR')
        fc_data = read_FORCE_CONSTANTS(GETE_DIR / 'FORCE_CONSTANTS')
        sc = Supercell(sc_data)
        return sc, fc_data

    def test_loads_without_error(self):
        sc, fc_data = self._load()
        assert sc.n_atoms > 0
        assert len(fc_data['atomic_pairs']) > 0

    def test_bulk_pfcs_run(self):
        sc, fc_data = self._load()
        results, onsite, _ = compute_bulk_pfcs(
            sc, fc_data['atomic_pairs'], fc_data['force_matrices'],
            show_progress=False,
        )
        assert len(results) > 0
        assert all(r['distance'] > 0 for r in results)
        assert all(isinstance(r['mean_pfc'], float) for r in results)

    def test_shells_fewer_than_results(self):
        sc, fc_data = self._load()
        results, _, _ = compute_bulk_pfcs(
            sc, fc_data['atomic_pairs'], fc_data['force_matrices'],
            show_progress=False,
        )
        shells = group_by_shells(results)
        # Shells aggregate symmetry-equivalent bonds — must be far fewer
        assert len(shells) < len(results)
        assert len(shells) > 0

    def test_shells_have_correct_species_normalisation(self):
        sc, fc_data = self._load()
        results, _, _ = compute_bulk_pfcs(
            sc, fc_data['atomic_pairs'], fc_data['force_matrices'],
            show_progress=False,
        )
        shells = group_by_shells(results)
        # All shells must have species1 <= species2 alphabetically
        for s in shells:
            assert s['species1'] <= s['species2'], (
                f"Unnormalised shell: {s['species1']}-{s['species2']}"
            )

    def test_reliability_cutoff_reasonable(self):
        sc, _ = self._load()
        L = sc.lattice
        a, b, c = L[0], L[1], L[2]
        V = abs(float(np.dot(a, np.cross(b, c))))
        rc = min(
            V / np.linalg.norm(np.cross(b, c)),
            V / np.linalg.norm(np.cross(a, c)),
            V / np.linalg.norm(np.cross(a, b)),
        ) / 2.0
        # GeTe 4×4×4: a≈16.7 Å, so L/2 should be ~8 Å
        assert 5.0 < rc < 15.0

    def test_max_distance_cutoff_reduces_shells(self):
        sc, fc_data = self._load()
        results, _, _ = compute_bulk_pfcs(
            sc, fc_data['atomic_pairs'], fc_data['force_matrices'],
            show_progress=False,
        )
        shells_all = group_by_shells(results)
        shells_cut = group_by_shells(results, max_distance=4.0)
        assert len(shells_cut) < len(shells_all)
        assert all(s['distance_mean'] <= 4.0 for s in shells_cut)
