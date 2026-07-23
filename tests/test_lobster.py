"""
Tests for betapy.core.lobster

Run with:  python -m pytest tests/
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from betapy.core.lobster import (
    _canonical, _parse_label,
    parse_car_header, enrich_cobicar_distances, load_car_curves,
    _parse_ilist, load_pairs, lookup,
    find_lobster_dir,
    parse_poscar_lobster, load_nccobicar_curves, load_nccobicar_orbital_curves,
    load_nc_entry_orbital_curves, entry_to_directive,
    orbital_type, group_orbital_curves_by_type, filter_orbital_curves,
    parse_ncicobi_list, lookup_ncicobi, lookup_ncicobi_record,
    is_translation_free, check_ncicobi_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_tmp(content: str, suffix: str = '') -> Path:
    f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def make_cohpcar(pair_vals) -> str:
    """
    Build a minimal COHPCAR.lobster string for testing.

    pair_vals: list of (sp1_elem, sp2_elem, distance, ef_ival)
    Returns a string with 3 energy points; Fermi level at E=0 (index 1).
    """
    n = len(pair_vals)
    lines = [
        'COHPCAR.lobster',
        f'{n + 1} 1 3 -1.0 1.0 0.0',
        'Average',
    ]
    for i, (sp1, sp2, dist, _) in enumerate(pair_vals, 1):
        lines.append(f'No.{i}:{sp1}1->{sp2}{i + 1}({dist:.5f})')
    for e_idx, energy in enumerate([-1.0, 0.0, 1.0]):
        row = [energy, 0.0, 0.0]
        for _, _, _, ef_ival in pair_vals:
            # icurve ramps linearly to ef_ival at E=0, stays flat after
            ival = ef_ival * e_idx if e_idx <= 1 else ef_ival
            row += [1.0, ival]
        lines.append(' '.join(f'{v:.6f}' for v in row))
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# _parse_label / _canonical
# ---------------------------------------------------------------------------

def test_parse_label_basic():
    assert _parse_label('Sc1') == ('Sc', 0)
    assert _parse_label('F12') == ('F', 11)


def test_parse_label_two_letter_element():
    assert _parse_label('Ge1') == ('Ge', 0)
    assert _parse_label('Te2') == ('Te', 1)


def test_parse_label_invalid():
    with pytest.raises(ValueError):
        _parse_label('not_a_label')


def test_canonical_alphabetical():
    assert _canonical('Sc', 'F') == ('F', 'Sc')
    assert _canonical('F', 'Sc') == ('F', 'Sc')


def test_canonical_same_species():
    assert _canonical('Ge', 'Ge') == ('Ge', 'Ge')


# ---------------------------------------------------------------------------
# parse_car_header — COHPCAR style (with Average, explicit distances)
# ---------------------------------------------------------------------------

COHPCAR_MINIMAL = """\
COHPCAR.lobster
3 1 5 -2.0 2.0 0.0
Average
No.1:Sc1->F2(2.01173)
No.2:Sc1->F3(2.01173)
-2.00000 0.000 0.000 0.10 0.04 0.10 0.04
-1.00000 0.100 0.100 0.20 0.12 0.20 0.12
 0.00000 0.200 0.200 0.30 0.22 0.30 0.22
 1.00000 0.100 0.300 0.20 0.32 0.20 0.32
 2.00000 0.000 0.300 0.00 0.32 0.00 0.32
"""


def test_parse_cohp_header_meta():
    path = write_tmp(COHPCAR_MINIMAL, '.lobster')
    hdr = parse_car_header(path)
    assert hdr['n_spins'] == 1
    assert hdr['n_e'] == 5
    assert hdr['e_fermi'] == pytest.approx(0.0)
    assert hdr['has_average'] is True
    assert hdr['n_pairs'] == 2


def test_parse_cohp_header_pairs():
    path = write_tmp(COHPCAR_MINIMAL, '.lobster')
    hdr = parse_car_header(path)
    assert len(hdr['pairs']) == 2
    p = hdr['pairs'][0]
    assert p['sp1'] == 'Sc'
    assert p['sp2'] == 'F'
    assert p['distance'] == pytest.approx(2.01173)
    assert p['cell1'] is None


# ---------------------------------------------------------------------------
# parse_car_header — COBICAR style (translation vectors, no Average)
# ---------------------------------------------------------------------------

COBICAR_MINIMAL = """\
COBICAR.lobster
2 1 3 -1.0 1.0 0.0
No.1:Sc1[0 0 0]->F2[0 0 0]
No.2:Sc1[0 0 0]->F3[-1 0 0]
-1.00000 0.10 0.04 0.10 0.04
 0.00000 0.20 0.22 0.20 0.22
 1.00000 0.00 0.22 0.00 0.22
"""


def test_parse_cobi_header_no_average():
    path = write_tmp(COBICAR_MINIMAL, '.lobster')
    hdr = parse_car_header(path)
    assert hdr['has_average'] is False
    assert hdr['n_pairs'] == 2


def test_parse_cobi_header_cell_vectors():
    path = write_tmp(COBICAR_MINIMAL, '.lobster')
    hdr = parse_car_header(path)
    p0 = hdr['pairs'][0]
    p1 = hdr['pairs'][1]
    assert p0['sp1'] == 'Sc'
    assert p0['sp2'] == 'F'
    assert p0['distance'] is None
    assert p0['cell1'] == [0, 0, 0]
    assert p0['cell2'] == [0, 0, 0]
    assert p1['cell2'] == [-1, 0, 0]


def test_parse_cobi_skips_three_centre():
    content = """\
COBICAR.lobster
1 1 3 -1.0 1.0 0.0
No.1:Sc1[0 0 0]->F2[0 0 0]->F3[-1 0 0]
-1.0 0.1 0.0
 0.0 0.2 0.1
 1.0 0.0 0.1
"""
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    assert hdr['pairs'] == []


# ---------------------------------------------------------------------------
# enrich_cobicar_distances
# ---------------------------------------------------------------------------

POSCAR_LOB_SIMPLE = """\
ScF
   1.0
     4.0000000   0.0000000   0.0000000
     0.0000000   4.0000000   0.0000000
     0.0000000   0.0000000   4.0000000
Sc F
1 1
Direct
  0.0000000   0.0000000   0.0000000
  0.5000000   0.0000000   0.0000000
"""


def test_enrich_cobicar_distances():
    cobicar = """\
COBICAR.lobster
1 1 3 -1.0 1.0 0.0
No.1:Sc1[0 0 0]->F2[0 0 0]
-1.0 0.1 0.04
 0.0 0.2 0.22
 1.0 0.0 0.22
"""
    car_path = write_tmp(cobicar, '.lobster')
    pos_path = write_tmp(POSCAR_LOB_SIMPLE)
    hdr = parse_car_header(car_path)
    assert hdr['pairs'][0]['distance'] is None
    enrich_cobicar_distances(hdr, pos_path)
    # Sc at (0,0,0), F at (0.5*4, 0, 0) = (2.0, 0, 0) → distance = 2.0 Å
    assert hdr['pairs'][0]['distance'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _parse_ilist / load_pairs / lookup
# ---------------------------------------------------------------------------

ICOBILIST_ONE_SHELL = """\
ICOBILIST.lobster
1 Sc1 F2 2.01173 -0.01476 0.40650 0.40650
2 Sc1 F3 2.01173 -0.01476 0.40650 0.40650
3 Sc1 F4 2.01173 -0.01476 0.40650 0.40650
"""

ICOBILIST_TWO_SHELLS = """\
ICOBILIST.lobster
1 Sc1 F2 2.01173 -0.01476 0.40650 0.40650
2 Sc1 F3 3.50000 -0.00100 0.02000 0.02000
"""


def test_parse_ilist_deduplicates_same_shell():
    path = write_tmp(ICOBILIST_ONE_SHELL)
    rows = _parse_ilist(path)
    assert len(rows) == 1
    assert rows[0]['sp1'] == 'F'   # canonical order
    assert rows[0]['sp2'] == 'Sc'
    assert rows[0]['distance'] == pytest.approx(2.01173, abs=1e-4)
    assert rows[0]['value'] == pytest.approx(0.40650, abs=1e-4)


def test_parse_ilist_two_shells():
    path = write_tmp(ICOBILIST_TWO_SHELLS)
    rows = _parse_ilist(path)
    assert len(rows) == 2
    dists = sorted(r['distance'] for r in rows)
    assert dists[0] == pytest.approx(2.01173, abs=1e-4)
    assert dists[1] == pytest.approx(3.5, abs=1e-4)


def test_lookup_basic():
    pairs = [{'sp1': 'F', 'sp2': 'Sc', 'distance': 2.012, 'icobi': 0.407}]
    assert lookup(pairs, 'Sc', 'F', 2.012) == pytest.approx(0.407)


def test_lookup_order_independent():
    pairs = [{'sp1': 'F', 'sp2': 'Sc', 'distance': 2.012, 'icobi': 0.407}]
    assert lookup(pairs, 'F', 'Sc', 2.012) == pytest.approx(0.407)


def test_lookup_out_of_tolerance():
    pairs = [{'sp1': 'F', 'sp2': 'Sc', 'distance': 2.012, 'icobi': 0.407}]
    assert lookup(pairs, 'Sc', 'F', 3.0) is None


def test_lookup_missing_key():
    pairs = [{'sp1': 'F', 'sp2': 'Sc', 'distance': 2.012}]
    assert lookup(pairs, 'Sc', 'F', 2.012, key='icobi') is None


# ---------------------------------------------------------------------------
# load_car_curves — single and multiple groups
# ---------------------------------------------------------------------------

def test_load_car_curves_no_match():
    content = make_cohpcar([('Sc', 'F', 2.012, 0.300)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    assert load_car_curves(path, hdr, 'Sc', 'F', 5.0) == []


def test_load_car_curves_single_group():
    """Two pairs with identical EF integrals collapse into one group."""
    content = make_cohpcar([('Sc', 'F', 2.012, 0.300), ('Sc', 'F', 2.012, 0.300)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    groups = load_car_curves(path, hdr, 'Sc', 'F', 2.012)
    assert len(groups) == 1
    assert groups[0]['n'] == 2
    assert groups[0]['ival_ef'] == pytest.approx(0.300, abs=1e-4)
    assert len(groups[0]['energy']) == 3


def test_load_car_curves_two_groups():
    """Divergent EF values produce two groups sorted by |ival_ef| descending."""
    content = make_cohpcar([('Sc', 'F', 2.012, 0.400), ('Sc', 'F', 2.012, -0.015)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    groups = load_car_curves(path, hdr, 'Sc', 'F', 2.012)
    assert len(groups) == 2
    assert abs(groups[0]['ival_ef']) > abs(groups[1]['ival_ef'])
    assert groups[0]['ival_ef'] == pytest.approx(0.400, abs=1e-4)
    assert groups[1]['ival_ef'] == pytest.approx(-0.015, abs=1e-4)


def test_load_car_curves_order_independent():
    content = make_cohpcar([('Sc', 'F', 2.012, 0.300)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    g1 = load_car_curves(path, hdr, 'Sc', 'F', 2.012)
    g2 = load_car_curves(path, hdr, 'F', 'Sc', 2.012)
    assert len(g1) == len(g2) == 1
    assert g1[0]['ival_ef'] == pytest.approx(g2[0]['ival_ef'])


def test_load_car_curves_within_tolerance():
    """A pair 0.04 Å off still matches within the default tol=0.05."""
    content = make_cohpcar([('Sc', 'F', 2.012, 0.300)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    groups = load_car_curves(path, hdr, 'Sc', 'F', 2.050)
    assert len(groups) == 1


def test_load_car_curves_beyond_tolerance():
    content = make_cohpcar([('Sc', 'F', 2.012, 0.300)])
    path = write_tmp(content, '.lobster')
    hdr = parse_car_header(path)
    assert load_car_curves(path, hdr, 'Sc', 'F', 2.1) == []


# ---------------------------------------------------------------------------
# find_lobster_dir
# ---------------------------------------------------------------------------

def test_find_lobster_dir_exists(tmp_path):
    ph  = tmp_path / 'ScF3_ph'
    lob = tmp_path / 'ScF3_lobster'
    ph.mkdir()
    lob.mkdir()
    assert find_lobster_dir(ph) == lob


def test_find_lobster_dir_missing(tmp_path):
    ph = tmp_path / 'ScF3_ph'
    ph.mkdir()
    assert find_lobster_dir(ph) is None


def test_find_lobster_dir_no_ph_suffix(tmp_path):
    d = tmp_path / 'ScF3'
    d.mkdir()
    assert find_lobster_dir(d) is None


# ---------------------------------------------------------------------------
# Orbitalwise N-centre COBICAR — reproduces the cobiBetween "... orbitalwise"
# quirk seen in real LOBSTER output: the same No.k index is reused for the
# plain total row and for every orbital combination row of that chain
# (n_orbitals ** n_atoms rows), so the data column can only be recovered
# from sequential header-line position, not from the (repeated) printed
# index.
# ---------------------------------------------------------------------------

POSCAR_LOB_3ATOM = """\
A3 test
   1.0
     5.0000000   0.0000000   0.0000000
     0.0000000   5.0000000   0.0000000
     0.0000000   0.0000000   5.0000000
A
3
Direct
  0.0000000   0.0000000   0.0000000
  0.2000000   0.0000000   0.0000000
  0.4000000   0.0000000   0.0000000
"""

# One leading ordinary pairwise entry (No.1) so the chain's column offset
# cannot coincide with (index - 1) by accident, then a 3-centre chain
# (No.2) with a plain total row followed by all 2**3 = 8 orbital rows for a
# toy 2-orbital ('5s', '5p') basis.
_NC_ORB_HEADER_LINES = [
    'No.1:A1[0 0 0]->A2[0 0 0]',
    'No.2:A1[0 0 0]->A2[0 0 0]->A3[0 0 0]',
    'No.2:A1[0 0 0][5s]->A2[0 0 0][5s]->A3[0 0 0][5s]',
    'No.2:A1[0 0 0][5s]->A2[0 0 0][5s]->A3[0 0 0][5p]',
    'No.2:A1[0 0 0][5s]->A2[0 0 0][5p]->A3[0 0 0][5s]',
    'No.2:A1[0 0 0][5s]->A2[0 0 0][5p]->A3[0 0 0][5p]',
    'No.2:A1[0 0 0][5p]->A2[0 0 0][5s]->A3[0 0 0][5s]',
    'No.2:A1[0 0 0][5p]->A2[0 0 0][5s]->A3[0 0 0][5p]',
    'No.2:A1[0 0 0][5p]->A2[0 0 0][5p]->A3[0 0 0][5s]',
    'No.2:A1[0 0 0][5p]->A2[0 0 0][5p]->A3[0 0 0][5p]',
]

# ival_ef for each header line above, in the same order. The chain total is
# deliberately near-zero while sss/ppp are large and opposite in sign — this
# can genuinely happen when opposite-sign orbital contributions cancel, even
# though individual orbital contributions are large. A near-zero total must
# not be treated as "no data" / an error.
_NC_ORB_EF_VALUES = [0.123, 0.001, 0.30, 0.02, -0.01, 0.05, -0.02, 0.01, -0.005, -0.29]


def make_nc_orbitalwise_cobicar() -> str:
    n = len(_NC_ORB_HEADER_LINES)
    lines = ['COBICAR.lobster', f'{n} 1 3 -1.0 1.0 0.0'] + _NC_ORB_HEADER_LINES
    for e_idx, energy in enumerate([-1.0, 0.0, 1.0]):
        row = [energy]
        for ef in _NC_ORB_EF_VALUES:
            ival = ef * e_idx if e_idx <= 1 else ef
            row += [0.5, ival]
        lines.append(' '.join(f'{v:.6f}' for v in row))
    return '\n'.join(lines) + '\n'


def test_parse_car_header_captures_orbital_rows():
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    assert hdr['n_pairs'] == 10
    assert len(hdr['pairs']) == 1            # No.1, plain 2-atom pair
    assert hdr['pairs'][0]['col_position'] == 0
    assert len(hdr['nc_pairs']) == 1          # No.2, the 3-centre chain
    chain2 = next(e for e in hdr['nc_pairs'] if e['index'] == 2)
    assert chain2['col_position'] == 1
    assert len(chain2['orbital_rows']) == 8
    assert chain2['orbital_rows'][0]['orbitals'] == ['5s', '5s', '5s']
    assert chain2['orbital_rows'][0]['col_position'] == 2
    assert chain2['orbital_rows'][-1]['orbitals'] == ['5p', '5p', '5p']
    assert chain2['orbital_rows'][-1]['col_position'] == 9


def test_load_nccobicar_curves_total_uses_correct_column():
    """
    Regression test: previously the column offset was derived from the
    (repeated) printed 'No.k' index, which coincides with the true column
    only for the first occurrence. This asserts the TOTAL row for a chain
    that has orbital rows after it is read from its true position, not from
    a formula based on the literal index number.
    """
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    curves = load_nccobicar_curves(
        path, hdr, 'cobiBetween atom1 atom2 atom3', lob_poscar)
    assert len(curves) == 1
    assert curves[0]['ival_ef'] == pytest.approx(0.001, abs=1e-6)


def test_load_nccobicar_orbital_curves_full_breakdown():
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    result = load_nccobicar_orbital_curves(
        path, hdr, 'cobiBetween atom1 atom2 atom3', lob_poscar)

    assert result['total']['ival_ef'] == pytest.approx(0.001, abs=1e-6)
    assert len(result['orbitals']) == 8

    by_orb = {tuple(o['orbitals']): o for o in result['orbitals']}
    assert by_orb[('5s', '5s', '5s')]['ival_ef'] == pytest.approx(0.30, abs=1e-6)
    assert by_orb[('5p', '5p', '5p')]['ival_ef'] == pytest.approx(-0.29, abs=1e-6)
    assert by_orb[('5s', '5p', '5s')]['ival_ef'] == pytest.approx(-0.01, abs=1e-6)

    # sorted by |ival_ef| descending: sss (0.30) then ppp (0.29) first
    assert result['orbitals'][0]['orbitals'] == ['5s', '5s', '5s']
    assert result['orbitals'][1]['orbitals'] == ['5p', '5p', '5p']

    # the near-zero chain total does not suppress or hide the large,
    # opposite-sign individual orbital contributions
    assert abs(result['total']['ival_ef']) < 0.01
    assert max(abs(o['ival_ef']) for o in result['orbitals']) > 0.25


def test_load_nccobicar_orbital_curves_no_match_returns_none():
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    result = load_nccobicar_orbital_curves(
        path, hdr, 'cobiBetween atom1 atom2', lob_poscar)  # wrong chain length
    assert result is None


def test_load_nc_entry_orbital_curves_matches_directive_path():
    """load_nc_entry_orbital_curves(entry) must agree with the directive-string
    entrypoint when given the matching entry directly (no re-matching)."""
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    entry = next(e for e in hdr['nc_pairs'] if e['index'] == 2)
    direct = load_nc_entry_orbital_curves(path, hdr, entry)

    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    via_directive = load_nccobicar_orbital_curves(
        path, hdr, 'cobiBetween atom1 atom2 atom3', lob_poscar)

    assert direct['total']['ival_ef'] == pytest.approx(via_directive['total']['ival_ef'])
    assert len(direct['orbitals']) == len(via_directive['orbitals'])


def test_entry_to_directive_roundtrips():
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    entry = next(e for e in hdr['nc_pairs'] if e['index'] == 2)

    directive = entry_to_directive(entry)
    assert directive == 'cobiBetween atom 1 atom 2 atom 3'

    # feeding the generated directive back in must resolve to the same entry
    result = load_nccobicar_orbital_curves(path, hdr, directive, lob_poscar)
    assert result['total']['ival_ef'] == pytest.approx(0.001, abs=1e-6)


# ---------------------------------------------------------------------------
# orbital_type / group_orbital_curves_by_type / filter_orbital_curves
# ---------------------------------------------------------------------------

def test_orbital_type_basic():
    assert orbital_type('5s') == 's'
    assert orbital_type('5p_z') == 'p'
    assert orbital_type('4d_xy') == 'd'
    assert orbital_type('4d_x^2-y^2') == 'd'


def test_group_orbital_curves_by_type_sums_and_sorts():
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    entry = next(e for e in hdr['nc_pairs'] if e['index'] == 2)
    result = load_nc_entry_orbital_curves(path, hdr, entry)

    # toy fixture only uses '5s'/'5p' tags, so grouping by type is a no-op
    # relabelling (each of the 8 rows is already its own s/p type-combo)
    groups = group_orbital_curves_by_type(result['orbitals'])
    assert len(groups) == 8
    assert all(g['n_rows'] == 1 for g in groups)
    by_type = {g['orbital_types']: g for g in groups}
    assert by_type[('s', 's', 's')]['ival_ef'] == pytest.approx(0.30, abs=1e-6)
    # descending |ival_ef| order preserved
    assert abs(groups[0]['ival_ef']) >= abs(groups[1]['ival_ef'])


def test_group_orbital_curves_by_type_collapses_m_orbitals():
    """
    With 3 m-resolved orbitals per atom but only 2 orbital TYPES (s and p),
    3**3 = 27 individual rows must collapse into 2**3 = 8 type groups, with
    every row accounted for exactly once.
    """
    import itertools
    orbitals_per_atom = ['5s', '5p_x', '5p_y']
    energy = np.array([-1.0, 0.0, 1.0])
    rows = [
        {'orbitals': list(combo), 'energy': energy,
         'curve': np.array([0.1, 0.1, 0.1]), 'icurve': np.array([0.0, 0.01, 0.01]),
         'ival_ef': 0.01}
        for combo in itertools.product(orbitals_per_atom, repeat=3)
    ]
    assert len(rows) == 27

    groups = group_orbital_curves_by_type(rows)
    assert len(groups) == 8
    assert sum(g['n_rows'] for g in groups) == 27


def test_filter_orbital_curves_drops_below_threshold():
    curves = [{'ival_ef': 0.30}, {'ival_ef': -0.005}, {'ival_ef': 0.02}]
    kept = filter_orbital_curves(curves, threshold=0.01)
    assert len(kept) == 2
    assert all(abs(c['ival_ef']) >= 0.01 for c in kept)


# ---------------------------------------------------------------------------
# parse_ncicobi_list / lookup_ncicobi / lookup_ncicobi_record — orbitalwise
#
# NcICOBILIST.lobster shares COBICAR's repeated-index-per-orbital-combination
# quirk (see make_nc_orbitalwise_cobicar above), but with a twist: orbital
# rows omit cell vectors entirely, using a bare orbital tag per atom instead
# of a [cell] bracket, e.g. 'A1[5s]->A3[5s]->A1[5s]' rather than
# 'A1[0 0 0][5s]->A3[0 -1 1][5s]->A1[0 -1 1][5s]'. This means a row can only
# be classified as "total" (numeric cell bracket) vs "orbital" (non-numeric
# tag bracket) by inspecting the bracket contents, not by counting brackets.
# ---------------------------------------------------------------------------

NCICOBILIST_ORBITALWISE = """\
  COBI#   No. of atoms  Nc-ICOBI (at) eF  Atoms for Nc-ICOBI
                              for spin 1
      1              2           0.40000  A1[0 0 0]->A2[0 0 0]
      2              3          -0.01317  A1[0 0 0]->A3[0 -1 1]->A1[0 -1 1]
      2              3          -0.00004  A1[5s]->A3[5s]->A1[5s]
      2              3           0.00003  A1[5s]->A3[5s]->A1[5p_z]
      2              3          -0.01316  A1[5p_z]->A3[5s]->A1[5s]
"""


def test_parse_ncicobi_list_plain_and_orbitalwise_records():
    path = write_tmp(NCICOBILIST_ORBITALWISE)
    records = parse_ncicobi_list(path)
    assert len(records) == 2

    plain = next(r for r in records if r['n_atoms'] == 2)
    assert plain['icobi'] == pytest.approx(0.40000)
    assert plain['orbital_rows'] == []

    chain = next(r for r in records if r['n_atoms'] == 3)
    assert chain['icobi'] == pytest.approx(-0.01317)
    assert chain['atoms'] == [('A1', [0, 0, 0]), ('A3', [0, -1, 1]), ('A1', [0, -1, 1])]
    assert len(chain['orbital_rows']) == 3
    assert chain['orbital_rows'][0]['orbitals'] == ['5s', '5s', '5s']
    assert chain['orbital_rows'][0]['ival_ef'] == pytest.approx(-0.00004)


def test_lookup_ncicobi_still_returns_float():
    path = write_tmp(NCICOBILIST_ORBITALWISE)
    records = parse_ncicobi_list(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    val = lookup_ncicobi(records, 'cobiBetween atom1 atom3 cell 0 -1 1 atom1 cell 0 -1 1',
                         lob_poscar)
    assert val == pytest.approx(-0.01317)


def test_lookup_ncicobi_record_returns_orbital_rows():
    path = write_tmp(NCICOBILIST_ORBITALWISE)
    records = parse_ncicobi_list(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    rec = lookup_ncicobi_record(
        records, 'cobiBetween atom1 atom3 cell 0 -1 1 atom1 cell 0 -1 1', lob_poscar)
    assert rec is not None
    assert rec['icobi'] == pytest.approx(-0.01317)
    assert len(rec['orbital_rows']) == 3
    total_of_orbitals = sum(o['ival_ef'] for o in rec['orbital_rows'])
    # toy fixture isn't self-consistent by design (only 3 of the real 27
    # rows are included) — just check the shape/keys are usable downstream
    assert isinstance(total_of_orbitals, float)


def test_group_orbital_curves_by_type_without_curve_data():
    """
    NcICOBILIST-sourced orbital rows carry no energy/curve/icurve — only
    'orbitals' and 'ival_ef'. group_orbital_curves_by_type must still work
    (summing just ival_ef) rather than KeyError on the missing fields.
    """
    rows = [
        {'orbitals': ['5s', '5s', '5s'], 'ival_ef': -0.00004},
        {'orbitals': ['5s', '5s', '5p_z'], 'ival_ef': 0.00003},
        {'orbitals': ['5p_z', '5s', '5s'], 'ival_ef': -0.01316},
    ]
    groups = group_orbital_curves_by_type(rows)
    # positional grouping key: ('s','s','s'), ('s','s','p'), ('p','s','s') are
    # three distinct groups since orbital position along the chain matters
    assert len(groups) == 3
    by_type = {g['orbital_types']: g for g in groups}
    assert by_type[('s', 's', 's')]['ival_ef'] == pytest.approx(-0.00004)
    assert by_type[('s', 's', 's')]['n_rows'] == 1
    assert 'curve' not in by_type[('s', 's', 's')]
    assert by_type[('s', 's', 'p')]['n_rows'] == 1
    assert by_type[('p', 's', 's')]['ival_ef'] == pytest.approx(-0.01316)


def test_load_nccobicar_orbital_curves_official_spaced_syntax():
    """
    LOBSTER's official cobiBetween syntax writes 'atom' and its index as
    separate tokens ('atom 1'), not concatenated ('atom1'). Directives
    copied verbatim out of a real lobsterin file use this spaced form, so
    it must resolve identically to the concatenated form.
    """
    path = write_tmp(make_nc_orbitalwise_cobicar(), '.lobster')
    hdr = parse_car_header(path)
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_3ATOM))
    spaced = load_nccobicar_orbital_curves(
        path, hdr, 'cobiBetween atom 1 atom 2 atom 3', lob_poscar)
    concatenated = load_nccobicar_orbital_curves(
        path, hdr, 'cobiBetween atom1 atom2 atom3', lob_poscar)
    assert spaced is not None
    assert spaced['total']['ival_ef'] == pytest.approx(concatenated['total']['ival_ef'])
    assert len(spaced['orbitals']) == len(concatenated['orbitals']) == 8


# ---------------------------------------------------------------------------
# is_translation_free
# ---------------------------------------------------------------------------

def test_is_translation_free_true_when_all_atoms_same_cell():
    entry = {'atoms': [('A1', [0, 0, 0]), ('A2', [0, 0, 0]), ('A3', [0, 0, 0])]}
    assert is_translation_free(entry) is True


def test_is_translation_free_false_when_any_atom_differs():
    entry = {'atoms': [('A1', [0, 0, 0]), ('A3', [0, -1, 1]), ('A1', [0, -1, 1])]}
    assert is_translation_free(entry) is False


def test_is_translation_free_compares_to_first_atom_not_literal_zero():
    """A chain uniformly offset by the same non-zero cell is still
    translation-free — what matters is agreement with the first atom's
    cell, not literal [0,0,0]."""
    entry = {'atoms': [('A1', [1, 1, 1]), ('A2', [1, 1, 1])]}
    assert is_translation_free(entry) is True


# ---------------------------------------------------------------------------
# check_ncicobi_consistency
#
# Two atoms, species A, in a cubic a=5.0 Å cell: A1 at frac (0,0,0), A2 at
# frac (0.5,0,0). A2 reached via cell [0,0,0] sits at x=2.5; reached via
# cell [-1,0,0] it sits at x=-2.5 — same 2.5 Å distance, same physical bond,
# different periodic image. Reached via cell [1,0,0] it sits at x=7.5 — a
# genuinely different (longer) bond, not part of the same shell.
# ---------------------------------------------------------------------------

POSCAR_LOB_2ATOM = """\
A2 test
   1.0
     5.0000000   0.0000000   0.0000000
     0.0000000   5.0000000   0.0000000
     0.0000000   0.0000000   5.0000000
A
2
Direct
  0.0000000   0.0000000   0.0000000
  0.5000000   0.0000000   0.0000000
"""

NCICOBI_INCONSISTENT = """\
  COBI#   No. of atoms  Nc-ICOBI (at) eF  Atoms for Nc-ICOBI
                              for spin 1
      1              2           0.50000  A1[0 0 0]->A2[0 0 0]
      2              2           0.10000  A1[0 0 0]->A2[-1 0 0]
      3              2           0.30000  A1[0 0 0]->A2[1 0 0]
"""


def test_check_ncicobi_consistency_flags_divergent_translations():
    records = parse_ncicobi_list(write_tmp(NCICOBI_INCONSISTENT))
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_2ATOM))
    result = check_ncicobi_consistency(records, lob_poscar)

    # only one shell has >= 2 instances (the 2.5 Å bond via cell [0,0,0] and
    # [-1,0,0]); the 7.5 Å bond via cell [1,0,0] is a lone instance, excluded
    assert len(result) == 1
    shell = result[0]
    assert shell['distance'] == pytest.approx(2.5)
    assert sorted(shell['values']) == [0.1, 0.5]
    assert shell['spread'] == pytest.approx(0.4)
    assert shell['relative_spread'] == pytest.approx(0.4 / 0.3)


NCICOBI_CONSISTENT = """\
  COBI#   No. of atoms  Nc-ICOBI (at) eF  Atoms for Nc-ICOBI
                              for spin 1
      1              2           0.50000  A1[0 0 0]->A2[0 0 0]
      2              2           0.50010  A1[0 0 0]->A2[-1 0 0]
"""


def test_check_ncicobi_consistency_small_spread_for_agreeing_translations():
    records = parse_ncicobi_list(write_tmp(NCICOBI_CONSISTENT))
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_2ATOM))
    result = check_ncicobi_consistency(records, lob_poscar)
    assert len(result) == 1
    assert result[0]['relative_spread'] < 0.001


def test_check_ncicobi_consistency_empty_when_no_shell_has_duplicates():
    NCICOBI_SINGLETONS = """\
      1              2           0.50000  A1[0 0 0]->A2[0 0 0]
      2              2           0.30000  A1[0 0 0]->A2[1 0 0]
"""
    records = parse_ncicobi_list(write_tmp(NCICOBI_SINGLETONS))
    lob_poscar = parse_poscar_lobster(write_tmp(POSCAR_LOB_2ATOM))
    assert check_ncicobi_consistency(records, lob_poscar) == []
