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
