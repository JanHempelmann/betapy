"""
Parser and lookup utilities for LOBSTER output files.

Supported files
---------------
ICOBILIST.lobster   – integrated COBI per bond
ICOHPLIST.lobster   – integrated COHP per bond
ICOOPLIST.lobster   – integrated COOP per bond
CHARGE.lobster      – Mulliken / Löwdin charges per atom
COBICAR.lobster     – energy-resolved COBI curves
COHPCAR.lobster     – energy-resolved COHP curves
COOPCAR.lobster     – energy-resolved COOP curves

Typical workflow
----------------
    from betapy.core import lobster

    # Integrated values (for CSV annotation / status bar)
    pairs = lobster.load_pairs(lobster_dir)
    val   = lobster.lookup(pairs, 'Sc', 'F', 2.012, key='icobi')

    # Energy-resolved curves (for the COHP plotter)
    hdr = lobster.parse_car_header(lobster_dir / 'COHPCAR.lobster')
    groups = lobster.load_car_curves(lobster_dir / 'COHPCAR.lobster',
                                     hdr, 'Sc', 'F', 2.012)
    if groups:
        g = groups[0]   # highest |ival_ef|
        energy, cohp, icohp = g['energy'], g['curve'], g['icurve']

Directory discovery
-------------------
    lobster_dir = lobster.find_lobster_dir(ph_dir)
    # e.g. ScF3/ScF3_ph  →  ScF3/ScF3_lobster

The ICOBILIST / ICOHPLIST / ICOOPLIST files list every symmetry-equivalent
interaction separately (one row per image/translation).  Values within a
shell are identical by symmetry; load_pairs deduplicates by
(species1, species2, distance_rounded) and keeps the representative value.
Species pairs are stored in canonical (alphabetical) order so lookup is
order-independent.

CAR file formats
----------------
COHPCAR / COOPCAR:
    Line 1 : description
    Line 2 : n_total  n_spins  n_E  E_min  E_max  E_Fermi
             (n_total includes the "Average" entry)
    Line 3 : "Average"
    Lines 4…: No.k:sp1->sp2(distance_Å)
    Data   : energy  avg_val  avg_ival  pair1_val  pair1_ival  …
             (2 columns per entry per spin; stride = 2*n_spins)

COBICAR:
    Line 1 : description
    Line 2 : n_total  n_spins  n_E  E_min  E_max  E_Fermi
             (n_total does NOT include Average)
    Lines 3…: No.k:sp1[u v w]->sp2[u' v' w']
             (translation vectors, no distance — computed from POSCAR.lobster)
    Data   : same column layout as COHPCAR but without the Average columns
"""

import re
from pathlib import Path

import numpy as np


_ILIST_FILES = {
    'icobi': 'ICOBILIST.lobster',
    'icohp': 'ICOHPLIST.lobster',
    'icoop': 'ICOOPLIST.lobster',
}

_DIST_ROUND = 4   # decimal places used when grouping equivalent interactions
_VAL_TOL    = 1e-3  # max spread within a distance group to treat values as equivalent
                    # (LOBSTER writes 5 sig figs, so 1e-3 >> numerical noise ~1e-5
                    #  but catches genuinely distinct environments at the same distance)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_label(label: str):
    """
    'Sc1' → ('Sc', 0)

    LOBSTER labels atoms as <species><1-based-absolute-index>.
    Returns (species_string, 0-based_index).
    """
    m = re.match(r'([A-Za-z]+)(\d+)$', label.strip())
    if not m:
        raise ValueError(f"Cannot parse LOBSTER atom label: {label!r}")
    return m.group(1), int(m.group(2)) - 1


def label_to_atom_index(label: str) -> int:
    """
    Public wrapper around _parse_label(): LOBSTER atom label (e.g. 'Sc1') ->
    1-based POSCAR atom index. For callers (e.g. 3D highlighting) that only
    need the index, not the species string.
    """
    return _parse_label(label)[1] + 1


def _canonical(sp1: str, sp2: str):
    """Return (sp1, sp2) in alphabetical order so pairs are order-independent."""
    return (sp1, sp2) if sp1 <= sp2 else (sp2, sp1)


def _parse_ilist(path) -> list:
    """
    Parse one ICO*LIST.lobster file.

    Returns a list of dicts {sp1, sp2, distance, value} with one entry per
    unique (species-pair, distance) shell.  Rows for symmetry-equivalent
    interactions are averaged (values are identical within a shell).
    """
    buckets: dict = {}
    with open(Path(path)) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                int(parts[0])           # data lines start with an integer index
            except ValueError:
                continue
            if len(parts) < 7:
                continue
            try:
                sp1, _ = _parse_label(parts[1])
                sp2, _ = _parse_label(parts[2])
                dist = float(parts[3])
                val = float(parts[-1])
            except (ValueError, IndexError):
                continue
            key = (*_canonical(sp1, sp2), round(dist, _DIST_ROUND))
            buckets.setdefault(key, []).append(val)

    result = []
    for k, vals in sorted(buckets.items()):
        # Always average for now.  A spread check (max-min > _VAL_TOL) would
        # detect genuinely inequivalent environments at the same distance
        # (e.g. the two Sc-F shell types in rocksalt-related structures), but
        # the LOBSTER release build has a translation-vector bug that produces
        # artificially wrong values for some cell-translated interactions,
        # creating false positives.  Threshold-based detection cannot
        # distinguish the two cases without bond-direction information.
        # TODO: restore ambiguity detection once direction-cosine matching is
        # implemented (pairs matched by |cos θ| ≥ 0.99 rather than distance
        # alone will land in separate buckets, making spread detection reliable).
        value = sum(vals) / len(vals)
        result.append({'sp1': k[0], 'sp2': k[1], 'distance': k[2], 'value': value})
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_charges(path) -> list:
    """
    Parse CHARGE.lobster.

    Returns a list of dicts:
        {atom_idx (0-based int), species (str), mulliken (float), loewdin (float)}
    """
    records = []
    with open(Path(path)) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                idx = int(parts[0]) - 1
                sp = re.sub(r'\d+', '', parts[1])
                mulliken = float(parts[2])
                loewdin = float(parts[3])
                records.append({'atom_idx': idx, 'species': sp,
                                'mulliken': mulliken, 'loewdin': loewdin})
            except ValueError:
                continue
    return records


def load_pairs(lobster_dir, available=None) -> list:
    """
    Read all available ICO*LIST files from *lobster_dir* and merge into a
    single list of pair records.

    Parameters
    ----------
    lobster_dir : path-like
        Directory containing LOBSTER output files.
    available : iterable of {'icobi', 'icohp', 'icoop'} or None
        Which quantities to load.  Defaults to all files that exist.

    Returns
    -------
    List of dicts with keys: sp1, sp2, distance, and whichever of
    icobi/icohp/icoop were found.  One entry per unique (species-pair,
    distance) shell.
    """
    d = Path(lobster_dir)
    keys_to_load = list(available) if available is not None else list(_ILIST_FILES)

    merged: dict = {}
    for key in keys_to_load:
        fpath = d / _ILIST_FILES[key]
        if not fpath.exists():
            continue
        for row in _parse_ilist(fpath):
            k = (row['sp1'], row['sp2'], row['distance'])
            merged.setdefault(k, {'sp1': row['sp1'], 'sp2': row['sp2'],
                                  'distance': row['distance']})
            merged[k][key] = row['value']

    return list(merged.values())


def lookup(pairs: list, sp1: str, sp2: str, distance: float,
           key: str = 'icobi', tol: float = 0.05):
    """
    Return the integrated value for the bond (sp1, sp2) nearest to *distance*.

    Parameters
    ----------
    pairs    : output of load_pairs()
    sp1, sp2 : species strings (order-independent)
    distance : bond length in Å
    key      : 'icobi', 'icohp', or 'icoop'
    tol      : maximum allowed deviation in Å; returns None if no match

    Returns
    -------
    float or None
        None if no match within *tol*, or if the nearest match is ambiguous
        (two structurally inequivalent environments with the same distance).
    """
    cs1, cs2 = _canonical(sp1, sp2)
    best_val = None
    best_dev = tol + 1.0
    for row in pairs:
        if row['sp1'] != cs1 or row['sp2'] != cs2:
            continue
        if key not in row:
            continue
        dev = abs(row['distance'] - distance)
        if dev < best_dev:
            best_dev = dev
            best_val = row[key]   # may be None if flagged as ambiguous
    return best_val if best_dev <= tol else None


# ---------------------------------------------------------------------------
# Energy-resolved CAR file parsing
# ---------------------------------------------------------------------------

_CAR_FILES = {
    'cohp': 'COHPCAR.lobster',
    'coop': 'COOPCAR.lobster',
    'cobi': 'COBICAR.lobster',
}


def _parse_poscar_lobster(path) -> dict:
    """
    Parse POSCAR.lobster (standard POSCAR format).

    Returns {lattice (3×3 ndarray, rows = lattice vectors),
             positions_frac (N×3 ndarray),
             species (list of str, one per atom)}
    """
    with open(Path(path)) as f:
        lines = [l.rstrip('\n') for l in f]
    scale = float(lines[1].split()[0])
    lat = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale
    sp_names = lines[5].split()
    sp_counts = list(map(int, lines[6].split()))
    species = []
    for sp, n in zip(sp_names, sp_counts):
        species.extend([sp] * n)
    # line 7 is "Direct" or "Cartesian"
    n_atoms = sum(sp_counts)
    fracs = np.array([list(map(float, lines[8 + i].split()[:3]))
                      for i in range(n_atoms)])
    return {'lattice': lat, 'positions_frac': fracs, 'species': species}


def parse_car_header(path) -> dict:
    """
    Parse the header of any *CAR.lobster file (COHPCAR, COOPCAR, COBICAR).

    Returns a dict:
        n_spins       : int
        n_e           : int
        e_fermi       : float
        has_average   : bool  (True for COHP/COOP, False for COBI)
        n_pairs       : int
        first_data_line: int (0-based index into file lines)
        pairs         : list of dicts, each with:
            index        : int (1-based, matches No.k label)
            col_position : int (0-based sequential header-line position;
                           this — not 'index' — is what determines the data
                           column, since orbitalwise N-centre entries reuse
                           the same 'index' for many consecutive lines)
            sp1, sp2   : str
            distance   : float or None (None for COBICAR until enriched)
            atm1, atm2 : str (full LOBSTER atom labels, e.g. 'Sc1')
            cell1, cell2: list[int] or None (COBICAR translation vectors)
        nc_pairs      : list of dicts, each with:
            index        : int (No.k label, shared by the total row and all
                           of its orbitalwise rows)
            atoms        : list of (label, cell) — the chain, atom-only
            col_position : int or None (None if only orbitalwise rows were
                           found for this chain, no total/summed row)
            orbital_rows : list of dicts {orbitals: list[str] (one tag per
                           chain atom, e.g. '5s', '5p_z'), col_position: int}
    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()

    meta = lines[1].split()
    n_total = int(meta[0])
    n_spins = int(meta[1])
    n_e     = int(meta[2])
    e_fermi = float(meta[5])

    has_average = lines[2].strip() == 'Average'
    hdr_start   = 3 if has_average else 2
    n_pairs     = n_total - (1 if has_average else 0)

    # Regex for COHPCAR/COOPCAR: No.k:Sc1->F2(2.01173)
    _re_dist = re.compile(
        r'No\.(\d+):([A-Za-z]+\d+)->([A-Za-z]+\d+)\(([\d.eE+\-]+)\)'
    )
    # Regex for COBICAR: No.k:Sc1[0 0 0]->F2[-1 0 0]
    _re_cell = re.compile(
        r'No\.(\d+):([A-Za-z]+\d+)\[([^\]]+)\]->([A-Za-z]+\d+)\[([^\]]+)\]'
    )

    # Regex for N-centre COBICAR: No.k:S10[0 0 0]->Bi43[0 0 1]->S7[0 0 1]
    _re_nc = re.compile(
        r'No\.(\d+):((?:[A-Za-z]+\d+\[[^\]]+\]->)+[A-Za-z]+\d+\[[^\]]+\])'
    )
    _re_nc_atom = re.compile(r'([A-Za-z]+\d+)\[([^\]]+)\]')

    # Regex for orbitalwise N-centre COBICAR: each atom carries a second
    # bracket with its orbital tag, e.g.
    #   No.17:A1[0 0 0][5s]->B1[0 0 0][5s]->C1[0 0 0][5p_z]
    # The same No.k index is reused for every orbital combination of a given
    # chain (n_orbitals ** n_atoms rows), immediately following that chain's
    # plain total row.
    _re_nc_orb = re.compile(
        r'No\.(\d+):((?:[A-Za-z]+\d+\[[^\]]+\]\[[^\]]+\]->)+'
        r'[A-Za-z]+\d+\[[^\]]+\]\[[^\]]+\])'
    )
    _re_nc_orb_atom = re.compile(r'([A-Za-z]+\d+)\[([^\]]+)\]\[([^\]]+)\]')

    pairs       = []
    nc_pairs    = []
    nc_by_index = {}   # index -> entry already appended to nc_pairs
    for i in range(hdr_start, hdr_start + n_pairs):
        pos  = i - hdr_start
        line = lines[i].strip()
        m = _re_dist.match(line)
        if m:
            sp1 = re.match(r'([A-Za-z]+)', m.group(2)).group(1)
            sp2 = re.match(r'([A-Za-z]+)', m.group(3)).group(1)
            pairs.append({
                'index': int(m.group(1)), 'col_position': pos,
                'sp1': sp1, 'sp2': sp2,
                'distance': float(m.group(4)),
                'atm1': m.group(2), 'atm2': m.group(3),
                'cell1': None, 'cell2': None,
            })
            continue
        m = _re_cell.match(line)
        if m:
            if '->' in line[m.end():]:   # N-centre COBICAR entry
                m2 = _re_nc.match(line)
                if m2:
                    idx   = int(m2.group(1))
                    atoms = [(a.group(1), list(map(int, a.group(2).split())))
                             for a in _re_nc_atom.finditer(m2.group(2))]
                    entry = {'index': idx, 'atoms': atoms,
                             'col_position': pos, 'orbital_rows': []}
                    nc_pairs.append(entry)
                    nc_by_index[idx] = entry
                continue
            sp1 = re.match(r'([A-Za-z]+)', m.group(2)).group(1)
            sp2 = re.match(r'([A-Za-z]+)', m.group(4)).group(1)
            pairs.append({
                'index': int(m.group(1)), 'col_position': pos,
                'sp1': sp1, 'sp2': sp2,
                'distance': None,
                'atm1': m.group(2), 'atm2': m.group(4),
                'cell1': list(map(int, m.group(3).split())),
                'cell2': list(map(int, m.group(5).split())),
            })
            continue
        m = _re_nc_orb.match(line)
        if m:
            idx    = int(m.group(1))
            triples = [(a.group(1), list(map(int, a.group(2).split())), a.group(3))
                       for a in _re_nc_orb_atom.finditer(m.group(2))]
            atoms    = [(lbl, cell) for lbl, cell, orb in triples]
            orbitals = [orb for lbl, cell, orb in triples]
            entry = nc_by_index.get(idx)
            if entry is None:
                entry = {'index': idx, 'atoms': atoms,
                          'col_position': None, 'orbital_rows': []}
                nc_pairs.append(entry)
                nc_by_index[idx] = entry
            entry['orbital_rows'].append({'orbitals': orbitals, 'col_position': pos})

    # Skip blank lines between header and data
    first_data = hdr_start + n_pairs
    while first_data < len(lines) and not lines[first_data].strip():
        first_data += 1

    return {
        'n_spins': n_spins, 'n_e': n_e, 'e_fermi': e_fermi,
        'has_average': has_average, 'n_total': n_total, 'n_pairs': n_pairs,
        'first_data_line': first_data, 'pairs': pairs, 'nc_pairs': nc_pairs,
    }


def enrich_cobicar_distances(header: dict, poscar_lobster_path) -> None:
    """
    Compute and fill in distances for COBICAR pairs (in-place).

    COBICAR headers carry translation vectors instead of distances.
    This function uses POSCAR.lobster atom positions to compute each distance.
    Call once after parse_car_header() for a COBICAR file.
    """
    pdata = _parse_poscar_lobster(poscar_lobster_path)
    lat   = pdata['lattice']
    fracs = pdata['positions_frac']

    for p in header['pairs']:
        if p['distance'] is not None or p['cell1'] is None:
            continue
        try:
            _, idx1 = _parse_label(p['atm1'])
            _, idx2 = _parse_label(p['atm2'])
            pos1 = (fracs[idx1] + np.asarray(p['cell1'])) @ lat
            pos2 = (fracs[idx2] + np.asarray(p['cell2'])) @ lat
            p['distance'] = float(np.linalg.norm(pos2 - pos1))
        except Exception:
            pass


def load_car_curves(path, header: dict,
                    sp1: str, sp2: str, distance: float,
                    tol: float = 0.05) -> list:
    """
    Load energy-resolved curves for the given species pair from a *CAR.lobster.

    Matching pairs are grouped by their integrated value at the Fermi level.
    Pairs whose EF integral differs by more than _VAL_TOL are placed in
    separate groups; otherwise they are averaged together.  This lets callers
    expose the individual groups when the distance shell contains inequivalent
    environments (or when a LOBSTER bug produces divergent values for
    symmetry-equivalent pairs).

    Parameters
    ----------
    path    : path to the *CAR.lobster file
    header  : output of parse_car_header() (optionally enriched with distances)
    sp1, sp2: species strings (order-independent)
    distance: target bond length in Å
    tol     : matching tolerance in Å

    Returns
    -------
    list of dicts sorted by |ival_ef| descending (strongest bonding first).
    Each dict has:
        energy   : 1-D ndarray (eV, Fermi = 0)
        curve    : 1-D ndarray (COHP / COOP / COBI per eV or dimensionless)
        icurve   : 1-D ndarray (integrated value up to each energy point)
        n        : int   (number of pairs averaged into this group)
        ival_ef  : float (group's mean integrated value at the Fermi level)
    Returns an empty list if no matching pairs are found.
    """
    cs1, cs2 = _canonical(sp1, sp2)
    matching = [p for p in header['pairs']
                if p['distance'] is not None
                and _canonical(p['sp1'], p['sp2']) == (cs1, cs2)
                and abs(p['distance'] - distance) <= tol]

    if not matching:
        return []

    n_spins = header['n_spins']
    has_avg = header['has_average']

    def _entry_cols(col_position):
        entry = col_position + 1 if has_avg else col_position
        base  = 1 + entry * 2 * n_spins
        return base, base + 1

    # Load all needed columns in one pass
    usecols = [0]
    slots   = []   # (curve_slot, icurve_slot) in usecols, one per `matching` entry
    for p in matching:
        c, ic = _entry_cols(p['col_position'])
        slots.append((len(usecols), len(usecols) + 1))
        usecols.extend([c, ic])

    try:
        data = np.loadtxt(path, skiprows=header['first_data_line'],
                          usecols=usecols)
    except Exception:
        return []

    energy = data[:, 0]
    ef_idx = int(np.argmin(np.abs(energy)))

    # Collect per-pair (ival_ef, curve, icurve) and sort by ival_ef
    pair_data = []
    for (s_c, s_ic) in slots:
        curve   = data[:, s_c]
        icurve  = data[:, s_ic]
        pair_data.append((float(icurve[ef_idx]), curve, icurve))
    pair_data.sort(key=lambda x: x[0])

    # Greedy consecutive clustering within _VAL_TOL
    clusters = []
    current  = [pair_data[0]]
    for item in pair_data[1:]:
        if abs(item[0] - current[-1][0]) <= _VAL_TOL:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)

    result = []
    for grp in clusters:
        result.append({
            'energy':  energy,
            'curve':   np.mean([x[1] for x in grp], axis=0),
            'icurve':  np.mean([x[2] for x in grp], axis=0),
            'n':       len(grp),
            'ival_ef': sum(x[0] for x in grp) / len(grp),
        })

    result.sort(key=lambda x: abs(x['ival_ef']), reverse=True)
    return result


# ---------------------------------------------------------------------------
# N-center COBI — NcICOBILIST and NcCOBICAR
# ---------------------------------------------------------------------------

_re_nc_icobi_row  = re.compile(r'^\s*(\d+)\s+(\d+)\s+([+-]?\d[\d.eE+-]*)\s+(.*)')
_re_nc_atom_cell  = re.compile(r'([A-Za-z]+\d+)\[([^\]]+)\]')


def parse_ncicobi_list(path) -> list:
    """
    Parse NcICOBILIST.lobster.

    When a chain was requested with `orbitalwise`, LOBSTER writes one extra
    row per orbital combination (n_orbitals ** n_atoms), sharing the same
    COBI# index as the plain total row and immediately following it — the
    same convention used by COBICAR.lobster's own orbitalwise entries (see
    parse_car_header). Unlike COBICAR, these rows omit cell vectors
    entirely (implied by the shared index) and give an orbital tag per atom
    instead, e.g. 'Sn1[5s]->Sn3[5s]->Sn1[5s]' rather than
    'Sn1[0 0 0]->Sn3[0 -1 1]->Sn1[0 -1 1]'.

    Returns list of dicts, one per requested chain:
        n_atoms      : int
        icobi        : float (Nc-ICOBI at EF, spin 1)
        atoms        : list of (label_str, [h,k,l]) e.g. [('Sc1',[0,0,0]),('F2',[0,0,0])]
        orbital_rows : list of {orbitals: list[str], ival_ef: float} — empty
                       unless the chain was requested orbitalwise
    """
    records  = []
    by_index = {}
    with open(Path(path)) as f:
        for line in f:
            m = _re_nc_icobi_row.match(line)
            if not m:
                continue
            try:
                idx     = int(m.group(1))
                n_atoms = int(m.group(2))
                icobi   = float(m.group(3))
                tokens  = [(am.group(1), am.group(2))
                           for am in _re_nc_atom_cell.finditer(m.group(4))]
            except (ValueError, AttributeError):
                continue
            if len(tokens) != n_atoms:
                continue

            # A total row's bracket holds a numeric cell vector; an orbital
            # row's bracket holds an orbital tag (non-numeric) instead.
            is_total = all(
                all(t.lstrip('-').isdigit() for t in bracket.split())
                for _, bracket in tokens
            )
            if is_total:
                atoms = [(lbl, list(map(int, bracket.split())))
                         for lbl, bracket in tokens]
                rec = {'n_atoms': n_atoms, 'icobi': icobi, 'atoms': atoms,
                       'orbital_rows': []}
                records.append(rec)
                by_index[idx] = rec
            else:
                rec = by_index.get(idx)
                if rec is not None:
                    rec['orbital_rows'].append(
                        {'orbitals': [bracket for _, bracket in tokens],
                         'ival_ef': icobi})
    return records


def parse_poscar_lobster(path) -> dict:
    """Public alias for the internal POSCAR.lobster parser."""
    return _parse_poscar_lobster(path)


def _directive_to_chain(directive_str, lob_poscar):
    """
    Convert a cobiBetween directive to a list of (lobster_label, cell) pairs.

    Accepts both the official LOBSTER syntax, 'atom' and its POSCAR index as
    separate tokens, and the concatenated form (not officially documented,
    but tolerated here in case it is still accepted by LOBSTER or appears in
    older files):

    'cobiBetween atom 1 atom 2 cell 1 0 0 atom 1'
    'cobiBetween atom1 atom2 cell 1 0 0 atom1'
      → [('Sc1',[0,0,0]), ('F2',[0,0,0]), ('Sc1',[1,0,0])]

    Returns None if parsing fails.
    """
    tokens = directive_str.split()
    if len(tokens) < 3 or tokens[0].lower() != 'cobibetween':
        return None
    tokens  = tokens[1:]
    species = lob_poscar['species']   # 0-based list
    chain   = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == 'atom' and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            idx0 = int(tokens[i + 1]) - 1      # 0-based POSCAR index
            i += 2
        elif tok.startswith('atom') and tok[4:].isdigit():
            idx0 = int(tok[4:]) - 1            # 0-based POSCAR index
            i += 1
        else:
            i += 1
            continue
        try:
            label = species[idx0] + str(idx0 + 1)
        except IndexError:
            return None
        cell = [0, 0, 0]
        if i < len(tokens) and tokens[i] == 'cell':
            try:
                cell = [int(tokens[i+1]), int(tokens[i+2]), int(tokens[i+3])]
                i += 4
            except (ValueError, IndexError):
                return None
        chain.append((label, cell))
    return chain if len(chain) >= 2 else None


def _chain_variants(chain):
    """
    Yield all normalized cyclic rotations and their reversals.

    Each variant has the first atom at cell [0,0,0].
    """
    n = len(chain)
    for start in range(n):
        rot  = chain[start:] + chain[:start]
        base = rot[0][1]
        yield [(lbl, [c - b for c, b in zip(cell, base)])
               for lbl, cell in rot]
    rev = list(reversed(chain))
    for start in range(n):
        rot  = rev[start:] + rev[:start]
        base = rot[0][1]
        yield [(lbl, [c - b for c, b in zip(cell, base)])
               for lbl, cell in rot]


def lookup_ncicobi_record(records, directive_str, lob_poscar):
    """
    Find the full NcICOBILIST record for a cobiBetween directive.

    Parameters
    ----------
    records      : output of parse_ncicobi_list()
    directive_str: full cobiBetween line string
    lob_poscar   : output of parse_poscar_lobster() / _parse_poscar_lobster()

    Returns
    -------
    dict (one item of `records`, including 'orbital_rows') or None if not found.
    """
    chain = _directive_to_chain(directive_str, lob_poscar)
    if chain is None:
        return None
    n = len(chain)
    for rec in records:
        if rec['n_atoms'] != n:
            continue
        for variant in _chain_variants(chain):
            if variant == rec['atoms']:
                return rec
    return None


def lookup_ncicobi(records, directive_str, lob_poscar):
    """
    Find the total NcICOBI value for a cobiBetween directive.

    Thin wrapper around lookup_ncicobi_record() for callers that only need
    the integrated total, not the orbital breakdown.

    Parameters
    ----------
    records      : output of parse_ncicobi_list()
    directive_str: full cobiBetween line string
    lob_poscar   : output of parse_poscar_lobster() / _parse_poscar_lobster()

    Returns
    -------
    float or None if not found.
    """
    rec = lookup_ncicobi_record(records, directive_str, lob_poscar)
    return rec['icobi'] if rec is not None else None


def is_translation_free(entry) -> bool:
    """
    True if every atom in *entry* (a header['nc_pairs'] entry from
    parse_car_header, or a parse_ncicobi_list() record — both share the
    same 'atoms' shape) sits in the same cell image as the chain's first
    atom, i.e. the chain involves no periodic-cell translation.

    In our own testing, COBICAR.lobster/COHPCAR.lobster/COOPCAR.lobster
    energy-resolved values agreed with the corresponding integrated value
    for translation-free interactions but not for translated ones (not
    traced to a betapy parsing issue — checked against the raw file text
    directly); see docs/lobster-cobi.md. This is the check used to warn
    before trusting COBICAR-derived curves for a selected chain.
    """
    base = entry['atoms'][0][1]
    return all(cell == base for _, cell in entry['atoms'])


def _atom_cart_position(label, cell, lob_poscar):
    """Cartesian position of a (label, cell) atom, e.g. ('Sn1', [0, -1, 1])."""
    _, idx0 = _parse_label(label)
    frac = lob_poscar['positions_frac'][idx0] + np.asarray(cell, dtype=float)
    return frac @ lob_poscar['lattice']


def check_ncicobi_consistency(records, lob_poscar, dist_round: int = 3) -> list:
    """
    Empirically check whether NcICOBILIST.lobster's own pairwise values are
    internally self-consistent across periodic-cell translations, for THIS
    LOBSTER installation and THIS run.

    Background: in our own testing (not traced to a betapy parsing issue —
    checked against the raw file text directly), COBICAR.lobster/
    COHPCAR.lobster/COOPCAR.lobster energy-resolved values for an
    interaction spanning a periodic-cell translation did not match the
    corresponding integrated value. We can't say how general this is across
    LOBSTER versions or settings, and NcICOBILIST isn't assumed exempt from
    the same kind of divergence just because it's a different file — that
    has to be checked per calculation, not assumed either way. This checks
    it directly: the same physical bond, reached via different
    translations, should give the same ICOBI value. LOBSTER writes an
    automatic Nc-ICOBI entry for every ordinary nearest-neighbour pair
    (regardless of any cobiBetween directive), and — because of periodicity
    — most bonds appear more than once at different translations, so this
    check is almost always possible without needing any special directive.

    A large spread within a shell suggests this directory's NcICOBILIST may
    show the same kind of translation-dependent divergence observed for
    COBICAR, and its multicenter (3+ atom) values should then be treated
    with the same caution, rather than trusted by default.

    Parameters
    ----------
    records    : output of parse_ncicobi_list() — only n_atoms == 2 entries
                 are used (multicenter entries have no shared distance to
                 group multiple instances by)
    lob_poscar : output of parse_poscar_lobster()
    dist_round : int, decimal places used to group entries into the same
                 distance shell

    Returns
    -------
    list of dicts, one per (species-pair, distance) shell with >= 2
    instances found:
        sp1, sp2       : str (canonical alphabetical order)
        distance       : float, Å
        values         : list[float], the ICOBI value at each translation found
        spread         : float, max(values) - min(values)
        relative_spread: float, spread / mean(|values|) (inf if that mean is 0)
    Sorted by relative_spread descending (worst first). Empty list if fewer
    than two translations of any bond were found (nothing to compare).
    """
    from collections import defaultdict

    buckets = defaultdict(list)
    for rec in records:
        if rec['n_atoms'] != 2:
            continue
        (l1, c1), (l2, c2) = rec['atoms']
        sp1, _ = _parse_label(l1)
        sp2, _ = _parse_label(l2)
        cs1, cs2 = _canonical(sp1, sp2)
        distance = float(np.linalg.norm(
            _atom_cart_position(l2, c2, lob_poscar)
            - _atom_cart_position(l1, c1, lob_poscar)))
        buckets[(cs1, cs2, round(distance, dist_round))].append(rec['icobi'])

    result = []
    for (sp1, sp2, dist), values in buckets.items():
        if len(values) < 2:
            continue
        spread    = max(values) - min(values)
        mean_abs  = sum(abs(v) for v in values) / len(values)
        rel_sprd  = spread / mean_abs if mean_abs > 0 else float('inf')
        result.append({'sp1': sp1, 'sp2': sp2, 'distance': dist,
                       'values': values, 'spread': spread,
                       'relative_spread': rel_sprd})
    result.sort(key=lambda x: x['relative_spread'], reverse=True)
    return result


def _match_nc_entry(header, directive_str, lob_poscar):
    """
    Find the header['nc_pairs'] entry matching a cobiBetween directive.

    Tries every cyclic rotation and reversal of the directive's chain (a
    cobiBetween directive is unordered up to those symmetries) against each
    entry's atom-only chain.

    Returns
    -------
    dict or None : the matching entry (with 'col_position' and
                   'orbital_rows' as parsed by parse_car_header), or None if
                   the chain length is invalid or no entry matches.
    """
    chain = _directive_to_chain(directive_str, lob_poscar)
    if chain is None:
        return None
    n = len(chain)
    for entry in header['nc_pairs']:
        if len(entry['atoms']) != n:
            continue
        for variant in _chain_variants(chain):
            if variant == entry['atoms']:
                return entry
    return None


def load_nccobicar_curves(path, header, directive_str, lob_poscar) -> list:
    """
    Load the energy-resolved TOTAL (summed-over-orbitals) NcCOBI curve for a
    cobiBetween directive, from a COBICAR.lobster file already parsed with
    parse_car_header().

    Parameters
    ----------
    path         : path to COBICAR.lobster
    header       : output of parse_car_header()
    directive_str: full cobiBetween line string
    lob_poscar   : output of parse_poscar_lobster()

    Returns
    -------
    list of 0 or 1 dicts {energy, curve, icurve, ival_ef}. Empty if the
    chain isn't found, or if it was requested orbitalwise only (no total
    row was written for it).
    """
    entry = _match_nc_entry(header, directive_str, lob_poscar)
    if entry is None or entry['col_position'] is None:
        return []

    n_spins = header['n_spins']
    base    = 1 + entry['col_position'] * 2 * n_spins

    try:
        data = np.loadtxt(path, skiprows=header['first_data_line'],
                          usecols=[0, base, base + 1])
    except Exception:
        return []

    if data.ndim == 1:
        data = data.reshape(1, -1)
    energy = data[:, 0]
    ef_idx = int(np.argmin(np.abs(energy)))
    return [{'energy': energy, 'curve': data[:, 1], 'icurve': data[:, 2],
             'ival_ef': float(data[ef_idx, 2])}]


def load_nc_entry_orbital_curves(path, header, entry) -> dict:
    """
    Load the total and all orbitalwise-resolved NcCOBI curves for one
    header['nc_pairs'] entry (as parsed by parse_car_header()), directly —
    no directive-string matching needed. This is the entry point for
    callers (e.g. a GUI listing every chain found in the file) that already
    have the entry in hand and don't want to round-trip through a
    cobiBetween directive string just to re-match it.

    Requesting a chain orbitalwise in lobsterin (``cobiBetween ... orbitalwise``)
    produces one row per orbital combination across the chain's atoms
    (n_orbitals ** n_atoms rows) in addition to the plain total row — see
    parse_car_header's 'orbital_rows'. This can be a large number of curves
    (729 for a 3-centre chain with 9 orbitals per atom); callers doing
    orbitalwise plotting/browsing should expect to filter/group them rather
    than display all of them at once — see group_orbital_curves_by_type()
    and filter_orbital_curves().

    Note: the total ICOBI can integrate to ~0 even when individual orbital
    contributions are large and non-zero — opposite-sign orbital
    contributions can cancel in the sum. A near-zero total is not evidence
    that the orbitalwise rows are wrong or absent.

    Parameters
    ----------
    path   : path to COBICAR.lobster
    header : output of parse_car_header()
    entry  : one item of header['nc_pairs']

    Returns
    -------
    dict:
        'total'    : {energy, curve, icurve, ival_ef} or None if no total
                     row was written for this chain.
        'orbitals' : list of dicts, one per orbital-combination row, each
                     {orbitals: list[str] (one tag per chain atom, in chain
                     order), energy, curve, icurve, ival_ef}, sorted by
                     |ival_ef| descending. Empty list if the chain was not
                     requested orbitalwise.
    """
    n_spins = header['n_spins']

    def _cols(col_position):
        base = 1 + col_position * 2 * n_spins
        return base, base + 1

    usecols = [0]
    slots   = []   # (kind, orbitals_or_None, curve_slot, icurve_slot)
    if entry['col_position'] is not None:
        c, ic = _cols(entry['col_position'])
        slots.append(('total', None, len(usecols), len(usecols) + 1))
        usecols.extend([c, ic])
    for row in entry['orbital_rows']:
        c, ic = _cols(row['col_position'])
        slots.append(('orbital', row['orbitals'], len(usecols), len(usecols) + 1))
        usecols.extend([c, ic])

    if len(usecols) == 1:
        return {'total': None, 'orbitals': []}

    data = np.loadtxt(path, skiprows=header['first_data_line'], usecols=usecols)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    energy = data[:, 0]
    ef_idx = int(np.argmin(np.abs(energy)))

    result = {'total': None, 'orbitals': []}
    for kind, orbitals, s_c, s_ic in slots:
        curve, icurve = data[:, s_c], data[:, s_ic]
        item = {'energy': energy, 'curve': curve, 'icurve': icurve,
                'ival_ef': float(icurve[ef_idx])}
        if kind == 'total':
            result['total'] = item
        else:
            item['orbitals'] = orbitals
            result['orbitals'].append(item)

    result['orbitals'].sort(key=lambda x: abs(x['ival_ef']), reverse=True)
    return result


def load_nccobicar_orbital_curves(path, header, directive_str, lob_poscar) -> dict:
    """
    Load the total and all orbitalwise-resolved NcCOBI curves for a
    cobiBetween directive, from a COBICAR.lobster file already parsed with
    parse_car_header(). Thin directive-string-matching wrapper around
    load_nc_entry_orbital_curves() — see that function for the return shape
    and orbitalwise-scale caveats.

    Parameters
    ----------
    path         : path to COBICAR.lobster
    header       : output of parse_car_header()
    directive_str: full cobiBetween line string
    lob_poscar   : output of parse_poscar_lobster()

    Returns
    -------
    None if the chain is not found in header['nc_pairs']; otherwise see
    load_nc_entry_orbital_curves().
    """
    entry = _match_nc_entry(header, directive_str, lob_poscar)
    if entry is None:
        return None
    return load_nc_entry_orbital_curves(path, header, entry)


def entry_to_directive(entry) -> str:
    """
    Build the canonical (official-syntax) cobiBetween directive string for a
    header['nc_pairs'] entry, e.g. for display or for re-parsing with
    _match_nc_entry()/load_nccobicar_orbital_curves(). Inverse of the atom
    part of _directive_to_chain(): cell tags are only emitted for atoms with
    a non-zero cell relative to the chain's first atom.

    Parameters
    ----------
    entry : one item of header['nc_pairs']

    Returns
    -------
    str, e.g. 'cobiBetween atom 1 atom 6 atom 8'
    """
    base_cell = entry['atoms'][0][1]
    parts = ['cobiBetween']
    for label, cell in entry['atoms']:
        _, idx0 = _parse_label(label)
        parts.extend(['atom', str(idx0 + 1)])
        rel = [c - b for c, b in zip(cell, base_cell)]
        if any(rel):
            parts.extend(['cell', str(rel[0]), str(rel[1]), str(rel[2])])
    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Orbital grouping / filtering — orbitalwise chains produce n_orbitals**N
# rows (hundreds to tens of thousands); callers must group and/or filter
# before displaying them.
# ---------------------------------------------------------------------------

def orbital_type(orbital_tag: str) -> str:
    """
    Coarse orbital type ('s', 'p', 'd', 'f', ...) for an orbital tag.

    '5s' -> 's', '5p_z' -> 'p', '4d_xy' -> 'd', '4d_x^2-y^2' -> 'd'
    """
    stripped = orbital_tag.lstrip('0123456789')
    return stripped[0] if stripped else orbital_tag


def group_orbital_curves_by_type(orbital_curves) -> list:
    """
    Collapse individual m-resolved orbital rows (e.g. '5p_x','5p_y','5p_z')
    into coarse orbital-type groups (e.g. 'p') by summing their values.

    This is the default view for browsing an orbitalwise chain: a 3-centre
    chain with an s/p/d basis (9 orbitals/atom) has 729 individual rows but
    only 3**3 = 27 orbital-type groups.

    Parameters
    ----------
    orbital_curves : list of dicts, each at minimum {orbitals, ival_ef} —
                     either energy-resolved rows as in
                     load_nc_entry_orbital_curves()['orbitals'] (which also
                     carry 'energy'/'curve'/'icurve', summed if present) or
                     integrated-only rows as in
                     parse_ncicobi_list()[i]['orbital_rows'] (no curve data)

    Returns
    -------
    list of dicts {orbital_types: tuple[str], ival_ef, n_rows: int (how many
    individual rows were summed), plus 'energy'/'curve'/'icurve' if the
    input rows carried them}, sorted by |ival_ef| descending.
    """
    groups: dict = {}
    for row in orbital_curves:
        key = tuple(orbital_type(o) for o in row['orbitals'])
        has_curve = 'curve' in row and row['curve'] is not None
        g = groups.get(key)
        if g is None:
            g = {'orbital_types': key, 'ival_ef': row['ival_ef'], 'n_rows': 1}
            if has_curve:
                g['energy'] = row['energy']
                g['curve']  = row['curve'].copy()
                g['icurve'] = row['icurve'].copy()
            groups[key] = g
        else:
            g['ival_ef'] = g['ival_ef'] + row['ival_ef']
            g['n_rows']  += 1
            if has_curve:
                g['curve']  = g['curve']  + row['curve']
                g['icurve'] = g['icurve'] + row['icurve']
    result = list(groups.values())
    result.sort(key=lambda x: abs(x['ival_ef']), reverse=True)
    return result


def filter_orbital_curves(curves, threshold: float) -> list:
    """
    Drop rows/groups whose |ival_ef| is below *threshold* — the "hide
    contributions with nothing going on" filter.

    Works on either individual orbital rows (from load_nc_entry_orbital_curves)
    or orbital-type groups (from group_orbital_curves_by_type); both carry
    'ival_ef'.

    Parameters
    ----------
    curves    : list of dicts, each with an 'ival_ef' key
    threshold : float, minimum |ival_ef| to keep (absolute ICOBI units)

    Returns
    -------
    list, same dicts, filtered; order preserved.
    """
    return [c for c in curves if abs(c['ival_ef']) >= threshold]


def find_lobster_dir(ph_dir) -> 'Path | None':
    """
    Infer the sibling LOBSTER directory from a phonopy directory.

    Convention: {parent}/{stem}_ph  →  {parent}/{stem}_lobster

    Returns the Path if it exists, otherwise None.
    """
    ph = Path(ph_dir).resolve()
    if ph.name.endswith('_ph'):
        candidate = ph.parent / (ph.name[:-3] + '_lobster')
        if candidate.is_dir():
            return candidate
    return None
