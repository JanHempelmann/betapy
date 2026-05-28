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
    result = lobster.load_car_curves(lobster_dir / 'COHPCAR.lobster',
                                     hdr, 'Sc', 'F', 2.012)
    if result:
        energy, cohp, icohp = result

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
        spread = max(vals) - min(vals)
        # None signals ambiguity: same (species, distance) but distinct LOBSTER
        # values, meaning two structurally inequivalent bond environments happen
        # to share the same interatomic distance.  lookup() returns None for
        # these rather than an incorrect average.
        value = None if spread > _VAL_TOL else sum(vals) / len(vals)
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
            index      : int (1-based, matches No.k label)
            sp1, sp2   : str
            distance   : float or None (None for COBICAR until enriched)
            atm1, atm2 : str (full LOBSTER atom labels, e.g. 'Sc1')
            cell1, cell2: list[int] or None (COBICAR translation vectors)
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

    pairs = []
    for i in range(hdr_start, hdr_start + n_pairs):
        line = lines[i].strip()
        m = _re_dist.match(line)
        if m:
            sp1 = re.match(r'([A-Za-z]+)', m.group(2)).group(1)
            sp2 = re.match(r'([A-Za-z]+)', m.group(3)).group(1)
            pairs.append({
                'index': int(m.group(1)), 'sp1': sp1, 'sp2': sp2,
                'distance': float(m.group(4)),
                'atm1': m.group(2), 'atm2': m.group(3),
                'cell1': None, 'cell2': None,
            })
            continue
        m = _re_cell.match(line)
        if m:
            if '->' in line[m.end():]:   # 3-centre COBI — skip
                continue
            sp1 = re.match(r'([A-Za-z]+)', m.group(2)).group(1)
            sp2 = re.match(r'([A-Za-z]+)', m.group(4)).group(1)
            pairs.append({
                'index': int(m.group(1)), 'sp1': sp1, 'sp2': sp2,
                'distance': None,
                'atm1': m.group(2), 'atm2': m.group(4),
                'cell1': list(map(int, m.group(3).split())),
                'cell2': list(map(int, m.group(5).split())),
            })

    # Skip blank lines between header and data
    first_data = hdr_start + n_pairs
    while first_data < len(lines) and not lines[first_data].strip():
        first_data += 1

    return {
        'n_spins': n_spins, 'n_e': n_e, 'e_fermi': e_fermi,
        'has_average': has_average, 'n_total': n_total, 'n_pairs': n_pairs,
        'first_data_line': first_data, 'pairs': pairs,
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
                    tol: float = 0.05):
    """
    Load energy-resolved curves for the given species pair from a *CAR.lobster.

    All pairs in *header* that match (sp1, sp2, distance ± tol) are averaged
    — within a symmetry shell they are identical, so averaging is safe.

    Parameters
    ----------
    path    : path to the *CAR.lobster file
    header  : output of parse_car_header() (optionally enriched with distances)
    sp1, sp2: species strings (order-independent)
    distance: target bond length in Å
    tol     : matching tolerance in Å

    Returns
    -------
    (energy, curve, icurve) as 1-D numpy arrays, or None if no match.
    energy  — eV relative to Fermi level
    curve   — COHP / COOP / COBI value (eV⁻¹ or dimensionless)
    icurve  — integrated value up to each energy point
    """
    cs1, cs2 = _canonical(sp1, sp2)
    matching_indices = []
    for p in header['pairs']:
        if p['distance'] is None:
            continue
        if _canonical(p['sp1'], p['sp2']) != (cs1, cs2):
            continue
        if abs(p['distance'] - distance) <= tol:
            matching_indices.append(p['index'])

    if not matching_indices:
        return None

    n_spins   = header['n_spins']
    has_avg   = header['has_average']

    def _entry_cols(pair_idx):
        """0-based data column indices for pair No.pair_idx."""
        entry = pair_idx if has_avg else pair_idx - 1
        base  = 1 + entry * 2 * n_spins
        return base, base + 1

    usecols  = [0]
    slot_map = []  # [(curve_slot, icurve_slot), ...] into usecols
    for idx in matching_indices:
        c, ic = _entry_cols(idx)
        slot_map.append((len(usecols), len(usecols) + 1))
        usecols.extend([c, ic])

    try:
        data = np.loadtxt(path, skiprows=header['first_data_line'],
                          usecols=usecols)
    except Exception:
        return None

    energy  = data[:, 0]
    curves  = np.mean([data[:, s[0]] for s in slot_map], axis=0)
    icurves = np.mean([data[:, s[1]] for s in slot_map], axis=0)
    return energy, curves, icurves


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
