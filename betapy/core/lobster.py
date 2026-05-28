"""
Parser and lookup utilities for LOBSTER output files.

Supported files
---------------
ICOBILIST.lobster  – integrated COBI per bond
ICOHPLIST.lobster  – integrated COHP per bond
ICOOPLIST.lobster  – integrated COOP per bond
CHARGE.lobster     – Mulliken / Löwdin charges per atom

Typical workflow
----------------
    from betapy.core import lobster

    pairs = lobster.load_pairs(lobster_dir)
    val   = lobster.lookup(pairs, 'Sc', 'F', 2.012, key='icobi')

    charges = lobster.parse_charges(lobster_dir / 'CHARGE.lobster')

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
"""

import re
from pathlib import Path


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
