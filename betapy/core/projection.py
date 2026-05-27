"""
Projected force constant (pFC) calculations for betapy.

All functions take a Supercell instance and raw force-constant data.
No file I/O, no UI concerns live here.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from betapy.core.constants import PFC_ROUNDING_DECIMALS


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _unit_vector(v):
    """Return unit vector of v, or zero vector if v is zero."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros(3)
    return v / norm


def _project_fc_matrix(fc_matrix, direction):
    """
    Project a 3x3 force-constant matrix along a unit direction vector.

    Returns (mean, rms) of the forward and transpose projections,
    matching the convention of the original script.
    """
    fc = np.asarray(fc_matrix)
    pfc          = fc @ direction
    pfc_transp   = fc.T @ direction
    mean = np.mean([np.linalg.norm(pfc), np.linalg.norm(pfc_transp)])
    rms  = np.sqrt(np.mean([
        np.linalg.norm(pfc)**2,
        np.linalg.norm(pfc_transp)**2,
    ]))
    return mean, rms


def _onsite_norm(fc_matrix):
    """
    For on-site (self-interaction) terms, return row norms and their mean.
    """
    norms = [np.linalg.norm(row) for row in fc_matrix]
    return norms, np.mean(norms)


# ---------------------------------------------------------------------------
# Bulk pFC: project along interatomic bond vectors
# ---------------------------------------------------------------------------

def compute_bulk_pfcs(supercell, atomic_pairs, force_matrices,
                      show_progress=True, progress_callback=None):
    """
    Separate on-site and off-site pairs, project off-site FCs along bond vectors.

    Parameters
    ----------
    supercell         : Supercell instance
    atomic_pairs      : list of [i, j] 1-based index pairs
    force_matrices    : list of (3,3) arrays, one per pair
    show_progress     : bool, show tqdm progress bar (default True)
    progress_callback : callable(n, total) or None
        Called periodically with the current pair count and total.
        Throttled to ~200 calls regardless of dataset size so callers
        (e.g. a Qt progress bar) are not flooded with events.

    Returns
    -------
    results : list of dicts, one per off-site pair, with keys:
        atom1_idx, atom2_idx, species1, species2,
        distance, mean_pfc, rms_pfc
    onsite  : list of dicts, one per on-site pair, with keys:
        atom_idx, species, mean_norm
    distances : list of floats, Angstrom, one per pair (on- and off-site)
    """
    results = []
    onsite  = []
    distances = []

    total = len(atomic_pairs)
    _cb_step = max(1, total // 200)  # ~200 GUI updates regardless of size

    for k, (pair, fc_mat) in enumerate(tqdm(zip(atomic_pairs, force_matrices),
                              total=total,
                              desc='bulk pFCs',
                              unit='pair',
                              disable=not show_progress)):
        if progress_callback is not None and k % _cb_step == 0:
            progress_callback(k, total)
        i, j = pair
        if i == j:
            # On-site term
            _, mean_norm = _onsite_norm(fc_mat)
            onsite.append({
                'atom_idx': i,
                'species': supercell.species(i),
                'mean_norm': mean_norm,
            })
            distances.append(0.0)
        else:
            # Off-site term: project along the i→j bond vector
            pos_i = supercell.positions[i - 1]
            pos_j = supercell.positions[j - 1]
            cart_vec = supercell.cart_diff(pos_i, pos_j)
            dist     = float(np.linalg.norm(cart_vec))
            direction = _unit_vector(cart_vec)

            mean_pfc, rms_pfc = _project_fc_matrix(fc_mat, direction)
            results.append({
                'atom1_idx': i,
                'atom2_idx': j,
                'species1':  supercell.species(i),
                'species2':  supercell.species(j),
                'distance':  dist,
                'mean_pfc':  mean_pfc,
                'rms_pfc':   rms_pfc,
            })
            distances.append(dist)

    if progress_callback is not None:
        progress_callback(total, total)

    return results, onsite, distances


def unique_pfcs(bulk_results):
    """
    Find unique pFC values (rounded to 5 decimal places) from bulk results.
    Uniqueness is determined per species-pair group: a Ge-Te value of 1.0 and
    a Ge-Ge value of 1.0 are distinct and both appear in the output.

    Returns a DataFrame with columns:
        atom1_idx, species1, atom2_idx, species2, distance, pfc_value
    Rows are sorted by species pair, then by pFC value within each pair.
    """
    if not bulk_results:
        return pd.DataFrame()

    from collections import defaultdict
    groups = defaultdict(list)
    for r in bulk_results:
        groups[(r['species1'], r['species2'])].append(r)

    rows = []
    for (sp1, sp2) in sorted(groups.keys()):
        records  = groups[(sp1, sp2)]
        pfc_vals = np.array([r['mean_pfc'] for r in records])
        rounded  = np.around(pfc_vals, PFC_ROUNDING_DECIMALS)
        _, indices, _ = np.unique(rounded, return_index=True, return_counts=True)
        for idx in indices:
            r = records[idx]
            rows.append([
                r['atom1_idx'], r['species1'],
                r['atom2_idx'], r['species2'],
                r['distance'],  r['mean_pfc'],
            ])

    return pd.DataFrame(rows, columns=[
        'Index 1', 'Atom 1', 'Index 2', 'Atom 2',
        'Distance (Angstr.)', 'pFC value',
    ])


# ---------------------------------------------------------------------------
# Shell grouping: aggregate symmetry-equivalent bonds by distance
# ---------------------------------------------------------------------------

def group_by_shells(results, dist_precision=0.01, max_distance=None):
    """
    Group off-site pFC results into distance shells.

    Two pairs belong to the same shell when their normalised (species1,
    species2) pair type matches and their distance rounds to the same bin
    at the given precision (default 0.01 A).  Species are always stored
    in alphabetical order so that (V, O) and (O, V) records — which arise
    when phonopy FORCE_CONSTANTS contains the full N×N matrix — are
    merged into one shell rather than kept as duplicates.

    Parameters
    ----------
    results        : list of dicts from compute_bulk_pfcs() or find_refsite_pairs()
    dist_precision : float, A, binning step (default 0.01)
    max_distance   : float or None
        If given, pairs beyond this distance are excluded.  Pass the
        half-cell reliability cutoff to keep the shell plot clean when
        full (N×N) force constants are used.

    Returns
    -------
    shells : list of dicts, each with keys:
        species1, species2,
        distance_mean, distance_std,
        pfc_mean, pfc_std, pfc_min, pfc_max,
        count, records
    Sorted by (species1, species2, distance_mean).
    """
    from collections import defaultdict
    bins = defaultdict(list)
    for r in results:
        if max_distance is not None and r['distance'] > max_distance:
            continue
        # Normalise species order so (A,B) and (B,A) land in the same bin.
        # Swap atom indices too so rep-atom logic in the GUI stays consistent.
        sp1, sp2 = r['species1'], r['species2']
        if sp1 > sp2:
            sp1, sp2 = sp2, sp1
            rec = {**r, 'species1': sp1, 'species2': sp2,
                   'atom1_idx': r['atom2_idx'], 'atom2_idx': r['atom1_idx']}
        else:
            rec = r
        d_bin = round(r['distance'] / dist_precision) * dist_precision
        bins[(sp1, sp2, d_bin)].append(rec)

    shells = []
    for (sp1, sp2, _), records in bins.items():
        pfcs  = np.array([rec['mean_pfc'] for rec in records])
        dists = np.array([rec['distance']  for rec in records])
        shells.append({
            'species1':      sp1,
            'species2':      sp2,
            'distance_mean': float(np.mean(dists)),
            'distance_std':  float(np.std(dists)),
            'pfc_mean':      float(np.mean(pfcs)),
            'pfc_std':       float(np.std(pfcs)),
            'pfc_min':       float(np.min(pfcs)),
            'pfc_max':       float(np.max(pfcs)),
            'count':         len(records),
            'records':       records,
        })
    shells.sort(key=lambda s: (s['species1'], s['species2'], s['distance_mean']))
    return shells


# ---------------------------------------------------------------------------
# Reference-site pFC: project along atom→refsite vectors
# ---------------------------------------------------------------------------

def find_refsite_pairs(supercell, atomic_pairs, force_matrices,
                       refsite_frac, cutoff, min_distance=0.0,
                       exclude_species=None, show_progress=True):
    """
    Find all pairs where both atoms are within `cutoff` Angstrom of
    `refsite_frac` (a fractional coordinate), and project their FCs
    along the atom1 → refsite vector.

    Parameters
    ----------
    supercell        : Supercell instance
    atomic_pairs     : list of [i, j] 1-based pairs
    force_matrices   : list of (3,3) arrays
    refsite_frac     : array-like (3,), fractional coordinates of reference site
    cutoff           : float, Angstrom
    min_distance     : float, Angstrom (default 0.0)
        Atoms closer than this to the reference site are excluded entirely.
        Use 0.1 Å for the stiffness-shift intercalated structure to exclude
        the site-occupying Li atom without affecting any real neighbours.
    exclude_species  : iterable of str or None (default None)
        Off-site pairs where either atom belongs to one of these species are
        dropped.  Typically set to the species that occupies the reference site
        so that the host-framework projection is not polluted by self-species
        contributions.

    Returns
    -------
    offsite_results : list of dicts
    onsite_results  : list of dicts
    """
    refsite_frac = np.asarray(refsite_frac)
    _excl = set(exclude_species) if exclude_species else set()
    offsite_results = []
    onsite_results  = []

    for pair, fc_mat in tqdm(zip(atomic_pairs, force_matrices),
                              total=len(atomic_pairs),
                              desc='refsite pFCs',
                              unit='pair',
                              disable=not show_progress):
        i, j = pair
        dist_i = supercell.distance_to_point(i, refsite_frac)
        dist_j = supercell.distance_to_point(j, refsite_frac)

        # Exclude any pair where either atom is the site-occupying atom
        if dist_i < min_distance or dist_j < min_distance:
            continue

        if dist_i > cutoff or dist_j > cutoff:
            continue

        if i == j:
            # On-site term
            _, mean_norm = _onsite_norm(fc_mat)
            onsite_results.append({
                'atom_idx':      i,
                'species':       supercell.species(i),
                'atom_ref_dist': dist_i,
                'mean_norm':     mean_norm,
            })
        else:
            # Off-site term: project along atom1 → refsite
            sp_i = supercell.species(i)
            sp_j = supercell.species(j)
            if _excl and (sp_i in _excl or sp_j in _excl):
                continue

            vec_to_ref = supercell.cart_vector_to_point(i, refsite_frac)
            direction  = _unit_vector(vec_to_ref)

            atom_dist = supercell.atom_distance(i, j)
            mean_pfc, rms_pfc = _project_fc_matrix(fc_mat, direction)
            offsite_results.append({
                'atom1_idx':      i,
                'atom2_idx':      j,
                'species1':       sp_i,
                'species2':       sp_j,
                'atom1_ref_dist': dist_i,
                'distance':       atom_dist,
                'mean_pfc':       mean_pfc,
                'rms_pfc':        rms_pfc,
            })

    return offsite_results, onsite_results


def refsite_results_to_dataframes(offsite_results, onsite_results, ref_label):
    """
    Convert refsite result lists into tidy DataFrames ready for output.

    Parameters
    ----------
    offsite_results : list of dicts from find_refsite_pairs
    onsite_results  : list of dicts from find_refsite_pairs
    ref_label       : str, label from the REFPOS file

    Returns
    -------
    df_offsite, df_onsite : pd.DataFrame
    """
    offsite_rows = []
    for r in offsite_results:
        offsite_rows.append([
            ref_label,
            r['atom1_idx'], r['species1'],
            r['atom2_idx'], r['species2'],
            r['atom1_ref_dist'],
            r['distance'],
            r['mean_pfc'], r['rms_pfc'],
        ])
    df_offsite = pd.DataFrame(offsite_rows, columns=[
        'Ref Label',
        'Atom1 Index', 'Atom1 Type',
        'Atom2 Index', 'Atom2 Type',
        'Atom1-Ref Distance (Angstr.)',
        'Atom-Atom Distance (Angstr.)',
        'Mean pFC value', 'RMS pFC value',
    ])

    onsite_rows = []
    for r in onsite_results:
        onsite_rows.append([
            ref_label,
            r['atom_idx'], r['species'],
            r['atom_ref_dist'],
            r['mean_norm'],
        ])
    df_onsite = pd.DataFrame(onsite_rows, columns=[
        'Ref Label',
        'Atom Index', 'Atom Type',
        'Atom-Ref Distance (Angstr.)',
        'Mean Norm pFC value',
    ])

    return df_offsite, df_onsite


# ---------------------------------------------------------------------------
# Atom matching across structures for stiffness-shift comparison
# ---------------------------------------------------------------------------

def match_atoms_across_structures(sc_a, sc_b, species, tolerance=0.1):
    """
    For each atom of `species` in sc_a, find the closest atom of the
    same species in sc_b by fractional-coordinate distance (PBC-aware).
    Each atom in sc_b can only be matched once (greedy nearest-neighbour).

    Parameters
    ----------
    sc_a, sc_b  : Supercell instances
    species     : str, chemical symbol to match (e.g. 'V', 'O')
    tolerance   : float, maximum fractional-coordinate distance (dimensionless,
                  default 0.1) to accept as a valid match.
                  Using fractional rather than Cartesian distance makes the
                  criterion invariant to cell-parameter changes between A and B
                  (e.g. cell expansion on intercalation).

    Returns
    -------
    matches   : dict {idx_a (1-based): idx_b (1-based)}
    unmatched : list of idx_a (1-based) that had no match within tolerance
    """
    atoms_a = [(i + 1, sc_a.positions[i])
               for i in range(sc_a.n_atoms)
               if sc_a.species(i + 1) == species]
    atoms_b = [(i + 1, sc_b.positions[i])
               for i in range(sc_b.n_atoms)
               if sc_b.species(i + 1) == species]

    if not atoms_b:
        # Species entirely absent in sc_b (e.g. Li in deintercalated)
        return {}, [idx for idx, _ in atoms_a]

    # Build cost matrix (fractional distances) and use optimal assignment
    # (Hungarian algorithm) to avoid greedy artifacts when atom counts match.
    pa = np.array([p for _, p in atoms_a])   # (Na, 3)
    pb = np.array([p for _, p in atoms_b])   # (Nb, 3)
    diff = pb[None, :, :] - pa[:, None, :]   # (Na, Nb, 3)
    diff -= np.floor(diff + 0.5)
    cost = np.linalg.norm(diff, axis=2)       # (Na, Nb)

    row_ind, col_ind = linear_sum_assignment(cost)
    row_to_col = dict(zip(row_ind.tolist(), col_ind.tolist()))

    matches   = {}
    unmatched = []
    for i, (idx_a, _) in enumerate(atoms_a):
        if i not in row_to_col:
            unmatched.append(idx_a)
        else:
            j = row_to_col[i]
            if cost[i, j] <= tolerance:
                matches[idx_a] = atoms_b[j][0]
            else:
                unmatched.append(idx_a)

    return matches, unmatched


def match_atoms_local(offsite_a, offsite_b, sc_a, sc_b,
                      refsite_a, refsite_b, tolerance=0.05):
    """
    Match atoms appearing in offsite_a to atoms in offsite_b using their
    LOCAL fractional displacement from the respective reference sites.

    This is origin-independent: it works even when A and B were computed
    with different unit-cell origins (common for intercalation pairs), because
    it compares displacement RELATIVE to the refsite rather than absolute
    fractional coordinates.  The same fractional tolerance applies to all
    three axes but is harmless: within the small local neighbourhood of the
    refsite (typically ≤ cutoff/cell_param ≈ 0.25), correct atom pairs
    are separated by <<0.05 and wrong candidates are ≥0.1 away.

    Parameters
    ----------
    offsite_a, offsite_b : list of dicts from find_refsite_pairs()
    sc_a, sc_b           : Supercell instances for structures A and B
    refsite_a, refsite_b : array-like (3,), fractional refsite coords in A / B
    tolerance            : float, max fractional displacement distance for a
                           valid match (default 0.05)

    Returns
    -------
    matches : dict {atom_idx_a (1-based): atom_idx_b (1-based)}
    """
    refsite_a = np.asarray(refsite_a)
    refsite_b = np.asarray(refsite_b)

    # Collect unique atoms from each offsite list and compute local displacements
    def _local_disps(offsite, sc, refsite):
        atoms = {}
        for r in offsite:
            for idx in (r['atom1_idx'], r['atom2_idx']):
                if idx not in atoms:
                    pos = np.asarray(sc.positions[idx - 1])
                    d = pos - refsite
                    d -= np.floor(d + 0.5)
                    atoms[idx] = d
        return atoms

    a_atoms = _local_disps(offsite_a, sc_a, refsite_a)
    b_atoms = _local_disps(offsite_b, sc_b, refsite_b)

    if not a_atoms or not b_atoms:
        return {}

    a_ids = list(a_atoms.keys())
    b_ids = list(b_atoms.keys())

    # Build cost matrix of pairwise fractional displacement distances
    a_arr = np.array([a_atoms[i] for i in a_ids])   # (Na, 3)
    b_arr = np.array([b_atoms[j] for j in b_ids])   # (Nb, 3)
    diff = b_arr[None, :, :] - a_arr[:, None, :]    # (Na, Nb, 3)
    diff -= np.floor(diff + 0.5)
    cost = np.linalg.norm(diff, axis=2)              # (Na, Nb)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[r, c] <= tolerance:
            matches[a_ids[r]] = b_ids[c]

    return matches


def match_fc_pairs_direct(results_a, results_b, sc_a, sc_b,
                          refsite_a, refsite_b, tol=0.05):
    """
    Match off-site pFC pairs directly by physical fingerprint:
    ordered species pair + local fractional displacement of BOTH atoms
    from the respective reference sites.

    Origin-independent: works even when A and B were relaxed with different
    cell settings or origins, because it compares positions relative to the
    refsite rather than global atom indices.

    Step 1 — match atom1 indices per species via Hungarian algorithm on
             local fractional displacement.
    Step 2 — for each matched atom1 pair, match atom2 by species and local
             displacement within that small group (another small Hungarian).

    Parameters
    ----------
    results_a, results_b : list of dicts from find_refsite_pairs()
    sc_a, sc_b           : Supercell instances
    refsite_a, refsite_b : array-like (3,), fractional refsite coords in A / B
    tol                  : float, max fractional displacement distance to
                           accept for both atom1 and atom2 (default 0.05)

    Returns
    -------
    matched_pairs, unmatched_a, unmatched_b  (same format as match_fc_pairs)
    """
    if not results_a or not results_b:
        return [], list(results_a), list(results_b)

    refsite_a = np.asarray(refsite_a)
    refsite_b = np.asarray(refsite_b)

    def _ld(sc, idx, ref):
        d = np.asarray(sc.positions[idx - 1]) - ref
        d -= np.floor(d + 0.5)
        return d

    # Cache local displacements for every atom appearing in either result set
    ca = {}
    for r in results_a:
        for k in (r['atom1_idx'], r['atom2_idx']):
            if k not in ca:
                ca[k] = _ld(sc_a, k, refsite_a)
    cb = {}
    for r in results_b:
        for k in (r['atom1_idx'], r['atom2_idx']):
            if k not in cb:
                cb[k] = _ld(sc_b, k, refsite_b)

    # Step 1: match distinct atom1 indices per species (small problem)
    a1_sp = {r['atom1_idx']: r['species1'] for r in results_a}
    b1_sp = {r['atom1_idx']: r['species1'] for r in results_b}
    atom1_match = {}
    for sp in set(a1_sp.values()) & set(b1_sp.values()):
        a_ids = [i for i, s in a1_sp.items() if s == sp]
        b_ids = [j for j, s in b1_sp.items() if s == sp]
        pa = np.array([ca[i] for i in a_ids])
        pb = np.array([cb[j] for j in b_ids])
        diff = pb[None] - pa[:, None]
        diff -= np.floor(diff + 0.5)
        cost = np.linalg.norm(diff, axis=2)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri.tolist(), ci.tolist()):
            if cost[r, c] <= tol:
                atom1_match[a_ids[r]] = b_ids[c]

    # Step 2: within each matched atom1 pair, match atom2 per species2
    a_by_a1 = {}
    for i, r in enumerate(results_a):
        a_by_a1.setdefault(r['atom1_idx'], []).append((i, r))
    b_by_a1 = {}
    for j, r in enumerate(results_b):
        b_by_a1.setdefault(r['atom1_idx'], []).append((j, r))

    a_to_b = {}   # orig index in results_a -> orig index in results_b
    for ia, ib in atom1_match.items():
        a_pairs = a_by_a1.get(ia, [])
        b_pairs = b_by_a1.get(ib, [])
        if not a_pairs or not b_pairs:
            continue
        a_by_sp2 = {}
        for idx, r in a_pairs:
            a_by_sp2.setdefault(r['species2'], []).append((idx, r))
        b_by_sp2 = {}
        for idx, r in b_pairs:
            b_by_sp2.setdefault(r['species2'], []).append((idx, r))
        for sp2 in set(a_by_sp2) & set(b_by_sp2):
            ag = a_by_sp2[sp2]
            bg = b_by_sp2[sp2]
            pa2 = np.array([ca[r['atom2_idx']] for _, r in ag])
            pb2 = np.array([cb[r['atom2_idx']] for _, r in bg])
            diff = pb2[None] - pa2[:, None]
            diff -= np.floor(diff + 0.5)
            cost = np.linalg.norm(diff, axis=2)
            ri, ci = linear_sum_assignment(cost)
            for r2, c2 in zip(ri.tolist(), ci.tolist()):
                if cost[r2, c2] <= tol:
                    a_to_b[ag[r2][0]] = bg[c2][0]

    # Assemble output in the same format as match_fc_pairs
    matched_pairs = []
    unmatched_a   = []
    matched_b_set = set(a_to_b.values())

    for i, ra in enumerate(results_a):
        if i not in a_to_b:
            unmatched_a.append(ra)
            continue
        rb = results_b[a_to_b[i]]
        matched_pairs.append({
            'atom1_idx_a':      ra['atom1_idx'],
            'atom2_idx_a':      ra['atom2_idx'],
            'atom1_idx_b':      rb['atom1_idx'],
            'atom2_idx_b':      rb['atom2_idx'],
            'species1':         ra['species1'],
            'species2':         ra['species2'],
            'distance_a':       ra['distance'],
            'distance_b':       rb['distance'],
            'atom1_ref_dist_a': ra.get('atom1_ref_dist', 0.0),
            'atom1_ref_dist_b': rb.get('atom1_ref_dist', 0.0),
            'mean_pfc_a':       ra['mean_pfc'],
            'mean_pfc_b':       rb['mean_pfc'],
            'delta_pfc':        rb['mean_pfc'] - ra['mean_pfc'],
        })

    unmatched_b = [results_b[j] for j in range(len(results_b))
                   if j not in matched_b_set]

    return matched_pairs, unmatched_a, unmatched_b


def match_fc_pairs(results_a, results_b, atom_matches, sc_a):
    """
    Match off-site force-constant pairs across two structures using
    pre-computed atom-index matches.

    For each pair (i, j) in results_a, look for the pair (i', j') in
    results_b where i' = atom_matches[i] and j' = atom_matches[j].
    Only pairs where both atoms have a valid match are included.
    Pairs involving species absent in sc_b (e.g. Li-containing pairs
    in the intercalated → deintercalated direction) are skipped silently.

    Parameters
    ----------
    results_a    : list of dicts from find_refsite_pairs() for structure A
    results_b    : list of dicts from find_refsite_pairs() for structure B
    atom_matches : dict {idx_a: idx_b} from match_atoms_across_structures()
                   covering all relevant species
    sc_a         : Supercell for structure A (used for distance lookup)

    Returns
    -------
    matched_pairs : list of dicts, each with keys:
        atom1_idx_a, atom2_idx_a, atom1_idx_b, atom2_idx_b,
        species1, species2, distance_a, distance_b,
        atom1_ref_dist_a, atom1_ref_dist_b,
        mean_pfc_a, mean_pfc_b, delta_pfc
    unmatched_a   : list of dicts from results_a with no counterpart in B
    unmatched_b   : list of dicts from results_b with no counterpart in A
    """
    # Build a fast lookup for results_b: (idx1, idx2) -> result dict
    lookup_b = {}
    for r in results_b:
        lookup_b[(r['atom1_idx'], r['atom2_idx'])] = r

    matched_pairs = []
    unmatched_a   = []

    for r in results_a:
        i, j = r['atom1_idx'], r['atom2_idx']

        if i not in atom_matches or j not in atom_matches:
            unmatched_a.append(r)
            continue

        i_b = atom_matches[i]
        j_b = atom_matches[j]

        counterpart = lookup_b.get((i_b, j_b))
        if counterpart is None:
            unmatched_a.append(r)
            continue

        dist_a = r['distance']
        dist_b = counterpart['distance']
        pfc_a  = r['mean_pfc']
        pfc_b  = counterpart['mean_pfc']

        matched_pairs.append({
            'atom1_idx_a':      i,
            'atom2_idx_a':      j,
            'atom1_idx_b':      i_b,
            'atom2_idx_b':      j_b,
            'species1':         r['species1'],
            'species2':         r['species2'],
            'distance_a':       dist_a,
            'distance_b':       dist_b,
            'atom1_ref_dist_a': r.get('atom1_ref_dist', 0.0),
            'atom1_ref_dist_b': counterpart.get('atom1_ref_dist', 0.0),
            'mean_pfc_a':       pfc_a,
            'mean_pfc_b':       pfc_b,
            'delta_pfc':        pfc_b - pfc_a,
        })

    matched_b_keys = {(m['atom1_idx_b'], m['atom2_idx_b']) for m in matched_pairs}
    unmatched_b = [r for r in results_b
                   if (r['atom1_idx'], r['atom2_idx']) not in matched_b_keys]

    return matched_pairs, unmatched_a, unmatched_b


def stiffness_shift_from_pairs(matched_pairs):
    """
    Compute total and per-species-pair stiffness shift from matched pairs.

    Parameters
    ----------
    matched_pairs : list of dicts from match_fc_pairs()

    Returns
    -------
    df          : pd.DataFrame with one row per matched pair
    total_shift : float, sum of all delta_pfc values
    """
    if not matched_pairs:
        return pd.DataFrame(), 0.0

    df = pd.DataFrame(matched_pairs)
    total_shift = float(df['delta_pfc'].sum())
    return df, total_shift


def fallback_equal_count_shift(results_a, results_b):
    """
    Fallback stiffness shift when atom matching fails.
    Sorts both result sets by atom-atom distance and truncates to equal
    length, then subtracts pFC sums. Returns (df, total_shift, n_pairs).
    This is the original workaround — called only when position matching
    fails and the program warns the user.
    """
    def sorted_pfcs(results):
        return sorted(results, key=lambda r: r['distance'])

    sorted_a = sorted_pfcs(results_a)
    sorted_b = sorted_pfcs(results_b)
    n = min(len(sorted_a), len(sorted_b))

    rows = []
    for ra, rb in zip(sorted_a[:n], sorted_b[:n]):
        pfc_a = ra['mean_pfc']
        pfc_b = rb['mean_pfc']
        rows.append({
            'species1':   ra['species1'],
            'species2':   ra['species2'],
            'distance_a': ra['distance'],
            'distance_b': rb['distance'],
            'mean_pfc_a': pfc_a,
            'mean_pfc_b': pfc_b,
            'delta_pfc':  pfc_b - pfc_a,
        })

    df = pd.DataFrame(rows)
    return df, float(df['delta_pfc'].sum()), n
