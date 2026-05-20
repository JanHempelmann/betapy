"""
Projected force constant (pFC) calculations for betapy.

All functions take a Supercell instance and raw force-constant data.
No file I/O, no UI concerns live here.
"""

import numpy as np
import pandas as pd


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

def compute_bulk_pfcs(supercell, atomic_pairs, force_matrices):
    """
    Separate on-site and off-site pairs, project off-site FCs along bond vectors.

    Parameters
    ----------
    supercell      : Supercell instance
    atomic_pairs   : list of [i, j] 1-based index pairs
    force_matrices : list of (3,3) arrays, one per pair

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

    for pair, fc_mat in zip(atomic_pairs, force_matrices):
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

    return results, onsite, distances


def unique_pfcs(bulk_results):
    """
    Find unique pFC values (rounded to 5 decimal places) from bulk results.

    Returns a DataFrame with columns:
        atom1_idx, species1, atom2_idx, species2, distance, pfc_value
    """
    if not bulk_results:
        return pd.DataFrame()

    pfc_vals = np.array([r['mean_pfc'] for r in bulk_results])
    rounded  = np.around(pfc_vals, 5)
    _, indices, _ = np.unique(rounded, return_index=True, return_counts=True)

    rows = []
    for idx in indices:
        r = bulk_results[idx]
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
# Reference-site pFC: project along atom→refsite vectors
# ---------------------------------------------------------------------------

def find_refsite_pairs(supercell, atomic_pairs, force_matrices,
                       refsite_frac, cutoff, min_distance=0.0):
    """
    Find all pairs where both atoms are within `cutoff` Angstrom of
    `refsite_frac` (a fractional coordinate), and project their FCs
    along the atom1 → refsite vector.

    Parameters
    ----------
    supercell     : Supercell instance
    atomic_pairs  : list of [i, j] 1-based pairs
    force_matrices: list of (3,3) arrays
    refsite_frac  : array-like (3,), fractional coordinates of reference site
    cutoff        : float, Angstrom
    min_distance  : float, Angstrom (default 0.0)
        Atoms closer than this to the reference site are excluded entirely.
        Use 0.1 Å for the stiffness-shift intercalated structure to exclude
        the site-occupying Li atom without affecting any real neighbours.

    Returns
    -------
    offsite_results : list of dicts
    onsite_results  : list of dicts
    """
    refsite_frac = np.asarray(refsite_frac)
    offsite_results = []
    onsite_results  = []

    for pair_idx, (pair, fc_mat) in enumerate(zip(atomic_pairs, force_matrices)):
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
            vec_to_ref = supercell.cart_vector_to_point(i, refsite_frac)
            direction  = _unit_vector(vec_to_ref)

            # Atom-atom distance
            atom_dist = supercell.atom_distance(i, j)

            mean_pfc, rms_pfc = _project_fc_matrix(fc_mat, direction)
            offsite_results.append({
                'atom1_idx':      i,
                'atom2_idx':      j,
                'species1':       supercell.species(i),
                'species2':       supercell.species(j),
                'atom1_ref_dist': dist_i,
                'atom_distance':  atom_dist,
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
            r['atom_distance'],
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
# Coordination shell identification and stiffness-shift analysis
# ---------------------------------------------------------------------------

def identify_shells(bulk_results, gap_factor=4.0, min_gap_abs=0.15,
                    manual_edges=None):
    """
    Assign each pair in bulk_results to a coordination shell.

    Shell boundaries are found per species-pair type by looking for gaps in
    the sorted distance distribution that are simultaneously:
      - larger than gap_factor × median inter-point spacing (relative criterion)
      - larger than min_gap_abs Angstrom (absolute minimum, prevents
        over-splitting the quasi-continuous far-distance region)

    Parameters
    ----------
    bulk_results : list of dicts from compute_bulk_pfcs()
    gap_factor   : float, relative gap sensitivity (default 4.0)
    min_gap_abs  : float, minimum gap size in Angstrom to count as a shell
                   boundary (default 0.15 Å)
    manual_edges : dict or None
        Override automatic detection for specific species pairs.
        Format: {('V', 'O'): [0.0, 2.2, 2.9, 4.5], ...}
        Keys are (species1, species2) tuples; values are lists of bin edges
        in Angstrom. Species pairs not present in this dict are handled
        automatically.

    Returns
    -------
    shell_edges : dict mapping (species1, species2) -> list of edge values
    results_with_shells : list of dicts, same as bulk_results but each dict
        gains a 'shell' key: (species1, species2, shell_index_1based)
    """
    if manual_edges is None:
        manual_edges = {}

    # Collect distances per species pair.
    # Results from compute_bulk_pfcs use 'distance';
    # results from find_refsite_pairs use 'atom_distance'.
    from collections import defaultdict
    pair_dists = defaultdict(list)
    for r in bulk_results:
        key  = (r['species1'], r['species2'])
        dist = r.get('distance', r.get('atom_distance'))
        if dist is not None:
            pair_dists[key].append(dist)

    # Compute shell edges for each species pair
    shell_edges = {}
    for pair, dists in pair_dists.items():
        if pair in manual_edges:
            shell_edges[pair] = manual_edges[pair]
        else:
            shell_edges[pair] = _auto_shell_edges(
                dists, gap_factor, min_gap_abs
            )

    # Assign each result to a shell
    results_with_shells = []
    for r in bulk_results:
        pair  = (r['species1'], r['species2'])
        edges = shell_edges[pair]
        dist  = r.get('distance', r.get('atom_distance'))
        shell_idx = _shell_index(dist, edges)
        results_with_shells.append({**r, 'shell': (pair[0], pair[1], shell_idx)})

    return shell_edges, results_with_shells


def _auto_shell_edges(dists, gap_factor, min_gap_abs):
    """Find shell boundary edges for one species pair."""
    dists = np.array(sorted(dists))
    if len(dists) < 2:
        return [dists[0] - 0.01, dists[0] + 0.01]

    gaps    = np.diff(dists)
    nonzero = gaps[gaps > 0]
    if len(nonzero) == 0:
        return [dists[0] - 0.01, dists[-1] + 0.01]

    median_gap = np.median(nonzero)
    threshold  = gap_factor * median_gap

    # Boundary if gap exceeds both the relative AND the absolute threshold
    is_boundary = (gaps > threshold) & (gaps > min_gap_abs)
    boundary_idxs = np.where(is_boundary)[0]

    edges = [float(dists[0]) - 0.01]
    for b in boundary_idxs:
        edges.append(float((dists[b] + dists[b + 1]) / 2.0))
    edges.append(float(dists[-1]) + 0.01)
    return edges


def _shell_index(distance, edges):
    """Return 1-based shell index for a given distance and edge list."""
    for i in range(len(edges) - 1):
        if edges[i] < distance <= edges[i + 1]:
            return i + 1
    return len(edges) - 1   # fallback: last shell


def compute_shell_pfcs(results_with_shells):
    """
    Sum and count pFC values per coordination shell.

    Parameters
    ----------
    results_with_shells : list of dicts with 'shell' and 'mean_pfc' keys,
        as returned by identify_shells()

    Returns
    -------
    shell_sums : dict mapping shell_label -> {'sum': float, 'count': int,
                                               'mean': float}
        shell_label is a tuple (species1, species2, shell_index)
    """
    from collections import defaultdict
    accum = defaultdict(lambda: {'sum': 0.0, 'count': 0})
    for r in results_with_shells:
        label = r['shell']
        accum[label]['sum']   += r['mean_pfc']
        accum[label]['count'] += 1

    return {
        label: {
            'sum':   v['sum'],
            'count': v['count'],
            'mean':  v['sum'] / v['count'] if v['count'] > 0 else 0.0,
        }
        for label, v in accum.items()
    }


def stiffness_shift(shell_sums_a, shell_sums_b, label_a='A', label_b='B'):
    """
    Compute the per-shell and total stiffness shift between two structures.

    Compares only shells present in both structures. Shells unique to one
    structure are reported separately so you can see what was gained or lost.

    Parameters
    ----------
    shell_sums_a : dict from compute_shell_pfcs() for structure A
                   (e.g. fully deintercalated / vacancy projected)
    shell_sums_b : dict from compute_shell_pfcs() for structure B
                   (e.g. fully intercalated / occupied site projected)
    label_a, label_b : str, names for the two structures in the output

    Returns
    -------
    df : pd.DataFrame with columns:
        shell, species_pair, shell_index,
        sum_A, count_A, sum_B, count_B,
        delta (= sum_B - sum_A),
        status ('shared' | 'only_in_A' | 'only_in_B')
    total_shift : float, sum of delta over shared shells
    """
    all_shells = sorted(set(shell_sums_a) | set(shell_sums_b))
    rows = []
    total_shift = 0.0

    for shell in all_shells:
        sp1, sp2, idx = shell
        in_a = shell in shell_sums_a
        in_b = shell in shell_sums_b

        sum_a   = shell_sums_a[shell]['sum']   if in_a else None
        count_a = shell_sums_a[shell]['count'] if in_a else None
        sum_b   = shell_sums_b[shell]['sum']   if in_b else None
        count_b = shell_sums_b[shell]['count'] if in_b else None

        if in_a and in_b:
            delta  = sum_b - sum_a
            status = 'shared'
            total_shift += delta
        elif in_a:
            delta  = None
            status = f'only_in_{label_a}'
        else:
            delta  = None
            status = f'only_in_{label_b}'

        rows.append({
            'shell':        f'{sp1}-{sp2} shell {idx}',
            'species_pair': f'{sp1}-{sp2}',
            'shell_index':  idx,
            f'sum_{label_a}':   sum_a,
            f'count_{label_a}': count_a,
            f'sum_{label_b}':   sum_b,
            f'count_{label_b}': count_b,
            'delta':            delta,
            'status':           status,
        })

    return pd.DataFrame(rows), total_shift


# ---------------------------------------------------------------------------
# Atom matching across structures for stiffness-shift comparison
# ---------------------------------------------------------------------------

def match_atoms_across_structures(sc_a, sc_b, species, tolerance=0.05):
    """
    For each atom of `species` in sc_a, find the closest atom of the
    same species in sc_b by fractional coordinate distance (PBC-aware).
    Each atom in sc_b can only be matched once (greedy nearest-neighbour).

    Parameters
    ----------
    sc_a, sc_b  : Supercell instances
    species     : str, chemical symbol to match (e.g. 'V', 'O')
    tolerance   : float, maximum fractional-coordinate distance to accept
                  as a valid match (default 0.05, i.e. 5% of lattice vector)

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

    matches   = {}
    unmatched = []
    used_b    = set()

    for idx_a, pos_a in atoms_a:
        best_dist  = float('inf')
        best_idx_b = None

        for idx_b, pos_b in atoms_b:
            if idx_b in used_b:
                continue
            diff = np.asarray(pos_b) - np.asarray(pos_a)
            diff -= np.floor(diff + 0.5)   # minimum image
            dist = float(np.linalg.norm(diff))
            if dist < best_dist:
                best_dist  = dist
                best_idx_b = idx_b

        if best_dist <= tolerance:
            matches[idx_a] = best_idx_b
            used_b.add(best_idx_b)
        else:
            unmatched.append(idx_a)

    return matches, unmatched


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
        mean_pfc_a, mean_pfc_b, delta_pfc
    unmatched_a   : list of dicts from results_a with no counterpart in B
    """
    # Build a fast lookup for results_b: (idx1, idx2) -> result dict
    lookup_b = {}
    for r in results_b:
        lookup_b[(r['atom1_idx'], r['atom2_idx'])] = r

    matched_pairs = []
    unmatched_a   = []

    for r in results_a:
        i, j = r['atom1_idx'], r['atom2_idx']

        # Skip if either atom has no match (e.g. Li-containing pairs)
        if i not in atom_matches or j not in atom_matches:
            continue

        i_b = atom_matches[i]
        j_b = atom_matches[j]

        counterpart = lookup_b.get((i_b, j_b))
        if counterpart is None:
            unmatched_a.append(r)
            continue

        dist_a = r.get('atom_distance', r.get('distance', 0.0))
        dist_b = counterpart.get('atom_distance', counterpart.get('distance', 0.0))
        pfc_a  = r['mean_pfc']
        pfc_b  = counterpart['mean_pfc']

        matched_pairs.append({
            'atom1_idx_a': i,
            'atom2_idx_a': j,
            'atom1_idx_b': i_b,
            'atom2_idx_b': j_b,
            'species1':    r['species1'],
            'species2':    r['species2'],
            'distance_a':  dist_a,
            'distance_b':  dist_b,
            'mean_pfc_a':  pfc_a,
            'mean_pfc_b':  pfc_b,
            'delta_pfc':   pfc_b - pfc_a,
        })

    return matched_pairs, unmatched_a


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
        dist_key = lambda r: r.get('atom_distance', r.get('distance', 0.0))
        return sorted(results, key=dist_key)

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
            'distance_a': ra.get('atom_distance', ra.get('distance', 0.0)),
            'distance_b': rb.get('atom_distance', rb.get('distance', 0.0)),
            'mean_pfc_a': pfc_a,
            'mean_pfc_b': pfc_b,
            'delta_pfc':  pfc_b - pfc_a,
        })

    df = pd.DataFrame(rows)
    return df, float(df['delta_pfc'].sum()), n
