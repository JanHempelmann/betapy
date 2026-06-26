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


def _project_fc_lt(fc_matrix, direction):
    """
    Decompose a 3×3 FC matrix into longitudinal and transverse components.

    Returns
    -------
    phi_l : float
        Longitudinal (bond-stretching): êᵀΦê.
        Negative for a bonding/restoring pair (phonopy off-diagonal convention).
    phi_t : float
        Average transverse (bending): (Tr(Φ) − phi_l) / 2.
        Both quantities are invariant under Φ ↔ Φᵀ, so no symmetrization needed.
    """
    fc = np.asarray(fc_matrix, dtype=float)
    e  = np.asarray(direction,  dtype=float)
    phi_l = float(e @ fc @ e)
    phi_t = float((np.trace(fc) - phi_l) / 2.0)
    return phi_l, phi_t


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
            phi_l, phi_t      = _project_fc_lt(fc_mat, direction)
            results.append({
                'atom1_idx': i,
                'atom2_idx': j,
                'species1':  supercell.species(i),
                'species2':  supercell.species(j),
                'distance':  dist,
                'mean_pfc':  mean_pfc,
                'rms_pfc':   rms_pfc,
                'phi_l':     phi_l,
                'phi_t':     phi_t,
                'direction': direction.tolist(),
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
        phi_l_mean, phi_l_std, phi_l_min, phi_l_max,
        phi_t_mean, phi_t_std,
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
        pfcs   = np.array([rec['mean_pfc'] for rec in records])
        dists  = np.array([rec['distance']  for rec in records])
        phi_ls = np.array([rec.get('phi_l', np.nan) for rec in records])
        phi_ts = np.array([rec.get('phi_t', np.nan) for rec in records])
        has_lt = not np.all(np.isnan(phi_ls))
        shells.append({
            'species1':      sp1,
            'species2':      sp2,
            'distance_mean': float(np.mean(dists)),
            'distance_std':  float(np.std(dists)),
            'pfc_mean':      float(np.mean(pfcs)),
            'pfc_std':       float(np.std(pfcs)),
            'pfc_min':       float(np.min(pfcs)),
            'pfc_max':       float(np.max(pfcs)),
            'phi_l_mean':    float(np.nanmean(phi_ls)) if has_lt else float('nan'),
            'phi_l_std':     float(np.nanstd(phi_ls))  if has_lt else float('nan'),
            'phi_l_min':     float(np.nanmin(phi_ls))  if has_lt else float('nan'),
            'phi_l_max':     float(np.nanmax(phi_ls))  if has_lt else float('nan'),
            'phi_t_mean':    float(np.nanmean(phi_ts)) if has_lt else float('nan'),
            'phi_t_std':     float(np.nanstd(phi_ts))  if has_lt else float('nan'),
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



def match_fc_pairs_direct(results_a, results_b, sc_a, sc_b,
                          refsite_a, refsite_b, tol=1.5, directional=True):
    """
    Match off-site pFC pairs by geometric fingerprint.

    Two fingerprint modes:

    directional=True  (default, for inter-structure comparison)
        5-component: (atom1_disp_x, atom1_disp_y, atom1_disp_z,
                      atom2_ref_dist, bond_frac_norm) × L_avg.
        The signed 3D fractional displacement of atom1 from the refsite encodes
        direction so bonds at the same distance but different directions get
        distinct fingerprints.  Works when A and B share the same orientation
        (e.g. two versions of the same structure).

    directional=False  (for intra-structure / same-supercell comparison)
        3-component Cartesian magnitude fingerprint:
        (d1_cart, d2_cart, bond_cart) in Å.
        All three are rotationally invariant, so the fingerprint matches
        correctly even when the two refsites are related by a symmetry
        operation that changes displacement directions (rotations, reflections,
        glide planes — common for different Wyckoff sites in the same cell).

    Parameters
    ----------
    results_a, results_b : list of dicts from find_refsite_pairs()
    sc_a, sc_b           : Supercell instances
    refsite_a, refsite_b : array-like (3,), fractional refsite coords
    tol                  : float, max L2 norm of fingerprint difference in Å
    directional          : bool, use signed directional (True) or Cartesian
                           magnitude (False) fingerprint

    Returns
    -------
    matched_pairs, unmatched_a, unmatched_b
    """
    if not results_a or not results_b:
        return [], list(results_a), list(results_b)

    refsite_a = np.asarray(refsite_a)
    refsite_b = np.asarray(refsite_b)

    # Scale factor for fractional fingerprint (directional mode only).
    L_avg = 0.5 * (abs(np.linalg.det(sc_a.lattice)) ** (1.0 / 3.0)
                   + abs(np.linalg.det(sc_b.lattice)) ** (1.0 / 3.0))

    def _frac_disp(sc, atom_idx, ref):
        pos = sc.positions[atom_idx - 1]
        return sc.frac_diff(pos, ref)

    def _frac_norm(sc, atom_idx, ref):
        return float(np.linalg.norm(_frac_disp(sc, atom_idx, ref)))

    def _frac_norm_pair(sc, idx1, idx2):
        pos1 = sc.positions[idx1 - 1]
        pos2 = sc.positions[idx2 - 1]
        return float(np.linalg.norm(sc.frac_diff(pos1, pos2)))

    def _cart_dist(sc, atom_idx, ref_frac):
        """Cartesian distance from refsite to atom (Å)."""
        frac_d = sc.frac_diff(sc.positions[atom_idx - 1], ref_frac)
        return float(np.linalg.norm(frac_d @ sc.lattice))

    def _cart_bond(sc, idx1, idx2):
        """Cartesian bond length (Å)."""
        frac_d = sc.frac_diff(sc.positions[idx1 - 1], sc.positions[idx2 - 1])
        return float(np.linalg.norm(frac_d @ sc.lattice))

    # Group pairs by ordered species pair, pre-computing atom2 ref distances
    by_sp_a = {}
    for i, r in enumerate(results_a):
        d2 = (_frac_norm(sc_a, r['atom2_idx'], refsite_a) if directional
              else _cart_dist(sc_a, r['atom2_idx'], refsite_a))
        by_sp_a.setdefault((r['species1'], r['species2']), []).append((i, r, d2))
    by_sp_b = {}
    for j, r in enumerate(results_b):
        d2 = (_frac_norm(sc_b, r['atom2_idx'], refsite_b) if directional
              else _cart_dist(sc_b, r['atom2_idx'], refsite_b))
        by_sp_b.setdefault((r['species1'], r['species2']), []).append((j, r, d2))

    a_to_b = {}
    for key in set(by_sp_a) & set(by_sp_b):
        ag = by_sp_a[key]
        bg = by_sp_b[key]

        if directional:
            # 5-component signed fractional fingerprint × L_avg
            fp_a = np.array([[*(_frac_disp(sc_a, r['atom1_idx'], refsite_a) * L_avg),
                              d2 * L_avg,
                              _frac_norm_pair(sc_a, r['atom1_idx'], r['atom2_idx']) * L_avg]
                             for _, r, d2 in ag])
            fp_b = np.array([[*(_frac_disp(sc_b, r['atom1_idx'], refsite_b) * L_avg),
                              d2 * L_avg,
                              _frac_norm_pair(sc_b, r['atom1_idx'], r['atom2_idx']) * L_avg]
                             for _, r, d2 in bg])
        else:
            # 3-component Cartesian magnitude fingerprint (Å) — rotation-invariant
            fp_a = np.array([[_cart_dist(sc_a, r['atom1_idx'], refsite_a),
                              d2,
                              _cart_bond(sc_a, r['atom1_idx'], r['atom2_idx'])]
                             for _, r, d2 in ag])
            fp_b = np.array([[_cart_dist(sc_b, r['atom1_idx'], refsite_b),
                              d2,
                              _cart_bond(sc_b, r['atom1_idx'], r['atom2_idx'])]
                             for _, r, d2 in bg])

        diff = fp_b[None] - fp_a[:, None]
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


def structural_disturbance(matched_pairs):
    """
    Compute structural-disturbance metrics from matched pairs.

    Unlike the signed stiffness shift, these metrics use absolute values so
    that stiffening and softening contributions do not cancel — appropriate
    when the goal is to quantify the total bond rearrangement per cycle.

    Parameters
    ----------
    matched_pairs : list of dicts from match_fc_pairs_direct()

    Returns
    -------
    dict with keys:
        n_pairs     : int
        total_abs   : float, Σ |ΔpFC|
        mean_abs    : float, Σ |ΔpFC| / n_pairs
        min_delta   : float, most negative ΔpFC (most softened bond)
        min_species : str, species pair of that bond (e.g. 'V–O')
    """
    if not matched_pairs:
        return {'n_pairs': 0, 'total_abs': 0.0, 'mean_abs': 0.0,
                'min_delta': 0.0, 'min_species': ''}
    deltas    = [p['delta_pfc'] for p in matched_pairs]
    abs_deltas = [abs(d) for d in deltas]
    min_idx   = int(np.argmin(deltas))
    total_abs = float(sum(abs_deltas))
    n         = len(matched_pairs)
    return {
        'n_pairs':    n,
        'total_abs':  total_abs,
        'mean_abs':   total_abs / n,
        'min_delta':  float(deltas[min_idx]),
        'min_species': (f"{matched_pairs[min_idx]['species1']}"
                        f"–{matched_pairs[min_idx]['species2']}"),
    }


def compare_refsite_projections(results_a, results_b):
    """
    Compare two refsite projections of the same supercell by direct index matching.

    Because both analyses share the same SPOSCAR, pair identity is determined by
    (atom1_idx, atom2_idx) — no fingerprint matching required.

    Parameters
    ----------
    results_a, results_b : lists of dicts from find_refsite_pairs()

    Returns
    -------
    matched     : list of dicts with keys atom1_idx, atom2_idx, species1, species2,
                  distance, mean_pfc_a, mean_pfc_b, delta_pfc (= pfc_b − pfc_a)
    unmatched_a : list of dicts — pairs in A absent from B
    unmatched_b : list of dicts — pairs in B absent from A
    total_delta : float, Σ delta_pfc over all matched pairs
    """
    by_pair_b    = {(r['atom1_idx'], r['atom2_idx']): r for r in results_b}
    matched      = []
    unmatched_a  = []
    matched_b_keys = set()
    for ra in results_a:
        key = (ra['atom1_idx'], ra['atom2_idx'])
        if key in by_pair_b:
            rb = by_pair_b[key]
            matched.append({
                'atom1_idx':  ra['atom1_idx'],
                'atom2_idx':  ra['atom2_idx'],
                'species1':   ra['species1'],
                'species2':   ra['species2'],
                'distance':   ra['distance'],
                'mean_pfc_a': ra['mean_pfc'],
                'mean_pfc_b': rb['mean_pfc'],
                'delta_pfc':  rb['mean_pfc'] - ra['mean_pfc'],
            })
            matched_b_keys.add(key)
        else:
            unmatched_a.append(ra)
    unmatched_b = [r for r in results_b
                   if (r['atom1_idx'], r['atom2_idx']) not in matched_b_keys]
    total_delta = sum(p['delta_pfc'] for p in matched)
    return matched, unmatched_a, unmatched_b, total_delta


def sum_intercalant_pfcs(results, intercalant_species):
    """
    Sum pFCs for framework → intercalant bonds from refsite-projection results.

    Filters to pairs where atom2 belongs to an intercalant species (atom1 is
    the framework atom whose FC is projected toward the reference site).
    Pairs where atom1 is the intercalant (sitting at the refsite) are excluded
    because their projection direction is degenerate (~zero vector).

    Parameters
    ----------
    results             : list of dicts from find_refsite_pairs()
    intercalant_species : set of str

    Returns
    -------
    total   : float, sum of mean_pfc for intercalant pairs
    records : list of matching result dicts
    """
    records = [r for r in results if r['species2'] in intercalant_species]
    total   = sum(r['mean_pfc'] for r in records)
    return total, records


