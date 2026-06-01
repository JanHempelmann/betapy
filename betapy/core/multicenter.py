"""
Multicenter bonding detection for betapy.

Detects anomalously large projected force constant (pFC) values that may
indicate electron-rich multicenter bonding (e.g. Te-Ge-Te in GeTe, I3-,
XeF2-type systems), traces the underlying atomic chain in the supercell, and
generates ``cobiBetween`` directives for LOBSTER COBI(N) analysis.

Typical workflow
----------------
    from betapy.core.multicenter import suggest_cobi_directives

    result = suggest_cobi_directives(
        bulk_results, supercell,
        poscar_lobster_path=lobster_dir / 'POSCAR',
        n_sigma=2.0, max_order=5,
    )
    for line in result['directives']:
        print(line)

Detection strategy
------------------
Anomaly detection works directly on individual pair records from
``compute_bulk_pfcs()``, bypassing shell averaging.  A robust Theil-Sen
regression is fitted to Phi_p^{-1/3} vs r per species pair.  Theil-Sen is
insensitive to outliers by design, so anomalous (multicenter) pairs do not
contaminate the baseline.  Individual pairs whose pFC residual exceeds
*n_sigma* robust standard deviations are flagged.

Reliability window
------------------
Force-constant values are only reliable up to half the shortest supercell
dimension (L/2).  Only pairs within this window are considered for detection,
and chain extension is automatically stopped before the limit.

POSCAR requirement
------------------
``poscar_lobster_path`` must point to the POSCAR used for the LOBSTER
calculation (same conventional/primitive cell from which the phonon supercell
was derived).  POSCAR and POSCAR.lobster.vasp from the lobster directory are
both acceptable.
"""

import numpy as np
from pathlib import Path
from scipy.stats import theilslopes, median_abs_deviation

from betapy.core.lobster import _parse_poscar_lobster


# ---------------------------------------------------------------------------
# Anomaly detection — individual pairs, robust regression
# ---------------------------------------------------------------------------

def detect_anomalous_pairs(bulk_results, n_sigma=2.0, min_pairs=4):
    """
    Flag individual pFC pairs with anomalously large values relative to distance.

    Groups pairs by species (order-independent) and fits a Theil-Sen regression
    to Phi_p^{-1/3} vs r — the Badger-type relationship.  Theil-Sen is
    insensitive to outliers, so anomalous (multicenter) pairs cannot bias the
    baseline.  Pairs where the actual Phi_p^{-1/3} lies more than *n_sigma*
    robust standard deviations below the fit (i.e. pFC is too large for its
    distance) are flagged.

    For species pairs with fewer than *min_pairs* valid records a monotonicity
    fallback is used: any pair whose pFC exceeds that of the nearest
    shorter-distance pair (when sorted by distance) is flagged.

    Parameters
    ----------
    bulk_results : list of dicts from compute_bulk_pfcs()
        Each dict must contain 'species1', 'species2', 'distance', 'mean_pfc',
        'atom1_idx', 'atom2_idx', and 'direction'.
    n_sigma      : float, detection threshold in robust std deviations. Default 2.0.
    min_pairs    : int, minimum valid pairs required for regression. Default 4.

    Returns
    -------
    list of dicts — each is the original pair record augmented with:
        'method'   : 'regression' or 'monotone'
        'residual' : float — regression: signed residual on Phi_p^{-1/3}
                     (negative = pFC larger than predicted); monotone: raw pFC
                     excess over the nearest shorter-distance pair
        'n_sigma'  : float, significance (nan for monotone method)
    """
    from collections import defaultdict

    # Deduplicate: full FORCE_CONSTANTS files list every (i,j) bond twice —
    # once as atom1=i and once as atom1=j.  Removing duplicates halves the
    # dataset entering the O(n²) Theil-Sen regression (4× fewer slope pairs).
    seen_bonds: set = set()
    deduped: list = []
    for r in bulk_results:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        if key not in seen_bonds:
            seen_bonds.add(key)
            deduped.append(r)

    by_pair = defaultdict(list)
    for r in deduped:
        key = tuple(sorted([r['species1'], r['species2']]))
        by_pair[key].append(r)

    flagged = []
    for pair_records in by_pair.values():
        pfcs  = np.array([r['mean_pfc'] for r in pair_records])
        dists = np.array([r['distance'] for r in pair_records])

        valid = pfcs > 0
        if valid.sum() < 2:
            continue

        v_pfcs    = pfcs[valid]
        v_dists   = dists[valid]
        v_records = [r for r, v in zip(pair_records, valid) if v]

        inv_cbrt = v_pfcs ** (-1.0 / 3.0)

        if valid.sum() >= min_pairs:
            slope, intercept, *_ = theilslopes(inv_cbrt, v_dists)
            predicted = slope * v_dists + intercept
            residuals = inv_cbrt - predicted        # negative => pFC too large
            std = float(median_abs_deviation(residuals) * 1.4826)
            # Floor prevents divide-by-zero and handles high-symmetry systems
            # where all normal pairs are exactly on the Badger line (MAD=0).
            # In that limit any negative residual is genuinely anomalous.
            std = max(std, 1e-6)
            for rec, res in zip(v_records, residuals):
                if res < -n_sigma * std:
                    flagged.append({**rec,
                                    'method':   'regression',
                                    'residual': float(res),
                                    'n_sigma':  float(-res / std)})
        else:
            order  = np.argsort(v_dists)
            s_pfcs = v_pfcs[order]
            s_recs = [v_records[i] for i in order]
            for i in range(1, len(s_recs)):
                if s_pfcs[i] > s_pfcs[i - 1]:
                    flagged.append({**s_recs[i],
                                    'method':   'monotone',
                                    'residual': float(s_pfcs[i] - s_pfcs[i - 1]),
                                    'n_sigma':  float('nan')})

    return flagged


# ---------------------------------------------------------------------------
# SPOSCAR → POSCAR.lobster atom mapping
# ---------------------------------------------------------------------------

def _map_sc_atom_to_poscar(sc_idx, supercell, lob_poscar, tol=0.15):
    """
    Map a 1-based SPOSCAR atom index to its POSCAR.lobster label and cell
    translation vector.

    The SPOSCAR and POSCAR must be commensurate (POSCAR is the cell from which
    the phonon supercell was derived).  Each supercell atom corresponds to one
    POSCAR atom plus an integer cell translation in POSCAR-cell units.

    Parameters
    ----------
    sc_idx    : int, 1-based atom index in SPOSCAR
    supercell : Supercell
    lob_poscar: dict from lobster._parse_poscar_lobster()
    tol       : float, Angstrom, matching tolerance. Default 0.15.

    Returns
    -------
    label : str, LOBSTER atom label, e.g. 'Ge1' or 'Te6'
    cell  : list[int], cell translation in POSCAR-cell units, e.g. [-1, 0, 0]

    Raises
    ------
    ValueError if no POSCAR atom is found within *tol*.
    """
    frac_sc = supercell.positions[sc_idx - 1]
    cart = frac_sc @ supercell.lattice

    lob_lat     = lob_poscar['lattice']
    lob_lat_inv = np.linalg.inv(lob_lat)
    f_lob = cart @ lob_lat_inv

    # Small epsilon avoids floor(n - epsilon) = n - 1 for near-integer values.
    cell      = np.floor(f_lob + 1e-9).astype(int)
    f_wrapped = f_lob - cell
    # Second wrap for residual floating-point drift
    f_wrapped = f_wrapped - np.floor(f_wrapped + 1e-9)

    fracs   = lob_poscar['positions_frac']
    species = lob_poscar['species']

    best_idx  = -1
    best_dist = tol + 1.0
    for i, fp in enumerate(fracs):
        diff = f_wrapped - fp
        diff -= np.round(diff)
        cart_dist = float(np.linalg.norm(diff @ lob_lat))
        if cart_dist < best_dist:
            best_dist = cart_dist
            best_idx  = i

    if best_idx < 0 or best_dist > tol:
        raise ValueError(
            f"SPOSCAR atom {sc_idx} (frac {frac_sc.tolist()}) did not match "
            f"any POSCAR atom within {tol:.2f} Å (best {best_dist:.3f} Å). "
            "Ensure POSCAR.lobster is commensurate with SPOSCAR."
        )

    return f"{species[best_idx]}{best_idx + 1}", cell.tolist()


# ---------------------------------------------------------------------------
# Chain finding
# ---------------------------------------------------------------------------

def _build_neighbor_lookup(bulk_results, bond_cutoff):
    """
    Build atom-to-neighbor dict from bulk_results, restricted to pairs within
    *bond_cutoff* Angstrom.

    Returns
    -------
    dict : atom_idx (1-based int) →
           list of {'idx': int, 'dist': float, 'dir': ndarray(3)}
    'dir' is the unit vector FROM that atom TO the neighbour.
    """
    neighbors: dict = {}
    for r in bulk_results:
        if r['distance'] > bond_cutoff:
            continue
        i, j    = r['atom1_idx'], r['atom2_idx']
        dir_ij  = np.array(r['direction'], dtype=float)
        neighbors.setdefault(i, []).append(
            {'idx': j, 'dist': r['distance'], 'dir':  dir_ij})
        neighbors.setdefault(j, []).append(
            {'idx': i, 'dist': r['distance'], 'dir': -dir_ij})
    return neighbors


def _build_neighbor_lookup_from_structure(supercell, bond_cutoff, _chunk=128):
    """
    Build a complete atom-to-neighbor dict from supercell geometry.

    Unlike ``_build_neighbor_lookup``, this uses atomic positions directly, so
    every atom gets full neighbor information regardless of which atoms appear
    as atom1 in a compact FORCE_CONSTANTS file.  The minimum-image convention
    is applied in fractional coordinates before converting to Cartesian.

    Parameters
    ----------
    supercell  : Supercell
    bond_cutoff: float, Angstrom — maximum neighbour distance
    _chunk     : int, rows processed per batch (memory bound)

    Returns
    -------
    dict : atom_idx (1-based int) →
           list of {'idx': int, 'dist': float, 'dir': ndarray(3)}
    """
    n     = supercell.n_atoms
    fracs = supercell.positions   # (N, 3) fractional
    latt  = supercell.lattice     # (3, 3)
    neighbors: dict = {}

    for i_start in range(0, n, _chunk):
        i_end = min(i_start + _chunk, n)
        # diff[ci, j] = frac(j) - frac(i_start+ci), shape (chunk, N, 3)
        raw   = fracs[np.newaxis, :, :] - fracs[i_start:i_end, np.newaxis, :]
        diff  = raw - np.floor(raw + 0.5)      # minimum image
        cart  = diff @ latt                    # Cartesian, (chunk, N, 3)
        dists = np.linalg.norm(cart, axis=-1)  # (chunk, N)

        ci_arr, j_arr = np.where((dists > 1e-6) & (dists <= bond_cutoff))
        for ci, j in zip(ci_arr.tolist(), j_arr.tolist()):
            i = i_start + ci
            if i >= j:
                continue  # each pair processed once
            d        = float(dists[ci, j])
            dir_ij   = cart[ci, j] / d
            idx_i, idx_j = i + 1, j + 1  # 1-based
            # Bond crosses a periodic boundary when minimum-image wrapping
            # changes the fractional difference vector.
            boundary = bool(np.any(np.abs(raw[ci, j] - diff[ci, j]) > 1e-6))
            neighbors.setdefault(idx_i, []).append(
                {'idx': idx_j, 'dist': d, 'dir':  dir_ij, 'boundary': boundary})
            neighbors.setdefault(idx_j, []).append(
                {'idx': idx_i, 'dist': d, 'dir': -dir_ij, 'boundary': boundary})

    return neighbors


def _grow_chain(start_idx, init_direction, neighbors,
                min_cos, max_order, reliability_limit):
    """
    Greedily extend a chain from *start_idx* in *init_direction*.

    At each step the neighbour with the highest directional cosine (above
    *min_cos*) is chosen.  Growth stops when no suitable neighbour exists or
    the *cumulative* bond length along the chain would exceed
    *reliability_limit*.

    Cumulative length is used rather than end-to-end distance so that the
    check is not fooled by the minimum-image convention wrapping a long chain
    back to a short periodic distance.  Neighbours flagged as boundary-crossing
    (where minimum-image wrapping changes the fractional vector) are skipped so
    that chains never extend across a supercell boundary.

    Parameters
    ----------
    start_idx         : int, 1-based SPOSCAR atom index
    init_direction    : array-like (3,), unit vector for initial direction
    neighbors         : dict from _build_neighbor_lookup_from_structure()
    min_cos           : float, cosine threshold (cos(180° - min_angle))
    max_order         : int, maximum number of atoms
    reliability_limit : float, Angstrom, max allowed cumulative chain length

    Returns
    -------
    list of 1-based SPOSCAR atom indices (length >= 1)
    """
    chain        = [start_idx]
    current_dir  = np.asarray(init_direction, dtype=float)
    chain_length = 0.0   # cumulative sum of step distances

    while len(chain) < max_order:
        last_idx = chain[-1]
        best_nb  = None
        best_cos = min_cos  # strict lower bound

        for nb in neighbors.get(last_idx, []):
            nb_idx = nb['idx']
            if nb_idx in chain:
                continue
            if nb.get('boundary', False):
                continue
            cos_angle = float(np.dot(current_dir, nb['dir']))
            if cos_angle <= best_cos:
                continue
            if chain_length + nb['dist'] > reliability_limit:
                continue
            best_cos = cos_angle
            best_nb  = nb

        if best_nb is None:
            break

        chain.append(best_nb['idx'])
        chain_length += best_nb['dist']
        current_dir   = best_nb['dir']   # unit vector already normalised

    return chain


def find_chains(flagged_records, supercell, bulk_results=None,
                min_angle_deg=150.0, max_order=5, bond_cutoff=4.0):
    """
    Trace multicenter bonding chains starting from anomalous pFC pair records.

    For each flagged pair (atom1_idx, atom2_idx) the chain is grown from
    atom1_idx in the atom1→atom2 direction, picking up bridging atoms at each
    step.  Sub-chains of all orders from 3 up to the full chain length are
    recorded; together they represent the hierarchy of multicenter interactions
    within the same bonding chain.

    The reliability window (half the shortest supercell dimension) is enforced:
    no chain extends beyond the reliable range of the force constants.

    Parameters
    ----------
    flagged_records : list of pair-record dicts
        Individual pair records as returned by detect_anomalous_pairs().
    supercell       : Supercell
    bulk_results    : list of dicts from compute_bulk_pfcs() [first return value]
    min_angle_deg   : float, minimum bond angle in chain (degrees). Default 150.
    max_order       : int, maximum number of atoms per chain. Default 5.
    bond_cutoff     : float, Angstrom, max step distance for extension. Default 4.0.

    Returns
    -------
    list of chain dicts, each with:
        'trigger_pair'   : the input flagged pair record
        'full_chain'     : list of 1-based SPOSCAR indices
        'species_chain'  : list of species strings
        'total_distance' : float, end-to-end distance in Angstrom
        'sub_chains'     : list of dicts
            All consecutive sub-sequences of length 3..len(full_chain).
            Keys: 'order' (int), 'indices' (list of int), 'directive' (None
            until filled by format_cobi_directive / suggest_cobi_directives).
    """
    reliability_limit = min(np.linalg.norm(v) for v in supercell.lattice) / 2.0
    min_cos   = np.cos(np.radians(180.0 - min_angle_deg))
    neighbors = _build_neighbor_lookup_from_structure(supercell, bond_cutoff)

    results = []
    for rec in flagged_records:
        start     = rec['atom1_idx']
        direction = np.array(rec['direction'], dtype=float)

        chain = _grow_chain(start, direction, neighbors,
                            min_cos, max_order, reliability_limit)

        if len(chain) < 3:
            continue

        pos0  = supercell.positions[chain[0]  - 1]
        posN  = supercell.positions[chain[-1] - 1]
        total_dist   = float(np.linalg.norm(supercell.cart_diff(pos0, posN)))
        species_chain = [supercell.species(idx) for idx in chain]

        sub_chains = []
        for length in range(3, len(chain) + 1):
            for start_pos in range(len(chain) - length + 1):
                sub_chains.append({
                    'order':     length,
                    'indices':   chain[start_pos: start_pos + length],
                    'directive': None,
                })

        results.append({
            'trigger_pair':   rec,
            'full_chain':     chain,
            'species_chain':  species_chain,
            'total_distance': total_dist,
            'sub_chains':     sub_chains,
        })

    return results


# ---------------------------------------------------------------------------
# lobsterin directive formatting
# ---------------------------------------------------------------------------

def format_cobi_directive(chain_sc_indices, supercell, lob_poscar):
    """
    Format a ``cobiBetween`` lobsterin line for the given chain of SPOSCAR atoms.

    Cell translations are normalised so the first atom in the chain is at
    [0 0 0].  Atoms with a zero relative cell translation have no ``cell`` tag.

    Parameters
    ----------
    chain_sc_indices : list of int, 1-based SPOSCAR atom indices (>= 3 atoms)
    supercell        : Supercell
    lob_poscar       : dict from lobster._parse_poscar_lobster()

    Returns
    -------
    str, e.g. ``'cobiBetween Te5 Ge1 cell -1 0 0 Te8'``

    Raises
    ------
    ValueError if any atom cannot be matched in POSCAR.
    """
    labels_cells = [_map_sc_atom_to_poscar(idx, supercell, lob_poscar)
                    for idx in chain_sc_indices]

    base_cell = np.array(labels_cells[0][1], dtype=int)

    # Supercell repetition matrix in POSCAR-cell units: L_sc = T @ L_lob.
    # Used to apply minimum-image convention to relative cell vectors so that
    # e.g. [0, 3, 0] in a 4×4×4 supercell becomes [0, -1, 0] — the nearest
    # periodic image in the POSCAR frame.
    T     = np.round(supercell.lattice @ np.linalg.inv(lob_poscar['lattice']))
    T_inv = np.linalg.inv(T)

    parts = ['cobiBetween']
    for label, raw_cell in labels_cells:
        rel      = np.array(raw_cell, dtype=float) - base_cell
        rel_frac = T_inv @ rel
        rel_mi   = np.round(T @ (rel_frac - np.floor(rel_frac + 0.5))).astype(int)

        parts.append(label)
        if np.any(rel_mi != 0):
            parts.extend(['cell', str(int(rel_mi[0])), str(int(rel_mi[1])), str(int(rel_mi[2]))])

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def suggest_cobi_directives(
        bulk_results, supercell, poscar_lobster_path,
        n_sigma=2.0, min_pairs=4,
        min_angle_deg=150.0, max_order=5, bond_cutoff=4.0):
    """
    Full pipeline: detect anomalous pFCs → trace chains → format directives.

    Parameters
    ----------
    bulk_results         : first element of projection.compute_bulk_pfcs() return value
    supercell            : Supercell
    poscar_lobster_path  : path-like, POSCAR used for the LOBSTER calculation
    n_sigma              : float, anomaly detection threshold (sigma). Default 2.0.
    min_pairs            : int, min pairs for regression detection. Default 4.
    min_angle_deg        : float, minimum bond angle for chain extension. Default 150.
    max_order            : int, maximum atoms per chain. Default 5.
    bond_cutoff          : float, Å, max step distance for chain extension. Default 4.0.

    Returns
    -------
    dict with:
        'flagged_pairs' : list of detection entries from detect_anomalous_pairs()
        'chains'        : list of chain dicts; sub_chains[*]['directive'] is filled
        'directives'    : list[str], unique cobiBetween lines ready for lobsterin
    """
    lob_poscar = _parse_poscar_lobster(poscar_lobster_path)

    reliability_limit = min(np.linalg.norm(v) for v in supercell.lattice) / 2.0
    reliable_pairs = [r for r in bulk_results if r['distance'] <= reliability_limit]

    flagged = detect_anomalous_pairs(
        reliable_pairs, n_sigma=n_sigma, min_pairs=min_pairs)

    chains = find_chains(
        flagged, supercell, bulk_results,
        min_angle_deg=min_angle_deg, max_order=max_order, bond_cutoff=bond_cutoff,
    )

    seen_keys: set = set()
    unique_directives: list = []
    for chain in chains:
        for sub in chain['sub_chains']:
            sp_key = tuple(supercell.species(idx) for idx in sub['indices'])
            # Forward and reverse describe the same LOBSTER COBI interaction;
            # normalise so the lexicographically smaller direction is canonical.
            canon_key = min(sp_key, sp_key[::-1])
            try:
                directive = format_cobi_directive(
                    sub['indices'], supercell, lob_poscar)
                sub['directive'] = directive
                if canon_key not in seen_keys:
                    seen_keys.add(canon_key)
                    unique_directives.append(directive)
            except ValueError as exc:
                sub['directive'] = f'# ERROR: {exc}'

    return {
        'flagged_pairs': flagged,
        'chains':        chains,
        'directives':    unique_directives,
    }


# ---------------------------------------------------------------------------
# lobsterin I/O utility
# ---------------------------------------------------------------------------

def append_cobi_directives(lobsterin_path, directives):
    """
    Append cobiBetween directives to an existing lobsterin file.

    Directives already present (exact string match) are skipped.
    A blank line separator is inserted before any new directives.

    Parameters
    ----------
    lobsterin_path : path-like
    directives     : list[str], output of suggest_cobi_directives()['directives']

    Returns
    -------
    int : number of directives actually written
    """
    path = Path(lobsterin_path)
    existing = path.read_text()
    to_add = [d for d in directives if d not in existing]
    if not to_add:
        return 0
    with open(path, 'a') as f:
        f.write('\n')
        for d in to_add:
            f.write(d + '\n')
    return len(to_add)
