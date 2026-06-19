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
``compute_bulk_pfcs()``, bypassing shell averaging.  A Theil-Sen regression
is fitted to Phi_p^{-1/3} vs r per species pair — the Badger-type
relationship.  Theil-Sen (median pairwise slope) is inherently insensitive to
outliers, so anomalous (multicenter) pairs cannot contaminate the baseline.
Because Phi_p^{-1/3} is a decreasing function of Phi_p, multicenter bonds
(high pFC) sit *below* the Badger line in Phi_p^{-1/3} space.  Individual
pairs whose residual exceeds *n_sigma* robust standard deviations below the
fit are flagged.

The detection catches both direct short-bond anomalies and indirect signals:
in metavalent systems such as Sb₂Te₃ the anomalously stiff end-to-end force
constants across a quintuple layer (Sb–Sb at ~6.3 Å, Te–Te at ~6.1 Å) are
flagged, and the chain-extension step then identifies the intermediate bridge
atom (Te for Sb–Te–Sb, Sb for Te–Sb–Te).

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
# Quantile regression (LP formulation) — kept for GUI visualisation only
# ---------------------------------------------------------------------------

def _quantile_regression_line(x, y, q):
    """
    Fit linear quantile regression y = a*x + b at quantile *q* via LP.

    Not used in detection (which uses Theil-Sen); retained for GUI overlays
    where a tunable envelope quantile is useful for visualisation.

    Returns
    -------
    slope, intercept : float
    """
    from scipy.optimize import linprog
    n = len(x)
    c = np.concatenate([[0.0, 0.0], np.full(n, q), np.full(n, 1.0 - q)])
    A_eq = np.zeros((n, 2 + 2 * n))
    A_eq[:, 0] = x
    A_eq[:, 1] = 1.0
    A_eq[np.arange(n), 2 + np.arange(n)] = 1.0
    A_eq[np.arange(n), 2 + n + np.arange(n)] = -1.0
    bounds = [(None, None), (None, None)] + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds)
    return float(res.x[0]), float(res.x[1])


# ---------------------------------------------------------------------------
# Symmetry dataset (spglib wrapper)
# ---------------------------------------------------------------------------

def _spglib_dataset(supercell):
    """
    Return the spglib symmetry dataset for *supercell*, or None if unavailable.

    The key field used downstream is ``dataset.equivalent_atoms``: a
    0-based integer array of length N where atoms with the same value are
    related by a crystal symmetry operation (Gleichergestalt equivalence).

    Parameters
    ----------
    supercell : Supercell

    Returns
    -------
    dict (spglib dataset) or None if spglib is not installed or analysis fails.
    """
    try:
        import spglib
    except ImportError:
        return None
    # Encode species as consecutive integers — spglib only needs consistent
    # type labels, not true atomic numbers.
    type_map  = {s: i for i, s in enumerate(supercell.chem_symbols)}
    atom_types = [type_map[supercell.species(i + 1)]
                  for i in range(supercell.n_atoms)]
    cell = (supercell.lattice, supercell.positions, atom_types)
    return spglib.get_symmetry_dataset(cell)


# ---------------------------------------------------------------------------
# Shell splitting helper
# ---------------------------------------------------------------------------

def _split_into_shells(records, rel_gap=0.50):
    """
    Partition bond records into distance shells separated by significant gaps.

    Bonds at qualitatively different distances — e.g. covalent 1st-shell C-C
    at 1.54 Å vs non-bonded 2nd-shell C-C at 2.52 Å (63 % gap) — do not
    follow the same Badger trend and must not share a baseline fit.  This
    function identifies such cross-regime boundaries and splits the record list.

    A boundary is placed where the relative gap between consecutive distances
    exceeds *rel_gap* (default 50 %).  The 50 % threshold was chosen to:

    * Split the diamond 1st→2nd-shell C-C boundary (63 % gap) so that the
      covalent first-shell bonds are not falsely compared against non-bonded
      second-shell interactions.
    * Leave GeTe's short/long Ge-Te bonds (12 % gap) intact so that the
      Badger slope can still identify multicenter-enhanced short bonds.
    * Leave Sb₂Te₃'s 2nd-shell-to-long-range Sb-Te boundary (~36 % gap)
      intact so that long-range pairs continue to anchor the baseline and
      make anomalous intra-layer bonds detectable.

    Parameters
    ----------
    records  : list of bond-record dicts (each has 'distance' key)
    rel_gap  : float, minimum relative gap that triggers a split. Default 0.20.

    Returns
    -------
    list of lists of dicts — one sub-list per shell.
    """
    if not records:
        return [records]
    dists = np.array([r['distance'] for r in records])
    order = np.argsort(dists)
    sorted_d = dists[order]

    split_at = []
    for k in range(len(sorted_d) - 1):
        if sorted_d[k] > 0 and (sorted_d[k + 1] - sorted_d[k]) / sorted_d[k] > rel_gap:
            split_at.append(k + 1)

    if not split_at:
        return [records]

    shells = []
    prev = 0
    for sp in split_at:
        shells.append([records[i] for i in order[prev:sp]])
        prev = sp
    shells.append([records[i] for i in order[prev:]])
    return [s for s in shells if s]


# ---------------------------------------------------------------------------
# Anomaly detection — individual pairs, robust regression
# ---------------------------------------------------------------------------

def detect_anomalous_pairs(bulk_results, n_sigma=1.5, min_pairs=4,
                           value_key='mean_pfc', min_rel_residual=0.08,
                           max_detect_dist=None, max_nn_ratio=None):
    """
    Flag individual pFC pairs with anomalously large values relative to distance.

    Groups pairs by species (order-independent) and fits a Theil-Sen regression
    to FC^{-1/3} vs r — the Badger-type relationship.  Theil-Sen is the median
    of all pairwise slopes, making it inherently insensitive to outliers:
    anomalous (multicenter) pairs cannot bias the baseline even when they are a
    substantial minority.  Individual pairs whose FC residual exceeds *n_sigma*
    robust standard deviations below the fit (i.e. FC is too large for its
    distance) are flagged.

    Using *value_key='phi_iso'* (default) is strongly preferred over 'mean_pfc'.
    Φ_iso = (|φ_l| + 2|φ_t|) / 3 removes the gleichergestalt orientation
    dependence that inflates the residual scatter when using conventional Φ_p,
    giving a tighter baseline and more discriminating n_sigma threshold.

    For groups with fewer than *min_pairs* valid records a monotonicity
    fallback is used: any pair whose FC exceeds that of the nearest
    shorter-distance pair (when sorted by distance) is flagged.

    Parameters
    ----------
    bulk_results : list of dicts — records must already contain *value_key*.
        Call compute_badger_quantities() first to add 'phi_iso' and 'xi'.
        Each dict must also contain 'species1', 'species2', 'distance',
        'atom1_idx', 'atom2_idx', and 'direction'.
    n_sigma      : float, detection threshold in robust std deviations. Default 2.0.
    min_pairs    : int, minimum valid pairs required for regression. Default 4.
    value_key        : str, force-constant field to use for the Badger baseline.
                       Default 'phi_iso' (rotationally invariant).
                       Pass 'mean_pfc' to use the conventional projected FC.
    min_rel_residual : float, minimum residual relative to the predicted
                       FC^{-1/3} value required to flag a pair.  Prevents
                       flagging pairs whose absolute FC deviation is physically
                       negligible even if statistically significant — e.g.
                       non-bonded O-O pairs in Quartz whose tiny phi_iso
                       scatter produces a large n_sigma on a near-zero MAD.
                       Default 0.08 (8 % of the predicted FC^{-1/3} value,
                       corresponding to ~25 % excess in FC space).
    max_detect_dist : float or None, Angstrom.  If set, only pairs with
                       distance <= max_detect_dist are considered for flagging
                       (pairs beyond this are still used to anchor the Badger
                       baseline if they are in the same shell).  Typically set
                       to ~75 % of the reliability limit (L/2) so that long-range
                       pairs whose tiny FC values produce spurious statistical
                       outliers near the aliasing boundary are excluded.
                       Default None (no extra cutoff).
    max_nn_ratio    : float or None, advanced override: skip flagging pairs whose
                       distance exceeds this multiple of the species-pair NN
                       distance.  In the normal pipeline this filter is applied
                       at the CHAIN SEGMENT level inside find_chains / _grow_chain
                       (where it correctly targets non-bonded contacts rather than
                       the end-to-end span of a multi-center chain).  Applying it
                       at the pair level is rarely correct — use only when calling
                       detect_anomalous_pairs directly outside the chain pipeline.
                       Default None (disabled).

    Returns
    -------
    list of dicts — each is the original pair record augmented with:
        'method'   : 'regression' or 'monotone'
        'residual' : float — regression: signed residual on FC^{-1/3}
                     (negative = FC larger than predicted); monotone: raw FC
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

    by_pair: dict = defaultdict(list)
    for r in deduped:
        sp_key = tuple(sorted([r['species1'], r['species2']]))
        by_pair[sp_key].append(r)

    flagged = []

    for pair_records in by_pair.values():
        fcs   = np.array([r.get(value_key, float('nan')) for r in pair_records])
        dists = np.array([r['distance'] for r in pair_records])

        valid = np.isfinite(fcs) & (fcs > 0)
        if valid.sum() < 2:
            continue

        v_fcs     = fcs[valid]
        v_dists   = dists[valid]
        v_records = [r for r, v in zip(pair_records, valid) if v]

        if float(v_dists.max() - v_dists.min()) < 0.05:
            continue

        inv_cbrt = v_fcs ** (-1.0 / 3.0)

        # One representative per unique (distance, value) pair.
        # Symmetry-equivalent bonds share both distance and force constant, so
        # collapsing on the pair correctly deduplicates them without merging
        # geometrically distinct bonds that happen to sit at the same distance
        # (e.g. NaCl has two bond types at 1.5a with different force constants).
        # All pairs are still evaluated against the resulting fit for flagging.
        _keys = np.stack([np.round(v_dists, 3), np.round(inv_cbrt, 4)], axis=1)
        _, _ux = np.unique(_keys, axis=0, return_index=True)

        if len(_ux) >= min_pairs:
            slope, intercept, *_ = theilslopes(inv_cbrt[_ux], v_dists[_ux],
                                               method='joint')
            pred_uniq = slope * v_dists[_ux] + intercept
            # Residuals in log(phi_iso) space: log(phi_actual / phi_expected).
            # Positive = bond is stronger than the Badger baseline.
            # Using log space makes scatter approximately distance-invariant
            # (multiplicative noise), so one σ applies across all distances —
            # unlike additive phi^{-1/3} residuals which under-weight short-range
            # anomalies where phi^{-1/3} is small.
            log_ratio_uniq = 3.0 * np.log(
                np.maximum(pred_uniq, 1e-12) / inv_cbrt[_ux])
            std_raw = float(median_abs_deviation(log_ratio_uniq) * 1.4826)
            std     = max(std_raw, 1e-6)
            # Minimum log-ratio for physical significance: equivalent to the
            # old min_rel_residual floor in phi^{-1/3} space, prevents σ→0
            # floor from triggering on numerically tiny deviations.
            min_log_r   = 3.0 * np.log(1.0 / max(1.0 - min_rel_residual, 1e-9))
            pred_all      = slope * v_dists + intercept
            safe_pred     = np.where(pred_all > 0, pred_all, 1.0)
            log_ratio_all = np.where(
                pred_all > 0,
                3.0 * np.log(safe_pred / inv_cbrt),
                -np.inf)
            min_d_sp = float(v_dists.min())
            for rec, log_r, pred in zip(v_records, log_ratio_all, pred_all):
                if log_r > n_sigma * std and log_r >= min_log_r:
                    if max_detect_dist is not None and rec['distance'] > max_detect_dist:
                        continue
                    if max_nn_ratio is not None and rec['distance'] > max_nn_ratio * min_d_sp:
                        continue
                    flagged.append({**rec,
                                    'method':   'regression',
                                    'residual': float(log_r),
                                    'n_sigma':  float(log_r / std)})
        else:
            order  = np.argsort(v_dists)
            s_fcs  = v_fcs[order]
            s_recs = [v_records[i] for i in order]
            for i in range(1, len(s_recs)):
                if s_fcs[i] > s_fcs[i - 1]:
                    flagged.append({**s_recs[i],
                                    'method':   'monotone',
                                    'residual': float(s_fcs[i] - s_fcs[i - 1]),
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
    label : str, LOBSTER atom label, e.g. 'atom1' or 'atom6'
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

    fracs    = lob_poscar['positions_frac']

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

    sc_species  = supercell.species(sc_idx)
    lob_species = lob_poscar['species'][best_idx]
    if sc_species != lob_species:
        raise ValueError(
            f"Species mismatch: SPOSCAR atom {sc_idx} is {sc_species!r} but "
            f"the geometrically matched POSCAR atom {best_idx + 1} is "
            f"{lob_species!r}. The POSCAR and SPOSCAR appear to describe "
            "different structures — check that you are using the POSCAR from "
            "the same LOBSTER run."
        )

    return f"atom{best_idx + 1}", cell.tolist()


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
                min_cos, max_order, reliability_limit,
                atom_species=None, nn_distances=None, max_nn_ratio=None):
    """
    Greedily extend a chain from *start_idx* in *init_direction*.

    At each step the neighbour with the highest directional cosine (above
    *min_cos*) is chosen.  Growth stops when no suitable neighbour exists or
    the *cumulative* bond length along the chain would exceed
    *reliability_limit*.

    Cumulative length is used rather than end-to-end distance so that the
    check is not fooled by the minimum-image convention wrapping a long chain
    back to a short periodic distance.  At most one supercell boundary crossing
    is permitted: this allows gap-crossing chains (e.g. vdW-gap multicenter
    bonds in layered materials) whose collinear entry path crosses a cell image,
    while still preventing wrap-around chains that would require two or more
    boundary crossings in the same direction.

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
    chain                = [start_idx]
    current_dir          = np.asarray(init_direction, dtype=float)
    chain_length         = 0.0   # cumulative sum of step distances
    n_boundary_crossings = 0     # allow at most 1 to reach gap-crossing chains

    while len(chain) < max_order:
        last_idx = chain[-1]
        best_nb  = None
        best_cos = min_cos  # strict lower bound

        for nb in neighbors.get(last_idx, []):
            nb_idx = nb['idx']
            if nb_idx in chain:
                continue
            if nb.get('boundary', False) and n_boundary_crossings >= 1:
                continue
            cos_angle = float(np.dot(current_dir, nb['dir']))
            if cos_angle <= best_cos:
                continue
            if chain_length + nb['dist'] > reliability_limit:
                continue
            if (max_nn_ratio is not None and atom_species is not None
                    and nn_distances is not None):
                sp_curr = atom_species.get(last_idx)
                sp_nb   = atom_species.get(nb_idx)
                if sp_curr and sp_nb:
                    nn_d = nn_distances.get(tuple(sorted([sp_curr, sp_nb])))
                    if nn_d is not None and nb['dist'] > max_nn_ratio * nn_d:
                        continue
            best_cos = cos_angle
            best_nb  = nb

        if best_nb is None:
            break

        chain.append(best_nb['idx'])
        chain_length += best_nb['dist']
        current_dir   = best_nb['dir']   # unit vector already normalised
        if best_nb.get('boundary', False):
            n_boundary_crossings += 1

    return chain


def find_chains(flagged_records, supercell, bulk_results=None,
                min_angle_deg=150.0, max_order=5, bond_cutoff=4.0,
                nn_distances=None, max_nn_ratio=1.5,
                reliability_cutoff=None):
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
    if reliability_cutoff is not None:
        reliability_limit = float(reliability_cutoff)
    else:
        L = supercell.lattice
        a, b, c = L[0], L[1], L[2]
        V = abs(float(np.dot(a, np.cross(b, c))))
        reliability_limit = min(
            V / np.linalg.norm(np.cross(b, c)),
            V / np.linalg.norm(np.cross(a, c)),
            V / np.linalg.norm(np.cross(a, b)),
        ) / 2.0
    min_cos   = np.cos(np.radians(180.0 - min_angle_deg))
    neighbors = _build_neighbor_lookup_from_structure(supercell, bond_cutoff)

    atom_species = None
    if max_nn_ratio is not None and nn_distances is not None:
        atom_species = {idx: supercell.species(idx)
                        for idx in range(1, supercell.n_atoms + 1)}

    results = []
    for rec in flagged_records:
        start     = rec['atom1_idx']
        direction = np.array(rec['direction'], dtype=float)

        chain = _grow_chain(start, direction, neighbors,
                            min_cos, max_order, reliability_limit,
                            atom_species=atom_species,
                            nn_distances=nn_distances,
                            max_nn_ratio=max_nn_ratio)

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
    str, e.g. ``'cobiBetween atom5 atom1 cell -1 0 0 atom8'``

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
        bulk_results, supercell, poscar_lobster_path=None,
        n_sigma=1.5, min_pairs=4,
        min_angle_deg=150.0, max_order=5, bond_cutoff=4.0,
        detect_cutoff_frac=1.0,
        max_nn_ratio=1.5,
        fit_quantile=None,
        reliability_cutoff=None,
        _skip_symmetry_expand=False):
    """
    Full pipeline: detect anomalous pFCs → trace chains → format directives.

    Parameters
    ----------
    bulk_results         : list of dicts from compute_bulk_pfcs()
    supercell            : Supercell
    poscar_lobster_path  : path-like or None, POSCAR used for the LOBSTER
                           calculation.  When None, chain detection still runs
                           but cobiBetween directives are not generated.
    n_sigma              : float, anomaly detection threshold (sigma). Default 2.5.
    min_pairs            : int, min pairs for regression detection. Default 4.
    min_angle_deg        : float, minimum bond angle for chain extension. Default 150.
    max_order            : int, maximum atoms per chain. Default 5.
    bond_cutoff          : float, Å, max step distance for chain extension. Default 4.0.
    detect_cutoff_frac   : float, fraction of the reliability limit (L/2) used as
                           the maximum pair distance considered for flagging.
                           Pairs beyond this threshold still anchor the Badger
                           baseline but cannot be flagged.  Default 1.0 (full
                           reliability window).
    max_nn_ratio         : float or None, maximum allowed ratio of a single chain
                           step distance to the species-pair NN distance.  Steps
                           longer than this multiple of the NN are not traversed,
                           preventing chains from forming through non-bonded
                           contacts (e.g. diamond 2nd-NN at 1.63× vs genuine
                           multicenter segments at ≤1.0×).  Default 1.5.
    fit_quantile         : ignored, kept for backward compatibility.

    Returns
    -------
    dict with:
        'flagged_pairs' : list of detection entries from detect_anomalous_pairs()
        'chains'        : list of chain dicts; sub_chains[*]['directive'] is filled
                          (or None when poscar_lobster_path was not provided)
        'directives'    : list[str], unique cobiBetween lines ready for lobsterin
                          (empty when poscar_lobster_path is None)
    """
    from betapy.core.badger import compute_badger_quantities

    lob_poscar = (_parse_poscar_lobster(poscar_lobster_path)
                  if poscar_lobster_path is not None else None)

    if not _skip_symmetry_expand:
        try:
            from betapy.core.symmetry import expand_by_symmetry
            bulk_results = expand_by_symmetry(bulk_results, supercell, bond_cutoff)
        except Exception:
            pass

    if reliability_cutoff is not None:
        reliability_limit = float(reliability_cutoff)
    else:
        L = supercell.lattice
        a, b, c = L[0], L[1], L[2]
        V = abs(float(np.dot(a, np.cross(b, c))))
        reliability_limit = min(
            V / np.linalg.norm(np.cross(b, c)),
            V / np.linalg.norm(np.cross(a, c)),
            V / np.linalg.norm(np.cross(a, b)),
        ) / 2.0
    reliable_pairs = [r for r in bulk_results if r['distance'] <= reliability_limit]

    # Augment with Φ_iso so detection uses the orientation-invariant baseline.
    reliable_pairs = compute_badger_quantities(reliable_pairs)

    max_detect_dist = reliability_limit * detect_cutoff_frac

    flagged = detect_anomalous_pairs(
        reliable_pairs, n_sigma=n_sigma, min_pairs=min_pairs,
        value_key='phi_iso', max_detect_dist=max_detect_dist)

    # NN distance per species pair — used to reject chain steps that jump
    # through non-bonded contacts (segment-level max_nn_ratio check).
    import math as _math
    from collections import defaultdict as _dd
    _seen: set = set()
    _by_sp: dict = _dd(list)
    for r in reliable_pairs:
        _k = (min(r['atom1_idx'], r['atom2_idx']), max(r['atom1_idx'], r['atom2_idx']))
        if _k not in _seen:
            _seen.add(_k)
            phi = r.get('phi_iso', float('nan'))
            if _math.isfinite(phi) and phi > 0:
                _by_sp[tuple(sorted([r['species1'], r['species2']]))].append(r['distance'])
    nn_distances = {sp: min(ds) for sp, ds in _by_sp.items()}

    chains = find_chains(
        flagged, supercell, bulk_results,
        min_angle_deg=min_angle_deg, max_order=max_order, bond_cutoff=bond_cutoff,
        nn_distances=nn_distances, max_nn_ratio=max_nn_ratio,
        reliability_cutoff=reliability_limit,
    )

    seen_keys: set = set()
    unique_directives: list = []
    for chain in chains:
        for sub in chain['sub_chains']:
            if lob_poscar is None:
                sub['directive'] = None
                continue
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
