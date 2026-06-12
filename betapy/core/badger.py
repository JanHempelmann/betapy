"""
Badger analysis for betapy — experimental feature.

Fits the Badger-type law Φ^{-1/3} = a·r + b to projected force constants
using two complementary quantities:

  conventional  Φ_p   — projected along bond direction (existing approach)
  isotropic     F_iso — (|phi_l| + 2·|phi_t|) / 3  =  mean absolute eigenvalue

F_iso is rotationally invariant: it does not depend on which direction the
force-constant matrix is projected along, and therefore collapses the multiple
parallel Badger lines that appear in covalent systems (the "gleichergestalt"
splitting).  The dimensionless anisotropy factor

    ξ = Φ_p / F_iso

encodes which conventional Badger line a pair belongs to.  The vertical
scatter in the conventional Badger plot decomposes exactly as

    Φ_p^{-1/3}  =  F_iso^{-1/3} · ξ^{-1/3}

so the isotropic fit removes ξ-driven scatter while preserving the physically
motivated −1/3 exponent and its dimensional consistency.

Using the sum of absolute eigenvalues rather than the absolute trace avoids
the sign-cancellation outliers that arise when phi_l ≈ −2·phi_t (Tr ≈ 0)
for long-range or non-bonding pairs.

Both phi_l and phi_t are already stored in bulk_results by
compute_bulk_pfcs(); no re-reading of force-constant matrices is required.
"""

import numpy as np
from collections import defaultdict
from scipy.stats import theilslopes, median_abs_deviation


# ---------------------------------------------------------------------------
# Shell splitting (same algorithm as in multicenter.py; kept independent so
# the two modules can be tuned separately without cross-module coupling)
# ---------------------------------------------------------------------------

def _split_into_shells(records, rel_gap=0.50):
    """
    Partition bond records into distance shells separated by large gaps.

    A boundary is placed where the relative gap between consecutive distances
    exceeds *rel_gap* (default 50 %).  Tuned to split the diamond 1st→2nd
    shell C-C gap (~63 %) while leaving GeTe short/long Ge-Te bonds (~12 %)
    and Sb₂Te₃ long-range Sb-Te bonds intact.

    Returns a list of lists; each sub-list contains records for one shell.
    """
    if not records:
        return [records]
    dists  = np.array([r['distance'] for r in records])
    order  = np.argsort(dists)
    sorted_d = dists[order]

    split_at = []
    for k in range(len(sorted_d) - 1):
        if sorted_d[k] > 0 and (sorted_d[k + 1] - sorted_d[k]) / sorted_d[k] > rel_gap:
            split_at.append(k + 1)

    if not split_at:
        return [records]

    shells, prev = [], 0
    for sp in split_at:
        shells.append([records[i] for i in order[prev:sp]])
        prev = sp
    shells.append([records[i] for i in order[prev:]])
    return [s for s in shells if s]


# ---------------------------------------------------------------------------
# NN-anchor helpers shared by fan-slope and nn-bond routines
# ---------------------------------------------------------------------------

def _nn_cutoffs(records, abs_gap=0.10):
    """
    Return per-species-pair NN cutoff distances (1st-shell max × 1.05).

    Uses an absolute distance gap rather than the relative-gap heuristic in
    _split_into_shells.  The relative criterion (default 50 %) fails for
    species pairs whose first two shells are close together in absolute terms
    but similar in relative distance — e.g. Si-Si in quartz sits at ~3.07 Å
    and ~3.30 Å (7 % relative gap), so the 50 %-threshold misses the boundary
    and treats the entire dataset as the first shell.  An absolute threshold of
    0.10 Å (0.22 Å gap for Si-Si in quartz) is robust across all typical
    crystalline bond lengths without affecting well-separated shells such as
    C-C in diamond (0.98 Å gap).
    """
    seen, by_sp = set(), defaultdict(list)
    for r in records:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        if key in seen:
            continue
        seen.add(key)
        by_sp[tuple(sorted([r['species1'], r['species2']]))].append(r)

    cutoffs = {}
    for sp_key, sp_recs in by_sp.items():
        dists = np.sort([r['distance'] for r in sp_recs if r['distance'] > 1e-6])
        if len(dists) == 0:
            continue
        cutoff = float(dists[-1])          # fallback: no gap found
        for k in range(len(dists) - 1):
            if dists[k + 1] - dists[k] > abs_gap:
                cutoff = float(dists[k]) * 1.05
                break
        cutoffs[sp_key] = cutoff
    return cutoffs


def _find_nn_anchor(records, value_key='mean_pfc'):
    """
    Return per-species-pair NN anchor points (r*, y*) for the fan model.

    (r*, y*) is the mean (distance, value^{-1/3}) of the first coordination
    shell — the approximate left-side convergence point of the Badger fan.
    """
    cutoffs = _nn_cutoffs(records)

    seen, by_sp = set(), defaultdict(list)
    for r in records:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        if key in seen:
            continue
        seen.add(key)
        by_sp[tuple(sorted([r['species1'], r['species2']]))].append(r)

    anchors = {}
    for sp_key, sp_recs in by_sp.items():
        cutoff = cutoffs.get(sp_key, float('inf'))
        valid  = [r for r in sp_recs
                  if r['distance'] <= cutoff
                  and np.isfinite(r.get(value_key, float('nan')))
                  and r[value_key] > 0]
        if not valid:
            continue
        anchors[sp_key] = (
            float(np.mean([r['distance']                       for r in valid])),
            float(np.mean([r[value_key] ** (-1.0 / 3.0)       for r in valid])),
        )
    return anchors


# ---------------------------------------------------------------------------
# Augment records with isotropic quantities and bond-anisotropy ratio
# ---------------------------------------------------------------------------

def compute_badger_quantities(bulk_results):
    """
    Add 'phi_iso', 'xi', and 'eta_pair' to each bulk result record.

    Uses phi_l and phi_t already computed by compute_bulk_pfcs().

        F_iso    = (|phi_l| + 2·|phi_t|) / 3  — mean absolute eigenvalue;
                   rotationally invariant and free of sign-cancellation
                   outliers (cf. |Tr(Φ)|/3 which → 0 when phi_l ≈ −2·phi_t)
        ξ        = mean_pfc / F_iso
        eta_pair = |phi_l / phi_t|  — longitudinal-to-transverse anisotropy
                   of the actual FC matrix for this pair; independent of
                   projection direction.  Large (>>1) for covalent bonds,
                   ~1–2 for ionic or isotropic interactions.

    Parameters
    ----------
    bulk_results : list of dicts from compute_bulk_pfcs().  Records must
                   contain 'phi_l', 'phi_t', and 'mean_pfc'.

    Returns
    -------
    list of new dicts — original records are not modified.
    """
    augmented = []
    for r in bulk_results:
        phi_l    = r.get('phi_l', float('nan'))
        phi_t    = r.get('phi_t', float('nan'))
        mean_pfc = r.get('mean_pfc', float('nan'))

        if not (np.isfinite(phi_l) and np.isfinite(phi_t) and np.isfinite(mean_pfc)):
            augmented.append({**r, 'phi_iso': float('nan'), 'xi': float('nan'),
                               'eta_pair': float('nan')})
            continue

        phi_iso  = (abs(phi_l) + 2.0 * abs(phi_t)) / 3.0
        xi       = mean_pfc / phi_iso if phi_iso > 1e-12 else float('nan')
        eta_pair = (abs(phi_l / phi_t)
                    if abs(phi_t) > 1e-12 else float('nan'))
        augmented.append({**r, 'phi_iso': phi_iso, 'xi': xi, 'eta_pair': eta_pair})
    return augmented


# ---------------------------------------------------------------------------
# Geometric bond-family classifier
# ---------------------------------------------------------------------------

def _compute_nn_bonds(records):
    """
    Build a per-atom dict of nearest-neighbor bond unit vectors.

    Uses the 1st coordination shell (smallest distance group per species
    pair, found by _split_into_shells).  Duplicate (i,j)/(j,i) records are
    collapsed so each physical bond is added exactly once per atom.

    Returns
    -------
    dict : atom_idx -> list of np.ndarray unit vectors pointing *away* from
           that atom toward each nearest neighbor.
    """
    nn_cutoffs = _nn_cutoffs(records)

    # Collect bond directions for each atom
    nn_bonds = defaultdict(list)
    seen = set()
    for r in records:
        i, j = r['atom1_idx'], r['atom2_idx']
        key  = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)

        sp_key = tuple(sorted([r['species1'], r['species2']]))
        cutoff = nn_cutoffs.get(sp_key)
        if cutoff is None or r['distance'] > cutoff:
            continue

        d    = r.get('direction', [0., 0., 0.])
        d_arr = np.array(d, dtype=float)
        norm  = np.linalg.norm(d_arr)
        if norm < 1e-10:
            continue
        d_hat = d_arr / norm

        nn_bonds[i].append(d_hat)    # i → j
        nn_bonds[j].append(-d_hat)   # j → i

    return nn_bonds


def compute_theta_geo(records):
    """
    Add 'cos2theta' to each record.

    cos²θ_geo is the maximum squared dot-product between the pair unit
    vector r̂_ij and any nearest-neighbor bond direction at either atom
    end:

        cos²θ = max( max_k (r̂_ij · b̂_k)²,
                     max_k (-r̂_ij · b̂_k)² )

    Interpretation
    --------------
    1   — pair vector is exactly along a nearest-neighbor bond (θ = 0°)
    2/3 — pair bisects two tetrahedral bonds (e.g. ⟨110⟩ in diamond, θ ≈ 35°)
    1/3 — pair along ⟨100⟩ in a tetrahedral lattice (θ ≈ 55°)
    0   — pair is perpendicular to every nearest-neighbor bond (θ = 90°)

    Purely geometric: computed from pair directions and crystal structure,
    independent of force-constant values.
    """
    nn_bonds = _compute_nn_bonds(records)

    augmented = []
    for r in records:
        i, j  = r['atom1_idx'], r['atom2_idx']
        d_arr = np.array(r.get('direction', [0., 0., 0.]), dtype=float)
        norm  = np.linalg.norm(d_arr)

        bonds_i = nn_bonds.get(i, [])
        bonds_j = nn_bonds.get(j, [])

        if norm < 1e-10 or (not bonds_i and not bonds_j):
            augmented.append({**r, 'cos2theta': float('nan')})
            continue

        r_hat     = d_arr / norm
        candidates = []
        if bonds_i:
            candidates.append(max(float(np.dot( r_hat, b) ** 2) for b in bonds_i))
        if bonds_j:
            candidates.append(max(float(np.dot(-r_hat, b) ** 2) for b in bonds_j))
        augmented.append({**r, 'cos2theta': max(candidates)})

    return augmented


# ---------------------------------------------------------------------------
# Fan-slope family detection
# ---------------------------------------------------------------------------

def compute_fan_slopes(records, value_key='mean_pfc'):
    """
    Add 'fan_slope' to each record.

    For a Badger fan whose lines approximately converge near the NN shell,
    the slope from that anchor uniquely identifies each family:

        α = (value^{-1/3} − y*) / (r − r*)

    where (r*, y*) is the mean NN-shell point (the approximate left-side
    convergence of the wedge).  Pairs at or within 0.1 Å of the anchor
    receive NaN — they define the convergence point, not a family divergence.

    Mathematical note: for pairs that truly lie on a line value^{-1/3} = a·r + b
    passing through (r*, y*), α equals the Badger slope a exactly.  Imperfect
    convergence introduces a small, distance-dependent bias that is small
    provided r* is close to the true convergence.
    """
    anchors  = _find_nn_anchor(records, value_key)
    cutoffs  = _nn_cutoffs(records)

    augmented = []
    for r in records:
        sp_key = tuple(sorted([r['species1'], r['species2']]))
        val    = r.get(value_key, float('nan'))
        dist   = r['distance']
        anchor = anchors.get(sp_key)
        cutoff = cutoffs.get(sp_key, float('inf'))

        if (anchor is None
                or not np.isfinite(val) or val <= 0
                or dist <= cutoff
                or dist - anchor[0] < 0.1):
            augmented.append({**r, 'fan_slope': float('nan')})
            continue

        r_star, y_star = anchor
        alpha = (val ** (-1.0 / 3.0) - y_star) / (dist - r_star)
        augmented.append({**r, 'fan_slope': float(alpha)})
    return augmented


def _kmeans_1d(values, k, n_iter=150):
    """
    Simple 1-D k-means.  Returns sorted array of k cluster centres.
    Initialises centres at evenly-spaced quantiles of *values*.
    """
    arr = np.sort(values)
    idx = np.linspace(0, len(arr) - 1, k, dtype=int)
    centres = arr[idx].astype(float)
    for _ in range(n_iter):
        dists     = np.abs(arr[:, None] - centres[None, :])   # (n, k)
        labels    = np.argmin(dists, axis=1)
        new_c     = np.array([arr[labels == j].mean()
                               if (labels == j).any() else centres[j]
                               for j in range(k)])
        if np.allclose(new_c, centres, rtol=1e-6):
            break
        centres = new_c
    return np.sort(centres)


def assign_families_kmeans(records, n_families=5, fuzzy_frac=0.30):
    """
    Add 'family_ids' (list of ints) to each record.

    1-D k-means is run on the fan_slope values per species pair.  Each pair is
    assigned to its nearest cluster (hard assignment) plus any adjacent cluster
    whose boundary it is within fuzzy_frac × inter-centre distance of — so one
    pair can belong to two families simultaneously.

    Families are numbered 0 … (k-1) in order of increasing fan_slope
    (0 = shallowest decay = stiffest).  Pairs without a valid fan_slope
    (NN shell) receive family_ids = [-1].

    Parameters
    ----------
    n_families  : int   — requested k for k-means
    fuzzy_frac  : float — fraction of inter-centre gap used as overlap zone
    """
    by_sp = defaultdict(list)
    for i, r in enumerate(records):
        sp_key = tuple(sorted([r['species1'], r['species2']]))
        by_sp[sp_key].append((i, r))

    fam_ids_all = [[-1] for _ in range(len(records))]

    for sp_key, idx_recs in by_sp.items():
        valid = [(i, r) for i, r in idx_recs
                 if np.isfinite(r.get('fan_slope', float('nan')))]
        if not valid:
            continue

        k_eff   = min(n_families, len(valid))
        slopes  = np.array([r['fan_slope'] for _, r in valid])
        centres = _kmeans_1d(slopes, k_eff)
        bounds  = (centres[:-1] + centres[1:]) / 2.0

        for idx, (i, _) in enumerate(valid):
            s    = slopes[idx]
            hard = int(np.searchsorted(bounds, s))
            mems = {hard}

            if hard > 0:
                gap = centres[hard] - centres[hard - 1]
                if gap > 0 and s - bounds[hard - 1] < fuzzy_frac * gap:
                    mems.add(hard - 1)
            if hard < k_eff - 1:
                gap = centres[hard + 1] - centres[hard]
                if gap > 0 and bounds[hard] - s < fuzzy_frac * gap:
                    mems.add(hard + 1)

            fam_ids_all[i] = sorted(mems)

    return [{**r, 'family_ids': fam_ids_all[i]} for i, r in enumerate(records)]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_one_shell(records, value_key, min_pairs):
    """
    Fit value^{-1/3} = a·r + b on a single shell using Theil-Sen regression.

    Returns a fit dict, or None if there are fewer than 2 valid pairs or
    less than 0.05 Å of distance variation (degenerate shell).

    The residuals dict uses canonical (min_idx, max_idx) pair keys so that
    symmetric (i,j) / (j,i) duplicates in the raw data both resolve to the
    same entry.
    """
    vals  = np.array([r[value_key] for r in records], dtype=float)
    dists = np.array([r['distance'] for r in records], dtype=float)

    valid = np.isfinite(vals) & (vals > 0)
    if valid.sum() < 2:
        return None

    v_vals    = vals[valid]
    v_dists   = dists[valid]
    v_records = [r for r, v in zip(records, valid) if v]

    if float(v_dists.max() - v_dists.min()) < 0.05:
        return None

    inv_cbrt = v_vals ** (-1.0 / 3.0)

    # One representative per unique (distance, value) pair for the slope fit.
    # Symmetry-equivalent bonds share both, so this collapses them without
    # merging geometrically distinct bonds at the same distance (e.g. NaCl has
    # two bond types at 1.5a with different force constants).
    # Residuals are still evaluated for all points.
    _keys = np.stack([np.round(v_dists, 3), np.round(inv_cbrt, 4)], axis=1)
    _, ux = np.unique(_keys, axis=0, return_index=True)
    slope, intercept, *_ = theilslopes(inv_cbrt[ux], v_dists[ux],
                                       method='joint')

    predicted = slope * v_dists + intercept
    residuals = inv_cbrt - predicted       # negative => value larger than fit predicts

    std_raw = float(median_abs_deviation(residuals) * 1.4826)
    std     = max(std_raw, 1e-6)

    # Robust R²: replace mean with median for total SS (insensitive to outliers)
    med    = float(np.median(inv_cbrt))
    ss_tot = float(np.sum((inv_cbrt - med) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    res_lookup = {
        (min(rec['atom1_idx'], rec['atom2_idx']),
         max(rec['atom1_idx'], rec['atom2_idx'])): float(res)
        for rec, res in zip(v_records, residuals)
    }

    return {
        'slope':     float(slope),
        'intercept': float(intercept),
        'std':       std,
        'r2_robust': r2,
        'n_pairs':   int(valid.sum()),
        'r_min':     float(v_dists.min()),
        'r_max':     float(v_dists.max()),
        'residuals': res_lookup,
    }


def fit_badger_line(records, value_key='mean_pfc', min_pairs=4, shell_split=True):
    """
    Fit value^{-1/3} = a·r + b per species pair using Theil-Sen regression.

    Parameters
    ----------
    records     : list of dicts.  Must contain 'species1', 'species2',
                  'distance', 'atom1_idx', 'atom2_idx', and value_key.
    value_key   : str — 'mean_pfc' for conventional, 'phi_iso' for isotropic.
    min_pairs   : int — minimum valid pairs for regression (default 4).
    shell_split : bool — split by distance shells before fitting (default True).

    Returns
    -------
    dict keyed by (species1, species2) in alphabetical order, mapping to a
    list of per-shell fit dicts.  Each shell fit dict contains:
        slope, intercept, std, r2_robust, n_pairs, r_min, r_max,
        residuals: {(min_idx, max_idx) -> float}
    """
    # Remove symmetric duplicates: FORCE_CONSTANTS lists every (i,j) and (j,i).
    seen, deduped = set(), []
    for r in records:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    by_pair = defaultdict(list)
    for r in deduped:
        sp_key = tuple(sorted([r['species1'], r['species2']]))
        by_pair[sp_key].append(r)

    fit_results = {}
    for sp_key, pair_records in by_pair.items():
        shells = (_split_into_shells(pair_records)
                  if shell_split and len(pair_records) >= min_pairs
                  else [pair_records])

        shell_fits = []
        for shell_records in shells:
            if len(shell_records) < 2:
                continue
            fit = _fit_one_shell(shell_records, value_key, min_pairs)
            if fit is not None:
                shell_fits.append(fit)

        if shell_fits:
            fit_results[sp_key] = shell_fits

    return fit_results


def fit_badger_families(records, value_key='mean_pfc'):
    """
    Fit one Badger line per (species pair, family_id), constrained to pass
    through the 1NN anchor (r*, y*).

    Because all gleichergestalt families share the same nearest-neighbour bond
    (same distance, same force constant), their Badger lines must converge at
    the 1NN point.  Enforcing this reduces each family fit to a single free
    parameter: the slope α (the median fan_slope of the family's members).
    The intercept follows as  b = y* − α·r*.

    This makes every family line well-determined regardless of how few
    members it has — even two points uniquely fix the median slope.

    Returns
    -------
    dict : {(sp_key, family_id): fit_dict}
        fit_dict keys: slope, intercept, r_min, r_max, n_pairs, r_anchor
    """
    anchors = _find_nn_anchor(records, value_key)

    seen, deduped = set(), []
    for r in records:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    by_key = defaultdict(list)
    for r in deduped:
        sp_key = tuple(sorted([r['species1'], r['species2']]))
        for fid in r.get('family_ids', [-1]):
            if fid < 0:
                continue
            by_key[(sp_key, fid)].append(r)

    fits = {}
    for (sp_key, fid), recs in by_key.items():
        anchor = anchors.get(sp_key)
        if anchor is None:
            continue
        r_star, y_star = anchor

        valid_slopes = [r['fan_slope'] for r in recs
                        if np.isfinite(r.get('fan_slope', float('nan')))]
        if not valid_slopes:
            continue

        slope     = float(np.median(valid_slopes))
        intercept = y_star - slope * r_star
        dists     = [r['distance'] for r in recs]
        fits[(sp_key, fid)] = {
            'slope':     slope,
            'intercept': intercept,
            'r_min':     float(min(dists)),
            'r_max':     float(max(dists)),
            'n_pairs':   len(recs),
            'r_anchor':  r_star,
        }
    return fits


# ---------------------------------------------------------------------------
# Result container and residual attachment
# ---------------------------------------------------------------------------

class BadgerAnalysisResult:
    """
    Container for a completed Badger analysis.

    Attributes
    ----------
    records      : list of augmented dicts — each record from bulk_results with:
                     'phi_iso'         float  — isotropic mean stiffness
                     'xi'            float  — anisotropy factor mean_pfc / phi_iso
                     'cos2theta'     float  — geometric alignment with NN bond
                     'fan_slope'     float  — α = (Φ^{-1/3}−y*)/(r−r*); family ID
                     'family_id'     int    — cluster index (0 = stiffest family)
                     'conv_residual' float  — residual of mean_pfc^{-1/3} from fit
                     'iso_residual'  float  — residual of phi_iso^{-1/3} from fit
                   NaN / −1 where a fit or assignment could not be made.
    conv_fits    : fit_badger_line() result for conventional (mean_pfc) Badger
    iso_fits     : fit_badger_line() result for isotropic (phi_iso) Badger
    family_fits  : fit_badger_families() result — per-(sp_key, family_id) fits
    """
    __slots__ = ('records', 'conv_fits', 'iso_fits', 'family_fits')

    def __init__(self, records, conv_fits, iso_fits, family_fits):
        self.records      = records
        self.conv_fits    = conv_fits
        self.iso_fits     = iso_fits
        self.family_fits  = family_fits


def _attach_residuals(records, fit_results, res_key):
    """
    Write per-record residuals from a fit_badger_line() result into records.

    Looks up each record's canonical (min_idx, max_idx) pair key in the
    flattened residuals from all species-pair shells.  Records without a
    match (e.g. pairs excluded from fitting) receive NaN.
    """
    lookup = {}
    for shell_fits in fit_results.values():
        for shell_fit in shell_fits:
            lookup.update(shell_fit['residuals'])

    for r in records:
        key = (min(r['atom1_idx'], r['atom2_idx']),
               max(r['atom1_idx'], r['atom2_idx']))
        r[res_key] = lookup.get(key, float('nan'))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_badger(bulk_results, n_families=5):
    """
    Run Badger analysis on bulk pFC results.

    Computes F_iso and ξ per pair, fits both the conventional (Φ_p) and
    isotropic (F_iso) Badger lines per species pair, detects Badger families
    via fan-slope k-means clustering, and attaches per-record residuals.

    Parameters
    ----------
    bulk_results : list of dicts from compute_bulk_pfcs().
    n_families   : int — number of Badger families to detect (default 5).

    Returns
    -------
    BadgerAnalysisResult
    """
    augmented    = compute_badger_quantities(bulk_results)
    augmented    = compute_theta_geo(augmented)
    augmented    = compute_fan_slopes(augmented)
    augmented    = assign_families_kmeans(augmented, n_families=n_families)
    conv_fits    = fit_badger_line(augmented, value_key='mean_pfc')
    iso_fits     = fit_badger_line(augmented, value_key='phi_iso', shell_split=False)
    family_fits  = fit_badger_families(augmented)
    _attach_residuals(augmented, conv_fits, 'conv_residual')
    _attach_residuals(augmented, iso_fits,  'iso_residual')
    return BadgerAnalysisResult(augmented, conv_fits, iso_fits, family_fits)
