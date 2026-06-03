"""
Badger analysis for betapy — experimental feature.

Fits the Badger-type law Φ^{-1/3} = a·r + b to projected force constants
using two complementary quantities:

  conventional  Φ_p   — projected along bond direction (existing approach)
  isotropic     F_iso — |Tr(Φ)| / 3  =  |phi_l + 2·phi_t| / 3

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

Sign note: Phonopy's off-diagonal FC blocks for bonded pairs are negative
(restoring convention), so phi_l < 0, phi_t < 0, Tr(Φ) < 0 for bonded pairs.
F_iso takes the absolute value so it is always a positive stiffness.

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
# Augment records with isotropic quantities
# ---------------------------------------------------------------------------

def compute_badger_quantities(bulk_results):
    """
    Add 'f_iso' and 'xi' to each bulk result record.

    Uses phi_l and phi_t already computed by compute_bulk_pfcs().

        F_iso = |phi_l + 2·phi_t| / 3  =  |Tr(Φ)| / 3
        ξ     = mean_pfc / F_iso

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
            augmented.append({**r, 'f_iso': float('nan'), 'xi': float('nan')})
            continue

        f_iso = abs(phi_l + 2.0 * phi_t) / 3.0
        xi    = mean_pfc / f_iso if f_iso > 1e-12 else float('nan')
        augmented.append({**r, 'f_iso': f_iso, 'xi': xi})
    return augmented


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

    slope, intercept, *_ = theilslopes(inv_cbrt, v_dists)

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
    value_key   : str — 'mean_pfc' for conventional, 'f_iso' for isotropic.
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


# ---------------------------------------------------------------------------
# Result container and residual attachment
# ---------------------------------------------------------------------------

class BadgerAnalysisResult:
    """
    Container for a completed Badger analysis.

    Attributes
    ----------
    records    : list of augmented dicts — each record from bulk_results with:
                   'f_iso'         float  — isotropic mean stiffness |Tr(Φ)|/3
                   'xi'            float  — anisotropy factor mean_pfc / f_iso
                   'conv_residual' float  — residual of mean_pfc^{-1/3} from fit
                   'iso_residual'  float  — residual of f_iso^{-1/3} from fit
                 NaN where a fit could not be assigned.
    conv_fits  : fit_badger_line() result for conventional (mean_pfc) Badger
    iso_fits   : fit_badger_line() result for isotropic (f_iso) Badger
    """
    __slots__ = ('records', 'conv_fits', 'iso_fits')

    def __init__(self, records, conv_fits, iso_fits):
        self.records   = records
        self.conv_fits = conv_fits
        self.iso_fits  = iso_fits


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

def analyze_badger(bulk_results):
    """
    Run Badger analysis on bulk pFC results.

    Computes F_iso and ξ per pair, fits both the conventional (Φ_p) and
    isotropic (F_iso) Badger lines per species pair, and attaches per-record
    residuals.

    Parameters
    ----------
    bulk_results : list of dicts from compute_bulk_pfcs().

    Returns
    -------
    BadgerAnalysisResult
    """
    augmented = compute_badger_quantities(bulk_results)
    conv_fits = fit_badger_line(augmented, value_key='mean_pfc')
    iso_fits  = fit_badger_line(augmented, value_key='f_iso')
    _attach_residuals(augmented, conv_fits, 'conv_residual')
    _attach_residuals(augmented, iso_fits,  'iso_residual')
    return BadgerAnalysisResult(augmented, conv_fits, iso_fits)
