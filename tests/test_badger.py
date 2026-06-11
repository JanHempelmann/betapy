"""
Tests for betapy.core.badger

Run with:  python -m pytest tests/
"""

import math
import numpy as np
import pytest

from betapy.core.badger import (
    compute_badger_quantities,
    fit_badger_line,
    analyze_badger,
    BadgerAnalysisResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pair(sp1, sp2, dist, mean_pfc, phi_l, phi_t,
              atom1_idx=None, atom2_idx=None):
    """Build a minimal bulk_result-style record."""
    # Use atom indices that produce a unique canonical key per call.
    # Caller may override for explicit deduplication tests.
    return {
        'atom1_idx': atom1_idx if atom1_idx is not None else id(object()),
        'atom2_idx': atom2_idx if atom2_idx is not None else id(object()) + 1,
        'species1':  sp1,
        'species2':  sp2,
        'distance':  dist,
        'mean_pfc':  mean_pfc,
        'phi_l':     phi_l,
        'phi_t':     phi_t,
        'direction': [1., 0., 0.],
    }


# ---------------------------------------------------------------------------
# compute_badger_quantities
# ---------------------------------------------------------------------------

class TestComputeBadgerQuantities:

    def test_f_iso_formula(self):
        # phi_l = -k_L, phi_t = -k_T  (Phonopy restoring convention)
        # F_iso = (|phi_l| + 2*|phi_t|) / 3 = (k_L + 2*k_T) / 3
        k_L, k_T = 10.0, 2.0
        r = make_pair('Na', 'Cl', 2.8, k_L, phi_l=-k_L, phi_t=-k_T)
        aug = compute_badger_quantities([r])
        expected = (k_L + 2.0 * k_T) / 3.0
        assert abs(aug[0]['phi_iso'] - expected) < 1e-10

    def test_xi_formula(self):
        k_L, k_T = 9.0, 0.0   # purely longitudinal → xi = 3
        r = make_pair('Na', 'Cl', 2.8, k_L, phi_l=-k_L, phi_t=-k_T)
        aug = compute_badger_quantities([r])
        assert abs(aug[0]['xi'] - 3.0) < 1e-10

    def test_isotropic_xi_is_one(self):
        # All three eigenvalues equal → F_iso = phi_l, xi = 1
        k = 6.0
        # phi_l = -k, phi_t = (Tr - phi_l)/2; for isotropic: Tr = 3*(-k)
        # phi_t = (3*(-k) - (-k))/2 = -k
        r = make_pair('Ge', 'Te', 3.0, k, phi_l=-k, phi_t=-k)
        aug = compute_badger_quantities([r])
        assert abs(aug[0]['phi_iso'] - k) < 1e-10
        assert abs(aug[0]['xi'] - 1.0) < 1e-10

    def test_nan_propagation(self):
        r = make_pair('C', 'C', 1.5, 5.0, phi_l=float('nan'), phi_t=-1.0)
        aug = compute_badger_quantities([r])
        assert math.isnan(aug[0]['phi_iso'])
        assert math.isnan(aug[0]['xi'])

    def test_zero_f_iso_gives_nan_xi(self):
        # F_iso = (|phi_l| + 2*|phi_t|) / 3 = 0 only when both are zero → xi = nan
        r = make_pair('X', 'Y', 2.0, 1.0, phi_l=0.0, phi_t=0.0)
        aug = compute_badger_quantities([r])
        assert aug[0]['phi_iso'] < 1e-12
        assert math.isnan(aug[0]['xi'])

    def test_mixed_sign_phi_no_cancellation(self):
        # Old formula |phi_l + 2*phi_t|/3 cancels to 0 for phi_l=2, phi_t=-1;
        # new formula uses absolute values to avoid sign cancellation
        r = make_pair('X', 'Y', 2.0, 1.0, phi_l=2.0, phi_t=-1.0)
        aug = compute_badger_quantities([r])
        expected = (abs(2.0) + 2.0 * abs(-1.0)) / 3.0  # = 4/3
        assert abs(aug[0]['phi_iso'] - expected) < 1e-10
        assert not math.isnan(aug[0]['xi'])

    def test_original_records_not_mutated(self):
        r = make_pair('Na', 'Cl', 2.8, 5.0, phi_l=-5.0, phi_t=-1.0)
        original_keys = set(r.keys())
        compute_badger_quantities([r])
        assert set(r.keys()) == original_keys


# ---------------------------------------------------------------------------
# fit_badger_line
# ---------------------------------------------------------------------------

def _ionic_records(n=20, seed=42):
    """
    Synthetic ionic-like records: mean_pfc = (r/r0)^{-3} with small noise.
    phi_l = -mean_pfc, phi_t = 0 (purely longitudinal).
    Atom indices are unique per record.
    """
    rng = np.random.default_rng(seed)
    r0  = 2.5
    records = []
    for k in range(n):
        r = r0 + k * 0.15
        pfc = (r / r0) ** (-3.0) + rng.normal(0, 0.01)
        pfc = max(pfc, 1e-6)
        records.append({
            'atom1_idx': 2 * k + 1,
            'atom2_idx': 2 * k + 2,
            'species1':  'Na',
            'species2':  'Cl',
            'distance':  r,
            'mean_pfc':  pfc,
            'phi_l':     -pfc,
            'phi_t':     0.0,
            'direction': [1., 0., 0.],
        })
    return records


class TestFitBadgerLine:

    def test_returns_fit_for_species_pair(self):
        records = _ionic_records()
        fits = fit_badger_line(records, value_key='mean_pfc')
        # sorted(['Na', 'Cl']) = ['Cl', 'Na'] ('C' < 'N')
        assert ('Cl', 'Na') in fits

    def test_slope_positive_for_ionic(self):
        # For pfc ∝ r^{-3}: pfc^{-1/3} ∝ r → positive slope
        records = _ionic_records(n=30)
        fits = fit_badger_line(records, value_key='mean_pfc')
        slope = fits[('Cl', 'Na')][0]['slope']
        assert slope > 0

    def test_r2_robust_near_one_for_clean_ionic(self):
        records = _ionic_records(n=30, seed=0)
        fits = fit_badger_line(records, value_key='mean_pfc')
        r2 = fits[('Cl', 'Na')][0]['r2_robust']
        assert r2 > 0.98

    def test_residuals_keyed_by_canonical_pair(self):
        records = _ionic_records(n=10)
        fits = fit_badger_line(records, value_key='mean_pfc')
        for shell_fit in fits[('Cl', 'Na')]:
            for (i, j) in shell_fit['residuals']:
                assert i <= j, "residual keys must use canonical (min, max) ordering"

    def test_species_order_normalised(self):
        # Records with (Cl, Na) and (Na, Cl) must land in the same fit group
        records = _ionic_records(n=10)
        swapped = []
        for k, r in enumerate(records):
            if k % 2 == 0:
                swapped.append({**r, 'species1': 'Cl', 'species2': 'Na'})
            else:
                swapped.append(r)
        fits = fit_badger_line(swapped, value_key='mean_pfc')
        assert ('Cl', 'Na') in fits
        assert len(fits) == 1, "both orderings must merge into a single group"

    def test_insufficient_pairs_returns_empty(self):
        # Only 2 records, need min_pairs=4
        records = _ionic_records(n=2)
        fits = fit_badger_line(records, value_key='mean_pfc', min_pairs=4)
        # Either no entry or the shell fit returns None (filtered)
        sp_fits = fits.get(('Na', 'Cl'), [])
        assert len(sp_fits) == 0

    def test_degenerate_distance_returns_no_fit(self):
        # All pairs at the same distance → no Badger slope can be defined
        records = [
            {'atom1_idx': i*2+1, 'atom2_idx': i*2+2,
             'species1': 'A', 'species2': 'B',
             'distance': 2.5, 'mean_pfc': 1.0 + i*0.001,
             'phi_l': -1.0, 'phi_t': 0.0, 'direction': [1.,0.,0.]}
            for i in range(10)
        ]
        fits = fit_badger_line(records, value_key='mean_pfc')
        assert fits.get(('A', 'B'), []) == []


# ---------------------------------------------------------------------------
# analyze_badger — integration
# ---------------------------------------------------------------------------

class TestAnalyzeBadger:

    def test_returns_result_object(self):
        records = _ionic_records(n=20)
        result = analyze_badger(records)
        assert isinstance(result, BadgerAnalysisResult)

    def test_records_augmented_with_f_iso_xi(self):
        records = _ionic_records(n=10)
        result = analyze_badger(records)
        for r in result.records:
            assert 'phi_iso' in r
            assert 'xi' in r

    def test_residuals_attached(self):
        records = _ionic_records(n=20)
        result = analyze_badger(records)
        has_conv = sum(1 for r in result.records
                       if not math.isnan(r.get('conv_residual', float('nan'))))
        has_iso  = sum(1 for r in result.records
                       if not math.isnan(r.get('iso_residual',  float('nan'))))
        assert has_conv > 0
        assert has_iso  > 0

    def test_decomposition_identity(self):
        """
        For purely longitudinal (ionic-limit) records: phi_t = 0, so
        f_iso = |phi_l| / 3 = mean_pfc / 3, giving xi = 3.
        The decomposition says:
            mean_pfc^{-1/3}  =  f_iso^{-1/3} * xi^{-1/3}
                              =  (mean_pfc/3)^{-1/3} * 3^{-1/3}
                              =  mean_pfc^{-1/3} * 3^{1/3} * 3^{-1/3}
                              =  mean_pfc^{-1/3}   ✓
        Check numerically that the decomposition holds for each record.
        """
        records = _ionic_records(n=20)
        result = analyze_badger(records)
        for r in result.records:
            if not (math.isfinite(r['phi_iso']) and r['phi_iso'] > 0 and
                    r['mean_pfc'] > 0 and math.isfinite(r['xi'])):
                continue
            lhs = r['mean_pfc'] ** (-1.0 / 3.0)
            rhs = r['phi_iso']  ** (-1.0 / 3.0) * r['xi'] ** (-1.0 / 3.0)
            assert abs(lhs - rhs) < 1e-10, (
                f"decomposition failed: lhs={lhs}, rhs={rhs}")

    def test_iso_r2_not_worse_than_conv_for_ionic(self):
        # For purely longitudinal records, iso and conv should give similar R²
        records = _ionic_records(n=30, seed=1)
        result = analyze_badger(records)
        sp = ('Cl', 'Na')
        conv_r2 = result.conv_fits[sp][0]['r2_robust']
        iso_r2  = result.iso_fits[sp][0]['r2_robust']
        # Both should be high; iso should not be meaningfully worse
        assert iso_r2 >= conv_r2 - 0.05
