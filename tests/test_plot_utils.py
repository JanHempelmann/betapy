"""
Tests for betapy.gui.plot_utils

Run with:  python -m pytest tests/
"""

import numpy as np
import pytest

from betapy.gui.plot_utils import symmetric_xlim, exact_energy_ylim


# ---------------------------------------------------------------------------
# symmetric_xlim
# ---------------------------------------------------------------------------

def test_symmetric_xlim_single_curve():
    low, high = symmetric_xlim([np.array([-0.1, 0.3, -0.05])])
    assert low == pytest.approx(-high)
    assert high == pytest.approx(0.3 * 1.05)


def test_symmetric_xlim_multiple_curves_uses_global_max():
    low, high = symmetric_xlim([np.array([-0.1, 0.2]), np.array([0.5, -0.4])])
    assert low == pytest.approx(-high)
    assert high == pytest.approx(0.5 * 1.05)


def test_symmetric_xlim_asymmetric_data_still_symmetric_limits():
    """A curve that's almost entirely positive still gets a centered-on-zero axis."""
    low, high = symmetric_xlim([np.array([0.0, 0.01, 0.02, 10.0])])
    assert low == pytest.approx(-high)
    assert high == pytest.approx(10.0 * 1.05)


def test_symmetric_xlim_ignores_none_entries():
    low, high = symmetric_xlim([None, np.array([-2.0, 1.0]), None])
    assert high == pytest.approx(2.0 * 1.05)


def test_symmetric_xlim_all_zero_returns_none():
    assert symmetric_xlim([np.zeros(5)]) is None


def test_symmetric_xlim_empty_returns_none():
    assert symmetric_xlim([]) is None


def test_symmetric_xlim_ignores_nonfinite():
    low, high = symmetric_xlim([np.array([np.nan, 0.4, np.inf * 0])])
    # only the finite 0.4 should count
    assert high == pytest.approx(0.4 * 1.05)


def test_symmetric_xlim_custom_pad():
    low, high = symmetric_xlim([np.array([1.0])], pad_frac=0.0)
    assert (low, high) == pytest.approx((-1.0, 1.0))


# ---------------------------------------------------------------------------
# exact_energy_ylim
# ---------------------------------------------------------------------------

def test_exact_energy_ylim_matches_data_range():
    energy = np.array([-15.02718, -12.0, 0.0, 8.01234])
    assert exact_energy_ylim(energy) == pytest.approx((-15.02718, 8.01234))


def test_exact_energy_ylim_no_padding():
    """Unlike symmetric_xlim, the energy axis gets no padding at all —
    it must match lobsterin's COHPstartEnergy/COHPendEnergy exactly."""
    energy = np.linspace(-15.0, 8.0, 401)
    low, high = exact_energy_ylim(energy)
    assert low == pytest.approx(-15.0)
    assert high == pytest.approx(8.0)


def test_exact_energy_ylim_empty_returns_none():
    assert exact_energy_ylim(np.array([])) is None
    assert exact_energy_ylim(None) is None
