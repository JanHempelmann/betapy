"""
Tests for betapy.gui.plot_utils

Run with:  python -m pytest tests/
"""

import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

import numpy as np
import pytest

from betapy.gui.plot_utils import (
    symmetric_xlim, exact_energy_ylim, export_figure, GOLDEN_RATIO,
)


def _make_figure():
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_xlabel('COBI')
    return fig


def _svg_size_pt(svg_text):
    """Parse the top-level width/height (in pt) from an SVG's <svg> tag."""
    w = float(re.search(r'width="([\d.]+)pt"', svg_text).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', svg_text).group(1))
    return w, h


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


# ---------------------------------------------------------------------------
# export_figure
# ---------------------------------------------------------------------------

def test_export_figure_keeps_text_editable_in_svg():
    """
    matplotlib's SVG default outlines every glyph to a <path> — Illustrator
    then sees shapes, not text. export_figure must override this so the
    saved file has real, editable <text> elements instead.
    """
    fig = _make_figure()
    path = tempfile.mktemp(suffix='.svg')
    export_figure(fig, path)
    content = Path(path).read_text()
    assert '<text' in content


def test_export_figure_default_svg_would_have_no_text():
    """Sanity check on the assumption above: matplotlib's un-overridden
    default really does omit <text> elements entirely (all-outlined)."""
    fig = _make_figure()
    path = tempfile.mktemp(suffix='.svg')
    fig.savefig(path)   # no export_figure — default rcParams
    content = Path(path).read_text()
    assert '<text' not in content


def test_export_figure_lock_aspect_produces_golden_ratio_file():
    fig = _make_figure()
    path = tempfile.mktemp(suffix='.svg')
    export_figure(fig, path, lock_aspect=True, width_in=3.5)
    w, h = _svg_size_pt(Path(path).read_text())
    assert h / w == pytest.approx(GOLDEN_RATIO, rel=1e-3)


def test_export_figure_lock_aspect_restores_original_figure_size():
    fig = _make_figure()
    orig = tuple(fig.get_size_inches())
    export_figure(fig, tempfile.mktemp(suffix='.svg'), lock_aspect=True)
    assert tuple(fig.get_size_inches()) == pytest.approx(orig)


def test_export_figure_no_lock_keeps_current_figure_shape():
    fig = _make_figure()
    fig.set_size_inches(4.0, 3.0)   # not golden-ratio
    path = tempfile.mktemp(suffix='.svg')
    export_figure(fig, path, lock_aspect=False)
    w, h = _svg_size_pt(Path(path).read_text())
    assert w / h == pytest.approx(4.0 / 3.0, rel=1e-3)
