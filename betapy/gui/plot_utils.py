"""
Shared axis conventions and export helpers for LOBSTER COHP/COOP/COBI curve plots.

Standard convention: the value axis (COHP/COOP/COBI) is symmetric about
zero so bonding and antibonding character get equal visual weight, and the
energy axis is pinned exactly to the data's own range (which matches
lobsterin's COHPstartEnergy/COHPendEnergy) rather than matplotlib's default
autoscale margin.
"""

import numpy as np
import matplotlib

# height / width for a "standard", relatively narrow portrait figure when
# aspect-locking on export. The golden ratio has no special claim on LOBSTER
# plots specifically — it's simply the most broadly recognized "standard"
# aesthetic ratio to default to when none is otherwise specified; adjust if
# your journal/group has a different convention.
GOLDEN_RATIO = 1.618


def symmetric_xlim(curves, pad_frac=0.05):
    """
    Symmetric x-axis limits covering every array in *curves*.

    Parameters
    ----------
    curves   : iterable of 1-D array-like curve values. None entries and
               entries with no finite non-zero values are ignored.
    pad_frac : float, fractional padding beyond the max magnitude, so lines
               at the extreme aren't flush against the axis edge.

    Returns
    -------
    (low, high), or None if no finite non-zero data was found across all
    curves — callers should leave the axis on matplotlib's default
    autoscale in that case (e.g. an all-zero or empty curve).
    """
    maxabs = 0.0
    for c in curves:
        if c is None:
            continue
        arr = np.asarray(c)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            maxabs = max(maxabs, float(np.max(np.abs(finite))))
    if maxabs <= 0:
        return None
    m = maxabs * (1.0 + pad_frac)
    return (-m, m)


def exact_energy_ylim(energy):
    """
    Energy-axis limits pinned exactly to *energy*'s range — no autoscale
    padding, so the plotted window matches lobsterin's requested
    COHPstartEnergy/COHPendEnergy.

    Returns (low, high), or None if *energy* is empty/None.
    """
    if energy is None or len(energy) == 0:
        return None
    return (float(np.min(energy)), float(np.max(energy)))


def export_figure(figure, path, lock_aspect=False, width_in=3.5, aspect=GOLDEN_RATIO):
    """
    Save *figure* to *path* as a publication-ready vector graphic.

    Fonts are kept as editable text objects rather than outlined shapes, so
    the exported SVG/PDF opens with live, selectable text in Illustrator:
    svg.fonttype='none' keeps SVG <text> elements as text instead of
    converting glyphs to <path> outlines; pdf.fonttype=42 embeds actual
    TrueType outlines instead of matplotlib's default Type 3 (bitmap-like,
    not real text). This only affects vector formats — PNG is unaffected.

    Parameters
    ----------
    figure      : matplotlib Figure to save
    path        : output path (extension determines format)
    lock_aspect : if True, temporarily resizes the figure to a fixed narrow
                  portrait aspect ratio before saving, then restores its
                  original size — an embedded GUI canvas's on-screen size is
                  unaffected, only the saved file's dimensions change.
    width_in    : figure width in inches when lock_aspect is True
    aspect      : height/width ratio when lock_aspect is True (default
                  GOLDEN_RATIO)
    """
    orig_size = figure.get_size_inches().copy()
    if lock_aspect:
        figure.set_size_inches(width_in, width_in * aspect)
        figure.tight_layout()

    try:
        with matplotlib.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
            figure.savefig(path)
    finally:
        if lock_aspect:
            figure.set_size_inches(*orig_size)
            figure.tight_layout()
