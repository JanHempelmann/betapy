"""
Shared axis conventions for LOBSTER COHP/COOP/COBI curve plots.

Standard convention: the value axis (COHP/COOP/COBI) is symmetric about
zero so bonding and antibonding character get equal visual weight, and the
energy axis is pinned exactly to the data's own range (which matches
lobsterin's COHPstartEnergy/COHPendEnergy) rather than matplotlib's default
autoscale margin.
"""

import numpy as np


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
