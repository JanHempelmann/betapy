"""
Shared constants for betapy.

Centralised here so that the same value is never duplicated across modules.
"""

# Decimal places used when rounding pFC values for uniqueness detection.
# Used in projection.unique_pfcs() and in the GUI pFC viewer deduplication.
PFC_ROUNDING_DECIMALS = 5

# Transition-metal elements for which same-species bonds are hidden by default
# in the 3D structure view.  User can re-enable them via the Bonds panel.
SAME_SPECIES_METALS = frozenset({
    'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
})
