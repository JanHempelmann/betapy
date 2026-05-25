"""
Shared constants for betapy.

Centralised here so that the same value is never duplicated across modules.
"""

# Decimal places used when rounding pFC values for uniqueness detection.
# Used in projection.unique_pfcs() and in the GUI pFC viewer deduplication.
PFC_ROUNDING_DECIMALS = 5

# Conversion factor: 1 eV/Å² = 16.02176634 N/m
# (1 eV = 1.602176634e-19 J exact; 1 Å² = 1e-20 m²)
EV_ANG2_TO_N_M = 16.02176634

# Canonical unit keys used in settings, QSettings, and YAML.
UNIT_EV   = 'eV/Ang2'   # default internal key
UNIT_NM   = 'N/m'
# Pretty labels for display (axis labels, tooltips, combo boxes)
UNIT_LABEL = {UNIT_EV: 'eV/Å²', UNIT_NM: 'N/m'}

# Transition-metal elements for which same-species bonds are hidden by default
# in the 3D structure view.  User can re-enable them via the Bonds panel.
SAME_SPECIES_METALS = frozenset({
    'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
})
