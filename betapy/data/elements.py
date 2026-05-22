"""
Element display properties for betapy structure rendering.

Covalent radii from:
    Cordero et al., Dalton Trans., 2008, 2832.
    DOI: 10.1039/b801115j
    (Alvarez 2008 dataset via qcelemental)

Two colour presets are provided:

  'Jmol'  — official Jmol/CPK extended table (jmol.sourceforge.net).
             De facto standard for most molecular viewers and databases.
             H and C are adjusted for visibility on a white background
             (H: white → light grey; C: medium grey → dark grey).

  'VESTA' — approximate VESTA colours (jp-minerals.org/vesta).
             Transition metals are identical to Jmol (the two programs agree
             for most metals).  Main differences: C is medium grey, S is amber,
             lanthanides use a more distinctive series.
             Recommended when preparing figures for a crystallography audience.
"""

# ---------------------------------------------------------------------------
# Covalent radii
# ---------------------------------------------------------------------------

# Covalent radii in Angstrom (Alvarez 2008)
COVALENT_RADII = {
    'H':  0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B':  0.84, 'C':  0.76, 'N':  0.71,
    'O':  0.66, 'F':  0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P':  1.07,
    'S':  1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K':  2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V':  1.53,
    'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50, 'Ni': 1.24,
    'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
    'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y':  1.90, 'Zr': 1.75, 'Nb': 1.64,
    'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
    'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39,
    'Te': 1.38, 'I':  1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03,
    'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90,
    'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W':  1.62,
    'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36,
    'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    'At': 1.50, 'Rn': 1.50,
}

DEFAULT_COVALENT_RADIUS = 1.50

# ---------------------------------------------------------------------------
# Colour presets
# ---------------------------------------------------------------------------

# Jmol/CPK extended — official values, white-background adjusted for H and C.
_JMOL = {
    # Period 1
    'H':  (0.90, 0.90, 0.90),   # Jmol white → light grey (invisible on white bg)
    'He': (0.85, 1.00, 1.00),
    # Period 2
    'Li': (0.80, 0.50, 1.00),
    'Be': (0.76, 1.00, 0.00),
    'B':  (1.00, 0.71, 0.71),
    'C':  (0.20, 0.20, 0.20),   # Jmol #909090 → darkened for contrast on white
    'N':  (0.19, 0.31, 0.97),
    'O':  (1.00, 0.05, 0.05),
    'F':  (0.56, 0.88, 0.31),
    'Ne': (0.70, 0.89, 0.96),
    # Period 3
    'Na': (0.67, 0.36, 0.95),
    'Mg': (0.54, 1.00, 0.00),
    'Al': (0.75, 0.65, 0.65),
    'Si': (0.94, 0.78, 0.63),
    'P':  (1.00, 0.50, 0.00),
    'S':  (1.00, 1.00, 0.19),
    'Cl': (0.12, 0.94, 0.12),
    'Ar': (0.50, 0.82, 0.89),
    # Period 4
    'K':  (0.56, 0.25, 0.83),
    'Ca': (0.24, 1.00, 0.00),
    'Sc': (0.90, 0.90, 0.90),
    'Ti': (0.75, 0.76, 0.78),
    'V':  (0.65, 0.65, 0.67),
    'Cr': (0.54, 0.60, 0.78),
    'Mn': (0.61, 0.48, 0.78),
    'Fe': (0.88, 0.40, 0.20),
    'Co': (0.94, 0.56, 0.63),
    'Ni': (0.31, 0.82, 0.31),
    'Cu': (0.78, 0.50, 0.20),
    'Zn': (0.49, 0.50, 0.69),
    'Ga': (0.76, 0.56, 0.56),
    'Ge': (0.40, 0.56, 0.56),
    'As': (0.74, 0.50, 0.89),
    'Se': (1.00, 0.63, 0.00),
    'Br': (0.65, 0.16, 0.16),
    'Kr': (0.36, 0.72, 0.82),
    # Period 5
    'Rb': (0.44, 0.18, 0.69),
    'Sr': (0.00, 1.00, 0.00),
    'Y':  (0.58, 1.00, 1.00),
    'Zr': (0.58, 0.88, 0.88),
    'Nb': (0.45, 0.76, 0.79),
    'Mo': (0.33, 0.71, 0.71),
    'Tc': (0.23, 0.62, 0.62),
    'Ru': (0.14, 0.56, 0.56),
    'Rh': (0.04, 0.49, 0.55),
    'Pd': (0.00, 0.41, 0.52),
    'Ag': (0.75, 0.75, 0.75),
    'Cd': (1.00, 0.85, 0.56),
    'In': (0.65, 0.46, 0.45),
    'Sn': (0.40, 0.50, 0.50),
    'Sb': (0.62, 0.39, 0.71),
    'Te': (0.83, 0.48, 0.00),
    'I':  (0.58, 0.00, 0.58),
    'Xe': (0.26, 0.62, 0.69),
    # Period 6
    'Cs': (0.34, 0.09, 0.56),
    'Ba': (0.00, 0.79, 0.00),
    # Lanthanides
    'La': (0.44, 0.83, 1.00),
    'Ce': (1.00, 1.00, 0.78),
    'Pr': (0.85, 1.00, 0.78),
    'Nd': (0.78, 1.00, 0.78),
    'Pm': (0.64, 1.00, 0.78),
    'Sm': (0.56, 1.00, 0.78),
    'Eu': (0.38, 1.00, 0.78),
    'Gd': (0.27, 1.00, 0.78),
    'Tb': (0.19, 1.00, 0.78),
    'Dy': (0.12, 1.00, 0.78),
    'Ho': (0.00, 1.00, 0.61),
    'Er': (0.00, 0.90, 0.46),
    'Tm': (0.00, 0.83, 0.32),
    'Yb': (0.00, 0.75, 0.22),
    'Lu': (0.00, 0.67, 0.14),
    # 5d transition metals
    'Hf': (0.30, 0.76, 1.00),
    'Ta': (0.30, 0.65, 1.00),
    'W':  (0.13, 0.58, 0.84),
    'Re': (0.15, 0.49, 0.67),
    'Os': (0.15, 0.40, 0.59),
    'Ir': (0.09, 0.33, 0.53),
    'Pt': (0.82, 0.82, 0.88),
    'Au': (1.00, 0.82, 0.14),
    'Hg': (0.72, 0.72, 0.82),
    'Tl': (0.65, 0.33, 0.30),
    'Pb': (0.34, 0.35, 0.38),
    'Bi': (0.62, 0.31, 0.71),
    'Po': (0.67, 0.36, 0.00),
    'At': (0.46, 0.31, 0.27),
    'Rn': (0.26, 0.51, 0.59),
}

# VESTA-inspired preset.
# Transition metals are identical to Jmol (the two programs agree).
# Main differences: C is medium grey, S is amber-yellow,
# lanthanides use a more distinctive series (avoid the pale Jmol pastels).
_VESTA = {
    **_JMOL,
    # C: lighter grey than betapy-dark, common in VESTA figures
    'C':  (0.50, 0.50, 0.50),
    # S: amber-yellow (more natural looking in oxide/sulfide structures)
    'S':  (0.90, 0.78, 0.00),
    # Sr / Ba: slightly more muted greens
    'Sr': (0.00, 0.80, 0.00),
    'Ba': (0.00, 0.70, 0.00),
    # Lanthanides — more distinguishable series:
    # warm → cool gradient across the row
    'La': (0.35, 0.83, 1.00),   # sky blue
    'Ce': (0.95, 0.90, 0.45),   # golden
    'Pr': (0.35, 0.78, 0.35),   # medium green
    'Nd': (0.40, 0.70, 0.40),   # green
    'Pm': (0.50, 0.65, 0.40),   # olive green
    'Sm': (0.60, 0.60, 0.45),   # sage
    'Eu': (0.65, 0.55, 0.60),   # mauve
    'Gd': (0.70, 0.55, 0.50),   # rose-grey
    'Tb': (0.45, 0.65, 0.55),   # sea green
    'Dy': (0.40, 0.60, 0.70),   # steel blue
    'Ho': (0.35, 0.55, 0.70),   # slate blue
    'Er': (0.55, 0.68, 0.38),   # olive-green
    'Tm': (0.45, 0.72, 0.55),   # sage green
    'Yb': (0.38, 0.52, 0.58),   # slate
    'Lu': (0.32, 0.46, 0.52),   # dark slate
}

# Active lookup table (changed by the GUI preset switcher).
# Do NOT replace this name — code everywhere calls element_colour().
ELEMENT_COLOURS = _JMOL

# Named presets exposed to the GUI.
COLOUR_PRESETS = {
    'Jmol':  _JMOL,
    'VESTA': _VESTA,
}

DEFAULT_COLOUR = (0.50, 0.50, 0.50)

# ---------------------------------------------------------------------------
# Display radii (sphere rendering, not covalent)
# ---------------------------------------------------------------------------

DISPLAY_RADII = {
    # Non-metals / main group
    'H':  0.25,
    'C':  0.35,
    'N':  0.33,
    'O':  0.30,
    'F':  0.28,
    'S':  0.38,
    'P':  0.38,
    'Cl': 0.38,
    'Br': 0.42,
    'I':  0.48,
    'Si': 0.42,
    'Al': 0.42,
    'Ge': 0.42,
    'Sn': 0.46,
    'Pb': 0.46,
    'Bi': 0.46,
    'Sb': 0.44,
    'As': 0.42,
    'Se': 0.40,
    'Te': 0.46,
    # Alkali / alkaline earth
    'Li': 0.45,
    'Na': 0.50,
    'K':  0.55,
    'Mg': 0.45,
    'Ca': 0.50,
    'Sr': 0.52,
    'Ba': 0.55,
    # 3d transition metals
    'Sc': 0.46,
    'Ti': 0.44,
    'V':  0.42,
    'Cr': 0.40,
    'Mn': 0.44,
    'Fe': 0.42,
    'Co': 0.41,
    'Ni': 0.40,
    'Cu': 0.40,
    'Zn': 0.40,
    # 4d transition metals
    'Y':  0.50,
    'Zr': 0.46,
    'Nb': 0.44,
    'Mo': 0.42,
    'Tc': 0.41,
    'Ru': 0.41,
    'Rh': 0.41,
    'Pd': 0.41,
    'Ag': 0.46,
    'Cd': 0.45,
    # 5d transition metals
    'Hf': 0.46,
    'Ta': 0.44,
    'W':  0.43,
    'Re': 0.42,
    'Os': 0.41,
    'Ir': 0.41,
    'Pt': 0.41,
    'Au': 0.41,
    'Hg': 0.40,
    # Lanthanides
    'La': 0.52,
    'Ce': 0.52,
    'Pr': 0.51,
    'Nd': 0.51,
    'Sm': 0.51,
    'Eu': 0.51,
    'Gd': 0.50,
    'Tb': 0.50,
    'Dy': 0.50,
    'Ho': 0.50,
    'Er': 0.49,
    'Tm': 0.49,
    'Yb': 0.49,
    'Lu': 0.49,
}

DEFAULT_DISPLAY_RADIUS = 0.42


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def covalent_radius(symbol):
    """Return covalent radius in Angstrom for element symbol."""
    return COVALENT_RADII.get(symbol, DEFAULT_COVALENT_RADIUS)


def element_colour(symbol):
    """Return (R, G, B) display colour for element symbol from the active preset."""
    return ELEMENT_COLOURS.get(symbol, DEFAULT_COLOUR)


def display_radius(symbol):
    """Return sphere display radius in Angstrom for element symbol."""
    return DISPLAY_RADII.get(symbol, DEFAULT_DISPLAY_RADIUS)
