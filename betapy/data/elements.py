"""
Element display properties for betapy structure rendering.

Covalent radii from:
    Cordero et al., Dalton Trans., 2008, 2832.
    DOI: 10.1039/b801115j
    (Alvarez 2008 dataset via qcelemental)

Colours follow the CPK/Jmol convention with a few adjustments
for visibility against a white background.
"""

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

# Default fallback radius for unknown elements
DEFAULT_COVALENT_RADIUS = 1.50

# CPK/Jmol-inspired colours as (R, G, B) floats in [0, 1]
# Adjusted for visibility on white background.
ELEMENT_COLOURS = {
    # Non-metals
    'H':  (0.90, 0.90, 0.90),   # near-white / light grey
    'C':  (0.20, 0.20, 0.20),   # dark grey (black on white is too harsh)
    'N':  (0.13, 0.37, 0.80),   # blue
    'O':  (0.85, 0.15, 0.15),   # red
    'F':  (0.56, 0.82, 0.31),   # green
    'P':  (1.00, 0.50, 0.00),   # orange
    'S':  (0.90, 0.78, 0.00),   # yellow
    'Cl': (0.12, 0.94, 0.12),   # bright green
    'Br': (0.65, 0.16, 0.16),   # dark red
    'I':  (0.58, 0.00, 0.58),   # purple
    # Alkali / alkaline earth
    'Li': (0.80, 0.50, 1.00),   # violet
    'Na': (0.67, 0.36, 0.95),   # medium violet
    'K':  (0.56, 0.25, 0.83),   # dark violet
    'Mg': (0.54, 1.00, 0.00),   # bright green
    'Ca': (0.24, 1.00, 0.00),   # green
    # Transition metals — range from steel blue to warm grey
    'Ti': (0.58, 0.58, 0.67),
    'V':  (0.65, 0.65, 0.67),
    'Cr': (0.54, 0.60, 0.78),
    'Mn': (0.61, 0.48, 0.78),
    'Fe': (0.88, 0.40, 0.20),   # rusty orange
    'Co': (0.94, 0.56, 0.63),
    'Ni': (0.31, 0.82, 0.31),
    'Cu': (0.78, 0.50, 0.20),   # copper
    'Zn': (0.49, 0.50, 0.69),
    'Zr': (0.58, 0.88, 0.88),
    'Nb': (0.45, 0.76, 0.79),
    'Mo': (0.33, 0.71, 0.71),
    'Ru': (0.24, 0.62, 0.62),
    'Rh': (0.04, 0.49, 0.55),
    'Pd': (0.00, 0.41, 0.52),
    'Ag': (0.75, 0.75, 0.75),   # silver
    'Pt': (0.82, 0.82, 0.88),
    'Au': (1.00, 0.82, 0.14),   # gold
    'Hg': (0.72, 0.72, 0.82),
    # Default for everything else: medium grey
}

DEFAULT_COLOUR = (0.50, 0.50, 0.50)

# Display radii for sphere rendering (not covalent radii — tuned for aesthetics)
# These are smaller than covalent radii to avoid overlap in dense structures.
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
    # Lanthanides (representative)
    'La': 0.52,
    'Ce': 0.52,
    'Nd': 0.51,
    'Sm': 0.51,
    'Eu': 0.51,
    'Gd': 0.50,
    'Tb': 0.50,
    'Dy': 0.50,
    'Ho': 0.50,
    'Er': 0.49,
    'Yb': 0.49,
    'Lu': 0.49,
}
DEFAULT_DISPLAY_RADIUS = 0.42   # fallback for elements not listed above


def covalent_radius(symbol):
    """Return covalent radius in Angstrom for element symbol."""
    return COVALENT_RADII.get(symbol, DEFAULT_COVALENT_RADIUS)


def element_colour(symbol):
    """Return (R, G, B) display colour for element symbol."""
    return ELEMENT_COLOURS.get(symbol, DEFAULT_COLOUR)


def display_radius(symbol):
    """Return sphere display radius in Angstrom for element symbol."""
    return DISPLAY_RADII.get(symbol, DEFAULT_DISPLAY_RADIUS)
