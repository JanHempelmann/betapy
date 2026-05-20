"""
Crystal structure representation for betapy.

The Supercell class owns all geometry: lattice vectors, fractional positions,
and the chemical species list. Distance calculations with periodic boundary
conditions live here so they don't get duplicated across the rest of the code.
"""

import numpy as np


class Supercell:
    """
    Represents a Phonopy supercell.

    Attributes
    ----------
    skal         : float, scaling factor from SPOSCAR
    lattice      : (3, 3) ndarray, lattice vectors in rows [Angstrom]
    chem_symbols : list of str, chemical symbols in order
    chem_atoms   : list of int, atom counts per species
    positions    : (N, 3) ndarray, fractional coordinates
    """

    def __init__(self, sposcar_dict):
        """
        Build a Supercell from the dict returned by io.read_SPOSCAR.
        """
        self.skal = sposcar_dict['skal']
        self.lattice = np.array(sposcar_dict['lattice']) * self.skal
        self.chem_symbols = sposcar_dict['chem_symbols']
        self.chem_atoms = sposcar_dict['chem_atoms']
        self.positions = np.array(sposcar_dict['positions'])

        # Pre-build a flat list of species labels, one per atom.
        # e.g. ['Li','Li','V','V','O','O','O'] for a small cell.
        # This makes atom-type lookup O(1) instead of O(n_species).
        self._species = []
        for symbol, count in zip(self.chem_symbols, self.chem_atoms):
            self._species.extend([symbol] * count)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_atoms(self):
        return len(self.positions)

    def species(self, atom_index_1based):
        """Return the chemical symbol for a 1-based atom index."""
        return self._species[atom_index_1based - 1]

    # ------------------------------------------------------------------
    # Distance calculations with periodic boundary conditions
    # ------------------------------------------------------------------

    def frac_diff(self, frac_a, frac_b):
        """
        Fractional displacement from point a to point b, wrapped to (-0.5, 0.5].

        Parameters
        ----------
        frac_a, frac_b : array-like of length 3, fractional coordinates

        Returns
        -------
        ndarray of shape (3,)
        """
        diff = np.asarray(frac_b) - np.asarray(frac_a)
        # Minimum image convention
        diff -= np.floor(diff + 0.5)
        return diff

    def cart_diff(self, frac_a, frac_b):
        """
        Cartesian displacement from point a to point b [Angstrom], PBC-wrapped.
        """
        return self.frac_diff(frac_a, frac_b) @ self.lattice

    def distance(self, frac_a, frac_b):
        """
        Scalar distance between two fractional-coordinate points [Angstrom].
        """
        return float(np.linalg.norm(self.cart_diff(frac_a, frac_b)))

    def atom_distance(self, idx_a_1based, idx_b_1based):
        """
        Distance between two atoms given their 1-based indices.
        """
        pos_a = self.positions[idx_a_1based - 1]
        pos_b = self.positions[idx_b_1based - 1]
        return self.distance(pos_a, pos_b)

    def distance_to_point(self, atom_idx_1based, frac_point):
        """
        Distance from an atom (1-based index) to an arbitrary fractional point.
        """
        pos = self.positions[atom_idx_1based - 1]
        return self.distance(pos, frac_point)

    def cart_vector_to_point(self, atom_idx_1based, frac_point):
        """
        Cartesian vector FROM atom TO frac_point, PBC-wrapped [Angstrom].
        """
        pos = self.positions[atom_idx_1based - 1]
        return self.cart_diff(pos, frac_point)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def atoms_within(self, frac_point, cutoff_angstrom):
        """
        Return a list of (atom_index_1based, distance) for all atoms
        within cutoff_angstrom of frac_point.
        """
        results = []
        for i, pos in enumerate(self.positions):
            d = self.distance(pos, frac_point)
            if d <= cutoff_angstrom:
                results.append((i + 1, d))
        results.sort(key=lambda x: x[1])
        return results

    def __repr__(self):
        formula = ''.join(
            f'{s}{n}' for s, n in zip(self.chem_symbols, self.chem_atoms)
        )
        return f'Supercell({formula}, {self.n_atoms} atoms)'
