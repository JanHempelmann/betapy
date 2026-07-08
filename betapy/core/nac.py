"""
Non-analytical term correction (NAC) for real-space force constants.

Computes Phi_short = Phi_DFT - Phi_dd using phonopy's Gonze-Lee (GL)
method, which evaluates the dipole-dipole (long-range Coulomb) correction
via a proper Ewald summation.  Removing Phi_dd leaves the short-range
bonding physics that betapy's pFC projection is designed to capture.

Phonopy's DynamicalMatrixGL computes the correct real-space short-range
force constants internally via:
  1. Full Ewald sum (G-sum) for D_dd(q) at all commensurate q-points.
  2. Inverse FT of D_sr(q) = D_DFT(q) - D_dd(q) back to real space.

The result is returned directly as dm.short_range_force_constants.

Requires phonopy as an optional dependency.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


# e^2 / (4 pi eps_0) in eV * Angstrom — the standard NAC unit conversion
# factor for VASP calculations (Born charges in |e|, distances in Angstrom).
_NAC_FACTOR_EV_ANG = 14.399651725922268

_PHONOPY_YAML_NAMES = ('phonopy_disp.yaml', 'phonopy.yaml')


def _require_phonopy():
    try:
        import phonopy
        return phonopy
    except ImportError:
        raise ImportError(
            'phonopy is required for NAC correction. '
            'Install it with: pip install phonopy'
        )


def find_phonopy_yaml(directory) -> Optional[Path]:
    """
    Look for phonopy_disp.yaml or phonopy.yaml in *directory*.
    Returns the first match, or None if neither is present.
    """
    for name in _PHONOPY_YAML_NAMES:
        p = Path(directory) / name
        if p.exists():
            return p
    return None


def compute_short_range_fc(
    phonopy_yaml,
    force_constants_path,
    born_dir,
) -> np.ndarray:
    """
    Return Phi_short = Phi_DFT - Phi_dd for a phonopy supercell with NAC.

    Uses phonopy's Gonze-Lee (GL) method: the dipole-dipole contribution
    is evaluated via a full Ewald sum at all commensurate q-points and
    transformed back to real space by DynmatToForceConstants.  This is
    phonopy's own short_range_force_constants computation.

    Parameters
    ----------
    phonopy_yaml : path-like
        Path to phonopy_disp.yaml (or phonopy.yaml).
    force_constants_path : path-like
        Path to the FORCE_CONSTANTS file (compact format, n_prim x n_sc).
    born_dir : path-like
        Directory containing both BORN (Born effective charges and dielectric
        tensor) and POSCAR (primitive cell used for the DFPT Born run).

    Returns
    -------
    phi_short : np.ndarray, shape (n_prim, n_sc, 3, 3)
        Short-range force constants with the dipole-dipole tail removed.
        Same shape and indexing convention as phonopy's compact FC array.
    """
    phonopy_mod = _require_phonopy()
    from phonopy.file_IO import parse_BORN
    from phonopy.interface.vasp import read_vasp

    born_dir = Path(born_dir)
    born_path = born_dir / 'BORN'
    poscar_path = born_dir / 'POSCAR'

    for p in (born_path, poscar_path):
        if not p.exists():
            raise FileNotFoundError(
                f'NAC correction requires {p.name} in born_dir ({born_dir})'
            )

    ph = phonopy_mod.load(
        phonopy_yaml=str(phonopy_yaml),
        force_constants_filename=str(force_constants_path),
        primitive_matrix='auto',
        is_compact_fc=True,
        symmetrize_fc=False,
        log_level=0,
    )

    prim_cell_for_born = read_vasp(str(poscar_path))
    nac_params = parse_BORN(prim_cell_for_born, filename=str(born_path))
    nac_params['factor'] = _NAC_FACTOR_EV_ANG
    nac_params['method'] = 'gonze'  # Gonze-Lee: full Ewald sum for Phi_dd
    ph.nac_params = nac_params

    dm = ph.dynamical_matrix  # DynamicalMatrixGL
    dm.make_Gonze_nac_dataset()

    phi_short = np.asarray(dm.short_range_force_constants)
    if phi_short is None or phi_short.size == 0:
        raise RuntimeError(
            'phonopy GL method did not produce short_range_force_constants. '
            'Check that phonopy >= 2.0 is installed and the BORN file is valid.'
        )
    return phi_short


def apply_to_betapy_fc_data(fc_data: dict, phi_short: np.ndarray) -> None:
    """
    Replace force matrices in a betapy FC dict with NAC-corrected values.

    Modifies *fc_data* in-place.  Only compact FC format is supported
    (header n_prim x n_sc, where n_prim < n_sc).

    Parameters
    ----------
    fc_data : dict
        As returned by betapy.core.io.read_FORCE_CONSTANTS.
    phi_short : np.ndarray, shape (n_prim, n_sc, 3, 3)
        Output of compute_short_range_fc().
    """
    n_prim, n_sc = fc_data['nats']
    if n_prim >= n_sc:
        raise ValueError(
            'apply_to_betapy_fc_data requires compact FORCE_CONSTANTS '
            f'(n_prim < n_sc), but got header [{n_prim}, {n_sc}].'
        )
    # In phonopy's compact FC file, the first column holds the 1-based
    # SUPERCELL index of each primitive atom (e.g. 1 and 344 for NaBr),
    # not the 1-based primitive-cell index (1..n_prim). Build the mapping.
    first_vals = sorted({pair[0] for pair in fc_data['atomic_pairs']})
    first_to_kappa = {sc_idx: kappa for kappa, sc_idx in enumerate(first_vals)}

    for k, (pair, _) in enumerate(
        zip(fc_data['atomic_pairs'], fc_data['force_matrices'])
    ):
        i, j = pair   # i = 1-based sc index of prim atom; j = 1-based sc atom
        kappa = first_to_kappa[i]
        fc_data['force_matrices'][k] = phi_short[kappa, j - 1].tolist()
