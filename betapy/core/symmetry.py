"""
Symmetry-based expansion of bulk pFC pair data.

Uses spglib (a phonopy dependency) to fill in force-constant records for atom
pairs that exist geometrically in the supercell but are absent from the
FORCE_CONSTANTS data.  This happens whenever a *compact* FORCE_CONSTANTS file
is used: phonopy stores pairs only for one representative atom per Wyckoff
orbit; pairs involving non-representative atoms are recovered by symmetry.

Key physics: phi_iso, mean_pfc, rms_pfc, phi_l and phi_t are all invariant
under crystal symmetry operations (they are scalars derived from the
eigenvalues or bond-aligned projections of the 3×3 FC tensor, which transform
trivially under orthogonal rotations).  We therefore copy them directly from
the equivalent pair without any tensor rotation.  Geometry (distance,
direction) is taken from the actual SPOSCAR positions and is always exact.

Performance note
----------------
Compact FC files for large supercells can have very many symmetry operations
(e.g. GeTe 512-atom supercell: 12 288 operations, 2 Wyckoff orbits).
The expansion is built on three O(n_atoms) or O(n_pairs) operations rather
than the naive O(n_ops × n_atoms) atom-map table:

  1.  equivalent_atoms[]  from spglib  →  which representative atom each atom
      belongs to, without any search.
  2.  one_op_per_atom     computed with early exit from the op loop; terminates
      after ≈ n_ops / orbit_size iterations (48 for GeTe, not 12 288).
  3.  pos_to_idx          dict keyed on rounded fractional coordinates  →
      O(1) lookup for where each atom's image lands under an operation.
"""

from __future__ import annotations

import numpy as np

try:
    import spglib as _spglib
    _HAVE_SPGLIB = True
except ImportError:
    _HAVE_SPGLIB = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_spglib_cell(supercell):
    """Return (lattice, positions, numbers) tuple expected by spglib."""
    species_list = [supercell.species(i + 1) for i in range(supercell.n_atoms)]
    unique_sp    = sorted(set(species_list))
    sp_to_int    = {sp: k + 1 for k, sp in enumerate(unique_sp)}
    numbers      = np.array([sp_to_int[sp] for sp in species_list], dtype=int)
    return supercell.lattice, supercell.positions, numbers


def _find_one_op_per_atom(fracs_w, eq_atoms, rotations, translations,
                           n_ops, tol=1e-3):
    """
    For each atom i find ONE operation that maps it to eq_atoms[i].

    Iterates over operations with early termination: as soon as all atoms have
    been assigned an operation the loop exits.  For systems with few distinct
    Wyckoff orbits (e.g. GeTe has 2) this finishes after ≈ n_ops / orbit_size
    iterations rather than the full n_ops.

    Returns
    -------
    one_op : (n_atoms,) int32   — operation index for each atom; -1 if none
             was found (structural inconsistency, triggers fallback).
    """
    n      = len(fracs_w)
    one_op = np.full(n, -1, dtype=np.int32)

    for op_idx in range(n_ops):
        unassigned = np.where(one_op < 0)[0]
        if len(unassigned) == 0:
            break
        R, t = rotations[op_idx], translations[op_idx]
        mapped_w = (fracs_w @ R.T + t) % 1.0   # (N, 3)
        targets  = fracs_w[eq_atoms[unassigned]] # where they should land
        diff     = mapped_w[unassigned] - targets
        diff    -= np.floor(diff + 0.5)          # minimum image
        dist     = np.linalg.norm(diff, axis=1)
        one_op[unassigned[dist < tol]] = op_idx

    return one_op


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_by_symmetry(bulk_results, supercell, bond_cutoff=4.0):
    """
    Fill in missing force-constant pair records using crystal symmetry.

    For each geometrically close atom pair (within *bond_cutoff* Å) that is
    absent from *bulk_results*, the function locates the symmetry-equivalent
    reference pair and copies its scalar FC quantities (mean_pfc, rms_pfc,
    phi_l, phi_t).  Geometry (distance, direction) is computed from SPOSCAR
    positions.  phi_iso is not copied because compute_badger_quantities() adds
    it downstream from phi_l and phi_t.

    Algorithm
    ---------
    For a missing pair (i, j):
      • eq_i  = equivalent_atoms[i]  — the Wyckoff representative for atom i.
      • op    = one_op_per_atom[i]   — the operation that maps i → eq_i.
      • j'    = applying op to atom j's position, looked up in pos_to_idx.
      • The reference pair (eq_i, j') is in fc_index because compact FC files
        store (ref_atom, ALL j) for every reference atom.

    Falls back to returning *bulk_results* unchanged when:
      • spglib is not importable,
      • space-group determination fails, or
      • any atom cannot be mapped to its representative (structural edge case).

    Parameters
    ----------
    bulk_results : list[dict]   — as returned by compute_bulk_pfcs().
    supercell    : betapy.core.structure.Supercell
    bond_cutoff  : float, Å — maximum distance to consider.  Should match the
                   bond_cutoff used in find_chains (default 4.0 Å).

    Returns
    -------
    list[dict] — bulk_results + synthetic records, or bulk_results unchanged.
    """
    if not _HAVE_SPGLIB:
        return bulk_results

    # ------------------------------------------------------------------
    # 1.  Symmetry dataset
    # ------------------------------------------------------------------
    cell    = _build_spglib_cell(supercell)
    dataset = _spglib.get_symmetry_dataset(cell)
    if dataset is None:
        return bulk_results

    # Support both the new attribute API (spglib ≥ 2.0) and the legacy dict API.
    try:
        rotations = np.array(dataset.rotations, dtype=float)
        translations = np.array(dataset.translations)
        eq_atoms     = np.array(dataset.equivalent_atoms)
    except AttributeError:
        rotations    = dataset['rotations'].astype(float)
        translations = dataset['translations']
        eq_atoms     = dataset['equivalent_atoms']
    n_ops        = len(rotations)

    # ------------------------------------------------------------------
    # 2.  Index existing pairs  (1-based atom indices)
    # ------------------------------------------------------------------
    fc_index: dict[tuple, dict] = {}
    for r in bulk_results:
        k = (min(r['atom1_idx'], r['atom2_idx']),
             max(r['atom1_idx'], r['atom2_idx']))
        if k not in fc_index:
            fc_index[k] = r

    # ------------------------------------------------------------------
    # 3.  Build O(1) position lookup and find one op per atom
    # ------------------------------------------------------------------
    n       = supercell.n_atoms
    fracs   = supercell.positions
    fracs_w = fracs % 1.0          # wrap all positions to [0, 1)

    # Round to 4 decimal places (~0.01% of a fractional unit, robust for
    # crystallographic positions which are always simple fractions).
    _rnd = 4
    pos_to_idx: dict[tuple, int] = {
        (round(float(fracs_w[i, 0]), _rnd),
         round(float(fracs_w[i, 1]), _rnd),
         round(float(fracs_w[i, 2]), _rnd)): i
        for i in range(n)
    }

    one_op = _find_one_op_per_atom(
        fracs_w, eq_atoms, rotations, translations, n_ops)
    if np.any(one_op < 0):
        return bulk_results  # structural edge case — safe fallback

    # ------------------------------------------------------------------
    # 4.  Enumerate geometric pairs within bond_cutoff; expand missing ones
    # ------------------------------------------------------------------
    latt      = supercell.lattice
    synthetic: list[dict] = []
    seen_new:  set[tuple] = set()

    _CHUNK = 128
    for i0 in range(0, n, _CHUNK):
        i1   = min(i0 + _CHUNK, n)
        raw  = fracs[np.newaxis, :, :] - fracs[i0:i1, np.newaxis, :]
        diff = raw - np.floor(raw + 0.5)
        cart = diff @ latt
        dist = np.linalg.norm(cart, axis=-1)   # (chunk, N)

        ci_arr, j_arr = np.where((dist > 1e-6) & (dist <= bond_cutoff))
        for ci, j in zip(ci_arr.tolist(), j_arr.tolist()):
            i = i0 + ci
            if i >= j:
                continue                       # each pair once
            idx_i, idx_j = i + 1, j + 1       # 1-based
            key = (idx_i, idx_j)
            if key in fc_index or key in seen_new:
                continue                       # already have FC data

            # Apply the precomputed op for atom i (maps i → eq_i) to atom j.
            op_idx = int(one_op[i])
            R, t   = rotations[op_idx], translations[op_idx]
            jp_w   = (fracs_w[j] @ R.T + t) % 1.0
            jp_key = (round(float(jp_w[0]), _rnd),
                      round(float(jp_w[1]), _rnd),
                      round(float(jp_w[2]), _rnd))
            jp = pos_to_idx.get(jp_key, -1)
            if jp < 0:
                continue                       # rounding edge case — skip

            eq_i   = int(eq_atoms[i])
            eq_key = (min(eq_i + 1, jp + 1), max(eq_i + 1, jp + 1))
            if eq_key == key:
                continue                       # would just copy itself

            eq_rec = fc_index.get(eq_key)
            if eq_rec is None:
                continue                       # not in compact FC — skip

            d      = float(dist[ci, j])
            dir_ij = (cart[ci, j] / d).tolist()
            seen_new.add(key)
            synthetic.append({
                'atom1_idx': idx_i,
                'atom2_idx': idx_j,
                'species1':  supercell.species(idx_i),
                'species2':  supercell.species(idx_j),
                'distance':  d,
                'direction': dir_ij,
                'mean_pfc':  eq_rec.get('mean_pfc', float('nan')),
                'rms_pfc':   eq_rec.get('rms_pfc',  float('nan')),
                'phi_l':     eq_rec.get('phi_l',    float('nan')),
                'phi_t':     eq_rec.get('phi_t',    float('nan')),
            })

    return bulk_results + synthetic
