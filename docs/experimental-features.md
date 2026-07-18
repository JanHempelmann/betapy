# Experimental Features

These features implement methods that are fully functional and actively used in ongoing research, but whose algorithmic details have not yet been formally published — they are tied to forthcoming manuscripts. For multicenter bond detection, see the dedicated guide: [docs/multicenter.md](multicenter.md).

---

## Reference-site projection

Projects force constants around any fractional coordinate in the cell rather than around an atom — a vacancy, an interstitial, or any other point of interest. The site does not need to coincide with an atom.

**CLI:**

```bash
betapy --sposcar SPOSCAR --fc FORCE_CONSTANTS \
       --refsite 5.0 --refpos REFPOS --store
```

`REFPOS` lists fractional coordinates in a VASP-POSCAR-like format (see the main README's Inputs section for the exact layout).

**GUI:** the **Ref. Site Projection** tab appears automatically when a `REFPOS` file is found in the working directory, or can be opened via the **"+"** tab menu. Snap the site to an atom by clicking in the 3D view or typing in the searchable atom list; analysis runs in a background thread so the GUI stays responsive. Results are shown as a scatter plot and sortable table, and can be exported to CSV or REFPOS directly from the tab.

---

## Stiffness-shift parameter

Compares the sum of reference-site projected force constants between two structures — typically a deintercalated structure (projection around a vacancy) and an intercalated structure (projection around the occupied site, excluding the site-occupying atom).

Atom pairs are matched across the two structures by a scalar Cartesian fingerprint — (distance from refsite to atom1, distance from refsite to atom2, bond length) — using the Hungarian algorithm per species group with a 0.3 Å tolerance, falling back to a distance-ordered equal-count comparison if matching fails outright. This makes the comparison robust to index reordering, cell-origin shifts, and structural distortions between separate VASP calculations.

**Settings file:**

```yaml
stiffness_shift:
  structure_a:
    sposcar: path/to/deintercalated/SPOSCAR
    force_constants: path/to/deintercalated/FORCE_CONSTANTS
    refpos: path/to/REFPOS
  structure_b:
    sposcar: path/to/intercalated/SPOSCAR
    force_constants: path/to/intercalated/FORCE_CONSTANTS
    refpos: path/to/REFPOS
  cutoff: 5.0
  min_site_dist: 0.1    # excludes site-occupying atom in intercalated structure
```

**GUI:** the **Stiffness Shift** tab loads both structures, lets you configure the reference site and cutoff, and computes the comparison in a background thread (progress bar, non-blocking).

---

## Badger analysis

Decomposes each pFC into a rotationally-invariant isotropic force constant and a dimensionless anisotropy factor, in order to remove orientation-dependent scatter from the conventional Badger relation (Φ<sup>−1/3</sup> = a·r + b).

- **Conventional** Φ_p — the existing bond-direction projection
- **Isotropic** Φ_iso = (|φ_L| + 2·|φ_T|) / 3 — the mean absolute eigenvalue of the projected force-constant matrix; rotationally invariant, so it collapses the multiple parallel Badger lines that appear in covalent systems (the "gleichergestalt" splitting)
- **Anisotropy factor** ξ = Φ_p / Φ_iso — encodes which conventional Badger line a pair belongs to; the conventional-plot scatter decomposes exactly as Φ_p<sup>−1/3</sup> = Φ_iso<sup>−1/3</sup> · ξ<sup>−1/3</sup>
- **η** = |φ_L / φ_T| — the longitudinal-to-transverse anisotropy of the actual FC matrix for a pair, independent of projection direction; large (≫1) for covalent bonds, ~1–2 for ionic or isotropic interactions

An opt-in, sign-preserving variant (Φ_iso,signed = (φ_L + 2·φ_T) / 3 = Tr(Φ)/3) is available for advanced users who want to inspect the sign-cancellation that the unsigned Φ_iso is designed to mask, rather than have it hidden.

**GUI:** the **Badger Analysis** tab (via the **"+"** menu) shows a 2×2 layout — conventional Φ_p<sup>−1/3</sup> vs. *r* (top-left), isotropic Φ_iso<sup>−1/3</sup> vs. *r* (top-right), a conventional scatter coloured by ξ (bottom-left), and a linked 3D structure view (bottom-right) that highlights the corresponding bond when a point is clicked in the ξ scatter.
