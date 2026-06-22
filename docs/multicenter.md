# Experimental Feature: Multicenter Bond Detection

## Summary

Multicenter bonding describes the situation where a single bonding interaction is delocalized over three or more atoms rather than confined to a single atom pair. In IV–VI chalcogenides such as GeTe and related phase-change materials, metavalent bonding gives rise to unusually stiff long-range interatomic interactions that cannot be explained by conventional two-center bonds alone. Projected force constants (pFCs) are sensitive to this: a pair involved in a multicenter interaction exhibits a pFC that is anomalously large relative to what the distant–force constant Badger relationship would predict for a given length. The feature uses an adapted spacially averaged definition of the force constant.

betapy detects such anomalous pairs automatically and traces chains of connected anomalous pairs through the supercell to identify the full extent of a multicenter interaction. It can further transfer the detected potential multicenter chains to lobster by writing the suitable `cobiBetween` directives into a sibling directory `lobsterin` file.

> **Status:** experimental. The scientific basis — pFCs as a probe for multicenter
> bonding in phase-change materials — is established in [Hempelmann et al., *Adv.
> Mater.* 2021](https://doi.org/10.1002/adma.202100163) and [Hempelmann et al.,
> *Angew. Chem. Int. Ed.* 2022](https://doi.org/10.1002/anie.202115778). The
> automated chain-detection algorithm is described in a forthcoming manuscript.

---

## Prerequisites

Multicenter detection runs on the bulk pFC data that betapy computes from
`SPOSCAR` and `FORCE_CONSTANTS`. Run the bulk analysis first:

```bash
betapy --store
```

This writes `bulk_pFCs.csv`, which the detector reads. If you skip `--store` and run `--multicenter` in the same call, betapy computes bulk pFCs in memory and passes them directly — the CSV is not required on disk.

**Compact FORCE_CONSTANTS.** Phonopy can write either a full N×N matrix or a compact file storing only one representative atom per Wyckoff orbit. betapy detects the compact case and reconstructs the missing pairs using spglib crystal symmetry. Install `spglib` if it is not already present:

```bash
pip install spglib
# or, reinstall betapy with GUI/full extras which include spglib:
pip install -e ".[gui]"
```

Detection still runs without spglib, but missing pairs will not be recovered and some chains may rarely go undetected. The fallback algorithm is robust but might still miss individual chains. This is only a concern for compact files; full FORCE_CONSTANTS files are unaffected.

---

## Running the detector

### CLI

```bash
# Minimal — runs with defaults betapy --multicenter

# Typical run with explicit parameters betapy --multicenter --mc-sigma 1.5 --mc-angle 150 --mc-max-order 5

# Slightly off-linear chains (e.g. Sb₂Te₃)
betapy --multicenter --mc-angle 130

# Disable the nearest-neighbour ratio filter (for unusual geometries)
betapy --multicenter --mc-ratio 0
```

The output lists flagged pairs with their distance, mean pFC, detection method, and significance in σ. Detected chains are printed with the species sequence, atom indices, and total end-to-end distance. If a POSCAR.lobster is found (in the auto-discovered LOBSTER directory or via `--lobster-dir`), cobiBetween directives are written to `multicenter_directives.txt`.

### GUI

Open the **Multicenter Bonding** tab from the **"+"** menu (top right of the tab bar). Load a SPOSCAR and FORCE_CONSTANTS first via the normal pFC Viewer workflow
— the multicenter tab reads whatever data is already loaded. Adjust σ, angle, and order in the controls at the top of the tab, then click **Detect**.

---

## Parameters

### `--mc-sigma` (default 1.5)

Detection threshold in robust standard deviations below the Badger baseline. A pair is flagged when its pFC exceeds what the Theil-Sen regression on FC<sup>−1/3</sup> vs. *r* predicts by more than this many MAD-based σ. Lower values flag more pairs and increase the risk of false positives from statistical scatter; higher values are more conservative.

1.5 σ works well for phase-change materials where the anomalous signal is strong. For systems where multicenter bonding is weaker or where the dataset is small, try 1.0–1.2.

### `--mc-angle` (default 150°)

Minimum bond angle (in degrees) for chain extension. When growing a chain from atom B, a candidate next atom C is only accepted if the angle A–B–C exceeds this threshold. This enforces approximate linearity and prevents chains from bending back on themselves or branching into cage-like structures.

150° is appropriate for GeTe-type rock-salt geometries andr slightly distorted structures such as Sb₂Te₃ or Bi₂Te₃ where the chains are not perfectly linear.

### `--mc-max-order` (default 5)

Maximum number of atoms in a chain. A value of 5 captures up to 5-center interactions (e.g. Te–Ge–Te–Ge–Te in GeTe). Raise this if you have reason to expect longer chains.

### `--mc-ratio` (default 1.5)

Maximum ratio of a chain step length to the species-pair nearest-neighbour distance. Chain growth is blocked when a candidate step exceeds this multiple of the NN distance. This prevents chains from forming through long-range contacts that are geometrically collinear but not genuinely bonded.
1.5 should be appropriate is generous for most compounds and should be able to capture even multicenter behavior at the primary/secondary bonding end of the spectrum.

### `--mc-bond-tol` (default 1.4)

Maximum ratio of a species pair's shortest *observed* distance to the sum of its covalent radii (Cordero/Alvarez values, the same table used for structure-view bond rendering) for that distance to be accepted as a real — possibly weak — bond.

This guards against a failure mode `--mc-ratio` cannot catch on its own: some species pairs never form a direct bond at *any* observed distance. In zincblende ZnS, for example, the only S-S contact in the structure is the 2nd-coordination-shell distance mediated through Zn (~3.8 Å) — there is no shorter, genuinely bonded S-S shell to compare against. Without this check, that distance gets adopted as "the S-S nearest-neighbour", so every periodic-image S-S hop sits at ratio ≈ 1.0 and trivially passes `--mc-ratio`, producing nonsensical single-sublattice chains (`S-S-S-S`, `Zn-Zn-Zn-Zn`) that just walk the lattice's own translation vector and contribute meaningless `cobiBetween atom1 atom1 cell ...` directives.

Species pairs whose shortest shell exceeds `--mc-bond-tol × (covalent radii sum)` are excluded from chain-hop traversal entirely (not merely ratio-limited). 1.4 was chosen empirically against the bundled example systems: it keeps the genuine weak/secondary same-species hop used in Sb₂Te₃ (Te-Te across the van der Waals gap, ratio 1.30) while excluding every purely geometric, non-bonded same-species contact observed (ZnS S-S/Zn-Zn, GeTe Ge-Ge/Te-Te, Sb₂Te₃ Sb-Sb, all ≥1.5). Pass 0 to disable and fall back to the previous behaviour (NN reference taken purely from the empirical shortest observed distance).

---

## Understanding the output

### Flagged pairs

Each flagged pair has a detection method and a significance value:

- **regression** — the pair's pFC was flagged relative to a Theil-Sen fit to the
  full species-pair dataset. The `n_sigma` value is the residual in robust
  standard deviations below the Badger line. A value of 2.0 means the pFC is
  two MADs larger than expected for that bond length.
- **monotone** — used when the species-pair dataset is too small (< 4 reliable
  pairs) to fit a meaningful baseline, but all available pairs show a pFC larger
  than the shortest-bond reference. Less statistically rigorous; treat these with
  more caution. `n_sigma` is NaN in this case.

### Chains

A chain is a sequence of atoms connected by consecutive flagged pairs, subject to the angle and ratio filters. The detector reports:

- The **species sequence** (e.g. Te–Ge–Te–Ge–Te) and the specific atom indices
- The **total end-to-end distance** and per-step distances
- **Sub-chains** — all shorter contiguous fragments within a longer chain. A
  5-center chain contains one 4-center and one 3-center sub-chain at each end,
  which are listed as separate entries so you can inspect the partial interactions.

In the GUI scatter plot, selecting a chain highlights all pairwise atom combinations with amber halos (anomalous pairs used to flag or build the chain)
and teal rings (end-to-end and intermediate pairs including non-consecutive ones). A dashed vertical line marks any pair that is missing from the scatter due to the reliability cutoff or compact FC expansion. This is expected for long-range pairs that fall outside the reliable region of the supercell.

### Reliability cutoff

Only pairs within half the shortest supercell dimension (L/2) enter the baseline fit and chain search. This is the region where periodic-image effects are negligible. Pairs beyond this limit are excluded from detection; the cutoff distance is printed in the output.

---

## cobiBetween directives

LOBSTER's `cobiBetween` keyword computes the crystal orbital bond index (COBI)
between two specific atoms that are not necessarily nearest neighbours, which is exactly what is needed for multicenter bond analysis. When a lobster POSCAR is found, betapy writes a `multicenter_directives.txt` containing one `cobiBetween` line per detected chain segment, formatted for direct insertion into the lobsterin input file.

The directives specify atoms by their lobster POSCAR index. Check the atom mapping by opening POSCAR in VESTA or by comparing the species list to your SPOSCAR.

---

## Worked example: GeTe

$\beta$-GeTe crystallizes in the rock-salt structure. Along the cubic ⟨100⟩ directions, alternating Ge and Te atoms form linear Te–Ge–Te–Ge–Te chains where the force constants are significantly enhanced relative to what the Badger relation would predict for isolated two-center force constants. This is the canonical case of "metavalent" multicenter bonding.

The `examples/GeTe/` directory contains a Phonopy supercell with a **compact**
FORCE_CONSTANTS (512 atoms, 8 Wyckoff orbits). Run detection from that directory:

```bash
cd examples/GeTe
betapy --multicenter --mc-sigma 1.5
```

betapy will:
1. Compute bulk pFCs from SPOSCAR and FORCE_CONSTANTS
2. Detect the compact format and expand the missing pairs via spglib
3. Fit the Theil-Sen baseline separately for Ge–Te and Te–Te pairs
4. Flag the anomalous short-range pairs in each group
5. Trace 5-center Te–Ge–Te–Ge–Te chains and report 4-center and 3-center
   sub-chains

Expected output: several Ge–Te pairs flagged at 2–4 σ, with Te–Ge–Te–Ge–Te as the dominant chain motif. The Te–Te end-to-end pair in a 3-center Te–Ge–Te is not detected as an anomaly. Due to the high symmetry of the cell and the low number of datapoints for the isoatomic species pairs as wekk as the high number of multicenter bonds relatative to the total number of pairs, the regression is quite poor. However, the pairs are still caught via the chain-building algorithm fromt he higher order trigger pairs.

In the GUI, open the Multicenter Bonding tab after loading the GeTe data, run detection, and click the Te–Ge–Te–Ge–Te entry in the chain tree. All 10 pairwise combinations within the 5-atom chain will be highlighted in the scatter plot.

---

## Common issues

**High symmetry, multi-element compounds with lots of unusual (i.e. multicenter) bonding have poor Badger fits.**
The detection relies on the "unusualness" of the force constants of atom pairs of a given pair species. In compounds with many different, elements and high symmetry, it becomes necessarily difficult to establish a baseline to trigger the anomalies. This is nicely illustrated in the example GeTe. Generally, the detector should still catch most cases through chain-building. If you encounter issues, please feel free to contact me.

**No chains detected despite known multicenter bonding.**  
Lower `--mc-sigma` incrementally (try 1.2, then 1.0). Check whether spglib is installed if using a compact FORCE_CONSTANTS — run `python -c "import spglib;
print(spglib.__version__)"` to verify. Also check the reliability cutoff printed in the output: if the supercell is small the cutoff may exclude the relevant pairs entirely.

**Too many false positives.**  
Raise `--mc-sigma` to 2.0 or higher. Check whether the species-pair dataset has outliers that are distorting the Badger baseline, for example very short or very long bonds at the edges of the distance range can pull the fit. The Theil-Sen regression is resistant to outliers but not immune.

**Chains that just walk one sublattice (e.g. `S-S-S-S` in ZnS), with `cobiBetween` directives between the same atom label at consecutive cell images.** This is not a sigma problem — it means a same-species pair has no genuine direct bond at any observed distance (its only shell is a 2nd-coordination-shell contact mediated through another species), so that distance gets misread as "the nearest-neighbour" and every periodic-image hop along it trivially passes `--mc-ratio`. `--mc-bond-tol` (default 1.4) screens species-pair NN references against covalent radii before they are used for chain-hop traversal; raising `--mc-sigma` will not fix this since the underlying statistical flag (e.g. ZnS S-S at 3.7σ) is not itself wrong, only its use as a chain-growth anchor is.

**Chains cut short at unexpected points.**  
The `--mc-ratio` filter may be blocking a step. Print the raw flagged pairs first
(look at the pFC vs distance scatter for the relevant species pair), and check whether the step in question is at a distance significantly larger than the NN. Pass `--mc-ratio 0` to temporarily disable the filter and see the full chain.

**Chain angle filter too strict or too loose.**  
If a known multicenter system produces no chains or too bent, nonsensical chains check the geometry of the relevant atom sequence in VESTA. For structures with a rhombohedral or orthorhombic distortion, the chains may be just a few degrees off linear. Adjust `--mc-angle` to 140° or 160°. the default 150° should be very generous in most cases, though.
