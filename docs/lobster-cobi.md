# LOBSTER Multicenter COBI (orbitalwise)

## Summary

LOBSTER's `cobiBetween ... orbitalwise` directive computes the crystal
orbital bond index (COBI) for a chain of three or more atoms, broken down
per orbital combination across the chain (e.g. p-p-p, or individual
p<sub>x</sub>-p<sub>y</sub>-p<sub>z</sub> combinations) rather than only as
a single summed value. This is standard, already-published LOBSTER
functionality (COBI itself, and its N-center/orbitalwise decomposition) —
this page documents betapy's *tooling* for browsing and plotting that data,
not a new analysis method.

This tooling is independent of the [multicenter bond detector](multicenter.md):
it works from any LOBSTER directory containing `COBICAR.lobster`, whether
the `cobiBetween` directives in it came from betapy's own detector, from
LOBSTER's `cobiBetween ... orbitalwise` output directly, or were written by
hand.

---

## Prerequisites

A LOBSTER directory containing:
- `POSCAR` (or `POSCAR.lobster` / `POSCAR.lobster.vasp`) — for atom-index mapping
- `COBICAR.lobster` — produced when the `lobsterin` file contains one or
  more `cobiBetween` directives; only entries requested with the
  `orbitalwise` flag carry the per-orbital breakdown, e.g.:

```
cobiBetween atom 1 atom 6 atom 8 orbitalwise
```

`NcICOBILIST.lobster` is optional and, if present, is used only as a
cross-check on the chain's total integrated value — it does not carry
orbital-resolved data itself.

**Directive syntax.** LOBSTER's official `cobiBetween` syntax writes `atom`
and its POSCAR index as separate tokens (`atom 1`, not `atom1`). betapy's
multicenter detector writes directives in this official form; the parser
also tolerates the concatenated form (`atom1`) for older files, but it is
not the format LOBSTER itself expects.

---

## GUI

Open the **LOBSTER COBI** tab from the **"+"** menu (top right of the tab
bar). This tab does not depend on any phonon force-constant data — only a
LOBSTER directory.

1. **Open LOBSTER directory…** — select a directory with `COBICAR.lobster`.
   Every multicenter (Nc) chain found in the file is listed on the left,
   with its integrated total (cross-checked against `NcICOBILIST.lobster`
   when available) and orbital-row count.
2. Select a chain to load its curves. The total energy-resolved curve and
   every orbital-combination curve become available in the middle panel.
3. **Group by orbital type (s/p/d/f)** (default on) collapses individual
   m-resolved rows (`5p_x`, `5p_y`, `5p_z`, …) into coarse type sums — a
   3-center chain with an s/p/d basis (9 orbitals/atom) has 729 individual
   rows but only 27 type-combinations. Uncheck to browse the full
   m-resolved list.
4. **Hide |ICOBI| <** unlists orbital contributions with nothing going on;
   0 (default) shows everything.
5. Select one or more rows to overlay their curves with the total on the
   shared plot; **Export plot…** saves it as SVG/PDF/PNG, **Export CSV…**
   saves the currently shown rows (respecting the active
   grouping/threshold) as a table of ICOBI(N) at the Fermi level.

**A near-zero total does not mean nothing is happening.** Opposite-sign
orbital contributions can cancel almost exactly in the sum even when
individual orbital contributions are large — always check the orbital
breakdown, not just the total, before concluding an interaction is
negligible.

---

## Plot conventions

All COHP/COOP/COBI curve plots in betapy (the pairwise LOBSTER viewer, the
multicenter Nc-COBI popup, and this orbitalwise browser) follow two
defaults:

- **The value axis (COHP/COOP/COBI) is symmetric about zero**, so bonding
  and antibonding character get equal visual weight regardless of which
  happens to be larger in magnitude.
- **The energy axis is pinned exactly to the data's own range** — matching
  `lobsterin`'s `COHPstartEnergy`/`COHPendEnergy` — rather than
  matplotlib's default autoscale margin.
