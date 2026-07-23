# LOBSTER Multicenter COBI (orbitalwise)

## Summary

LOBSTER's `cobiBetween ... orbitalwise` directive computes the crystal
orbital bond index (COBI) for a chain of three or more atoms, broken down
per orbital combination across the chain (e.g. p-p-p, or individual
p<sub>x</sub>-p<sub>y</sub>-p<sub>z</sub> combinations) rather than only as
a single summed value. This is standard, already-published LOBSTER
functionality (COBI itself, and its N-center/orbitalwise decomposition) —
this page documents betapy's *tooling* for browsing that data, not a new
analysis method.

This tooling is independent of the [multicenter bond detector](multicenter.md):
it works from any LOBSTER directory containing `NcICOBILIST.lobster`,
whether the `cobiBetween` directives that produced it came from betapy's
own detector, from LOBSTER's `cobiBetween ... orbitalwise` output directly,
or were written by hand.

---

## Data source: NcICOBILIST by default, COBICAR opt-in

This tooling defaults to `NcICOBILIST.lobster`'s own integrated "Nc-ICOBI
(at) eF" values rather than `COBICAR.lobster`'s energy-resolved curves. There
is a known LOBSTER issue related to cell translated interactions that can 
yield erroneous values.

LOBSTER writes an automatic Nc-ICOBI entry for every ordinary
nearest-neighbour pair regardless of any `cobiBetween` directive, and
because of periodicity most bonds naturally appear at more than one
translation — so the same physical bond can almost always be cross-checked
against itself directly in the file that's loaded.
`betapy.core.lobster.check_ncicobi_consistency` does this automatically for
every directory the GUI opens, grouping pairwise entries by (species pair,
real-space distance) and reporting the spread of values within each group.
A large spread there (versus the small, sub-percent scatter we'd expect
from ordinary numerical noise) suggests this *specific* directory's
`NcICOBILIST` may show the same kind of translation-dependent divergence we
observed in `COBICAR`, and its multicenter values should then be treated
with the same caution — not trusted just because they came from a
different file. The result is shown as a banner at the top of the GUI tab;
treat that banner, not simply the presence of `NcICOBILIST.lobster`, as the
basis for how much to trust the numbers.

Two ways to get a reliable energy-resolved curve out of a translated
chain, if you need one:
- **Re-run LOBSTER on a supercell** built large enough that the chain's
  atoms are genuinely distinct atom indices rather than periodic images of
  each other — a chain that needs a `cell 0 -1 1`-type translation in the
  conventional cell needs the supercell expanded (at least doubled) along
  whichever axis/axes that translation vector has a nonzero component in,
  not necessarily all three. Reduce `KPOINTS` roughly in proportion along
  each expanded direction to keep the same real-space k-sampling density.
  Once every atom in the `cobiBetween` directive sits at cell `[0,0,0]`,
  the divergence we observed for translated interactions no longer applies.
- **The "Use COBICAR energy-resolved curves" checkbox**, off by default —
  see the GUI section below.

---

## Prerequisites

A LOBSTER directory containing:
- `POSCAR` (or `POSCAR.lobster` / `POSCAR.lobster.vasp`) — for atom-index mapping
- `NcICOBILIST.lobster` — produced whenever the `lobsterin` file contains
  one or more `cobiBetween` directives (3+ atoms); only chains requested
  with the `orbitalwise` flag carry the per-orbital breakdown, e.g.:

```
cobiBetween atom 1 atom 6 atom 8 orbitalwise
```

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

1. **Open LOBSTER directory…** — select a directory with
   `NcICOBILIST.lobster`. A banner appears immediately with the automatic
   reliability check result for *this* directory (see above) — read it
   before trusting anything below it. Every multicenter (3+ atom) chain
   found in the file is listed underneath, with its integrated ICOBI(N),
   orbital-row count (or "no orbitalwise data" if that chain wasn't
   requested with the `orbitalwise` flag), and a `[cell-free]` tag if the
   chain involves no periodic-cell translation — i.e. the one case where
   COBICAR energy-resolved values are also trustworthy, independent of the
   banner's verdict. Plain 2-atom entries are excluded from the chain list
   itself — LOBSTER writes an Nc-ICOBI row for every ordinary
   nearest-neighbour pair too (that's what the reliability check uses),
   but those pairwise values are already covered by the existing pairwise
   LOBSTER integration elsewhere in the app.
2. Select a chain to load its breakdown into the middle panel and the
   chart on the right.
3. **Use COBICAR energy-resolved curves** (off by default) switches the
   data source and the chart from an integrated-value bar chart to an
   energy-resolved curve overlay, as described above. If the selected
   chain involves a translation, a warning banner appears above the
   chart — the checkbox doesn't block you from looking, it just flags that
   this is exactly the situation where we've seen COBICAR and NcICOBILIST
   disagree, so it's worth a second look before relying on the number. If
   `COBICAR.lobster` has no matching entry for the chain, it falls back to
   the NcICOBILIST bar chart automatically.
4. **Group by orbital type (s/p/d/f)** (default on) collapses individual
   m-resolved rows (`5p_x`, `5p_y`, `5p_z`, …) into coarse type sums — a
   3-center chain with an s/p/d basis (9 orbitals/atom) has 729 individual
   rows but only 27 type-combinations. Uncheck to browse the full
   m-resolved list.
5. **Hide |ICOBI| <** unlists orbital contributions with nothing going on;
   0 (default) shows everything.
6. **Bar-chart mode** (default): shows the total (top bar, black) plus
   every currently shown orbital row, sorted by |ICOBI(N)|, coloured by
   sign (blue = bonding/positive, red = antibonding/negative — same
   convention as the pairwise LOBSTER viewer). Clicking a row in the list
   outlines the matching bar.
   **Curve mode** (COBICAR checkbox on): select one or more rows in the
   list to overlay their energy-resolved curves with the total.
   Either way, **Export plot…** saves the chart as SVG/PDF/PNG,
   **Export CSV…** saves the currently shown rows (respecting the active
   grouping/threshold/data source) as a table of ICOBI(N) at the Fermi level.
7. **Show gridlines** toggles the dotted value-axis gridlines.
8. **Lock aspect ratio for export** forces the saved plot to a fixed narrow
   portrait shape (width : height = 1 : 1.618, the golden ratio) regardless
   of the panel's on-screen shape — only the exported file is affected, not
   the live preview. Leave unchecked to export whatever shape the panel
   currently is.

**A near-zero total does not mean nothing is happening.** Opposite-sign
orbital contributions can cancel almost exactly in the sum even when
individual orbital contributions are large — always check the orbital
breakdown, not just the total, before concluding an interaction is
negligible.

---

## Plot conventions

- **The value axis is symmetric about zero**, so bonding and antibonding
  character get equal visual weight regardless of which happens to be
  larger in magnitude. This convention also applies to betapy's other
  COHP/COOP/COBI curve plots (the pairwise LOBSTER viewer and the
  multicenter Nc-COBI popup), which — unlike this tab — do still read
  energy-resolved curves from `*CAR.lobster`; keep the caveat above in mind
  if any bond you inspect there involves a periodic-cell translation.

## Vector export

Plots exported to SVG or PDF (from this tab's **Export plot…** button) keep
text as editable text objects rather than outlined shapes, so labels and
axis text open as live, selectable text in Illustrator (or any other vector
editor) — not paths you'd have to redraw. This applies regardless of
whether **Lock aspect ratio for export** is checked.
