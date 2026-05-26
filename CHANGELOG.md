# Changelog

All notable changes to betapy are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-05-26

### Added
- **Settings-file path resolution** — relative file paths in YAML settings files
  (`sposcar`, `force_constants`, `refsite.file`, `stiffness_shift.refpos`, and all
  `structure_a`/`structure_b` paths) are now resolved relative to the YAML file's
  parent directory rather than the working directory; absolute paths pass through
  unchanged; makes settings files portable and shareable without editing paths
- **Non-blocking stiffness-shift analysis** — the Stiffness Shift tab now runs the
  heavy computation in a background `QThread` (`_StiffnessWorker`); an indeterminate
  `QProgressBar` appears while computing and the Run button is disabled to prevent
  double-triggering; the GUI remains fully responsive throughout
- **CITATION.cff** — GitHub "Cite this repository" metadata pointing to the 2021
  *Advanced Materials* paper that introduced projected force constants
- **62 automated tests** — full test suite covering shell grouping, species
  normalisation, reliability cutoff geometry, pair deduplication, progress
  callbacks, and end-to-end GeTe pipeline integration

### Changed
- Development status classifier updated from `4 - Beta` to `5 - Production/Stable`

---

## [0.4.0] — 2026-05-25

### Added
- **Shell view** in pFC Viewer — toggle between individual bond points and
  aggregated distance shells (one scatter point per shell, sized by bond count,
  with vertical pFC min/max range bars); shells are grouped by species pair and
  distance bin (0.01 Å precision, matching Phonopy's symmetry-equivalent bond
  distances exactly)
- **Multi-bond 3D highlighting** in shell mode — clicking a shell in the scatter
  plot highlights all bonds from the most-connected representative source atom in
  the 3D view; background atoms are dimmed at a gentler shell opacity
  (`SHELL_DIM_OPACITY = 0.35`) to preserve structural context
- **Reliability boundary** on all pFC scatter plots — two-zone shading marks the
  half-cell cutoff (L/2) computed correctly for any supercell geometry as
  `min(V/|b×c|, V/|a×c|, V/|a×b|) / 2`: yellow caution zone from 0.85·L/2 to
  L/2 and a red unreliable zone beyond L/2, with a dashed boundary line and Å
  label; the boundary is applied automatically whenever a SPOSCAR is loaded
- **GUI progress bar** for bulk pFC analysis — analysis now runs in a background
  `QThread` (`_AnalysisWorker`), keeping the GUI fully responsive; a `QProgressBar`
  appears in the status bar and is throttled to ~200 updates regardless of dataset
  size; the Analyse button is disabled during the run to prevent double-triggering
- Full N×N force constant matrix support in shell view — species-pair normalisation
  in `group_by_shells()` merges (A, B) and (B, A) records that arise when phonopy
  writes a complete N×N `FORCE_CONSTANTS` file, preventing the scatter plot from
  expanding to redundant mirrored pair types; a `max_distance` cutoff (L/2) keeps
  the shell count manageable for large supercells

### Fixed
- Shell mode 3D bond rendering — PyVista's `.tube()` on a batched `PolyData` with
  multiple disconnected 2-point line cells only tubed the first segment; each bond
  in a shell now gets its own `pv.Line(p1, p2).tube()` actor with a unique name
  (`highlight_bond_multi_{k}`) so all bonds are drawn
- Shell mode pair deduplication — full FC matrices store both (i, j) and (j, i),
  causing each bond to appear twice in the filtered pairs list after species
  normalisation and exactly half the expected bonds to be visible (as perfectly
  overlapping pairs); the click handler now deduplicates before passing pairs to
  `highlight_bonds()`

---

## [0.3.0] — 2025

### Added
- `betapy/core/constants.py` - single source of truth for shared constants
  (`PFC_ROUNDING_DECIMALS`, `EV_ANG2_TO_N_M`, `SAME_SPECIES_METALS`); eliminates
  duplicated magic numbers across modules
- Full Jmol colour table in `elements.py` - 80+ elements now have explicit
  colours; previously 52 elements (all 5d metals, lanthanides, Al, Si, Sn, Pb,
  etc.) fell back to medium grey
- VESTA colour preset in `elements.py` - identical to Jmol for transition metals,
  with distinct lanthanide colours and adjusted C/S tones for crystallographic figures
- `Preset` dropdown in the 3D structure view colour panel - switch between Jmol
  and VESTA at any time; individual colour overrides still work as before
- Display radii added for all transition metals, lanthanides, and common main-group
  elements in `elements.py`
- Refsite analysis now runs in a background `QThread` with an indeterminate
  progress dialog - GUI stays responsive during long calculations on large supercells
- Snap-to-atom combo is now searchable - type any element symbol or index to
  filter the list (contains-match, case-insensitive)
- Auto-load failures in the GUI now raise a `QMessageBox` warning instead of
  silently writing to the status bar
- **Unit toggle** - pFC values can be displayed in eV/Å² (native) or N/m
  (x16.022) across all GUI panels (scatter plots, tables, status bars, result
  labels); selection is persisted via `QSettings` and restored on next launch;
  CSV export always writes native eV/Å²
- `--unit eV/Ang2|N/m` CLI flag and `unit: N/m` YAML key - CLI output and
  stiffness-shift result observe the selected unit
- **Browser-style "+" tab button** - sits immediately to the right of the last
  tab; clicking opens a dropdown to add a New pFC Viewer, Ref. Site Projection,
  or Stiffness Shift tab without changing persistent preferences
- **Multiple pFC Viewer tabs** - additional pFC Viewer instances can be opened
  for side-by-side comparison; each tab is independent and closeable
- **Optional tabs** - Ref. Site Projection and Stiffness Shift are hidden by
  default and appear automatically when relevant files (`REFPOS` in CWD) or CLI
  flags (`--refsite`, `--stiffness-shift`) are detected; visibility mode
  (auto/always/never) is configurable per tab via the preferences dialog
  (⚙ button) and persisted via `QSettings`

### Fixed
- CSV separator inconsistency: `write_unique_pfcs()` now uses comma like all
  other CSV writers; `load_from_csv` in `pfc_viewer.py` simplified accordingly
- Result dict key `atom_distance` renamed to `distance` in refsite results to
  match the key used in bulk results - eliminates the `.get('atom_distance',
  r.get('distance', 0.0))` fallback pattern that appeared three times in
  `match_fc_pairs` and `fallback_equal_count_shift`
- `_try_autoload_from_settings()` now tracks what actually loaded vs what failed,
  and shows a dialog for errors instead of claiming success unconditionally
- Stiffness-shift atom-matching warning now names the current tolerance value and
  suggests concrete remedies (`match_tolerance`, cell origin check)
- `cli.py` now passes CLI arguments through to the GUI (`betapy --gui`) so that
  `--refsite` and `--stiffness-shift` flags correctly trigger optional tab
  visibility at startup
- Removed build artefacts (`__pycache__/`, `betapy.egg-info/`) from git tracking;
  `.gitignore` already covered these patterns but the files had been committed
  before the rules took effect

---

## [0.2.0] — 2025

Complete rewrite as an installable Python package.

### Added
- Package structure (`betapy/core/`, `betapy/gui/`, `betapy/data/`)
- `Supercell` class encapsulating PBC distance calculations
- `Settings` dataclass with YAML settings file support and CLI override hierarchy
- Reference-site pFC projection (`--refsite`) replacing the original `--vacancy` flag;
  projection point is now any fractional coordinate, not necessarily a vacancy
- Stiffness-shift analysis mode (`stiffness_shift:` in settings file):
  - Position-based atom matching across structures (robust to index reordering)
  - Fallback to equal-count distance-ordered comparison with terminal warning
  - Automatic exclusion of site-occupying atom in intercalated structure
- Interactive GUI (`betapy-gui`):
  - pFC scatter plot with click-to-highlight bond in 3D structure view
  - Reference site picker with 3D structure viewer
  - Auto-loading of `SPOSCAR`, `FORCE_CONSTANTS`, and `unique_pFCs.csv` from CWD
  - Load existing CSV without re-running analysis
  - CPK/Jmol atom colours with per-species colour picker
  - Automatic bond drawing (Alvarez 2008 covalent radii, KDTree-accelerated)
  - Per-species-pair bond type toggles
  - Atom-pair centering on selection (minimum-image convention)
  - Background atom dimming on pair selection
- `betapy --write-template` generates a commented YAML settings template
- Covalent radii data module (`betapy/data/elements.py`) from Alvarez 2008
  via qcelemental (86 elements)
- Test suite (`tests/`)

### Changed
- `VACPOS` renamed to `REFPOS`; `--vacancy` flag replaced by `--refsite`
- Force-constant projection direction convention preserved from original script
- CLI argument names updated: `--fc` for `FORCE_CONSTANTS`, `--refpos` for site file

### Fixed
- Asymmetric FORCE_CONSTANTS (`[N, M]` header with N < M) now handled correctly
- PBC distance calculation uses `floor(diff + 0.5)` minimum-image convention
  consistently throughout

---

## [0.1.0] — 2019–2023

Original single-file script (`betapy3FC.py`).

### Features
- Read SPOSCAR and FORCE_CONSTANTS from Phonopy
- Separate on-site and off-site force constants
- Project off-site FCs along interatomic bond vectors
- Identify unique pFC values
- Vacancy-site projection (`--vacancy DISTANCE`)
- Output to CSV

### Authors
- Original Python 2.7: J. Hempelmann (2019)
- Python 3 conversion and cleanup: M. Fecik
