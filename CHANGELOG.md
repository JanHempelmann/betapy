# Changelog

All notable changes to betapy are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- `betapy/core/constants.py` - single source of truth for shared constants
  (`PFC_ROUNDING_DECIMALS`, `SAME_SPECIES_METALS`); eliminates duplicated magic
  numbers across `projection.py`, `pfc_viewer.py`, and `structure_view.py`
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
