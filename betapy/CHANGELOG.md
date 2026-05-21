# Changelog

All notable changes to betapy are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
