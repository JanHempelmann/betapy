# betapy

**betapy — phonon-based bonding analysis for crystalline materials**

betapy post-processes the force constants calculated by [Phonopy](https://phonopy.github.io/phonopy/) to extract projected force constants (pFCs) along interatomic bond vectors. It is designed for layered oxide and related intercalation materials, with particular support for analysing force constant distributions around vacancy and reference sites — and for quantifying how structural stiffness changes between intercalated and deintercalated phases.

---

## Features

- **Bulk pFC analysis** — project force constants along all interatomic bond vectors; identify and tabulate unique pFC values per bond type
- **Reference-site projection** — project force constants around any fractional coordinate in the cell (vacancy, interstitial, or arbitrary point); does not need to coincide with an atom
- **Stiffness-shift parameter** — compare pFC sums between two structures (e.g. intercalated vs deintercalated) using position-based atom matching across structures; falls back to distance-ordered equal-count comparison if matching fails
- **Interactive GUI** — scatter plot of pFC vs bond length with click-to-highlight; 3D structure viewer with CPK colours, automatic bond drawing, and per-species-pair bond toggles
- **Settings-file workflow** — YAML settings file with CLI flag overrides, following the Phonopy convention

---

## Requirements

- Python 3.8 or later
- [Phonopy](https://phonopy.github.io/phonopy/) — for generating `SPOSCAR` and `FORCE_CONSTANTS` inputs
- See `pyproject.toml` for Python package dependencies (numpy, pandas, matplotlib, PyQt5, PyVista, pyyaml, scipy)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/betapy.git
cd betapy
pip install -e .
```

For the GUI, PyVista requires VTK. On conda environments this installs more reliably via conda-forge:

```bash
conda install -c conda-forge pyvista pyvistaqt
pip install -e .
```

---

## Inputs

betapy reads standard Phonopy output files directly:

| File | Description |
|------|-------------|
| `SPOSCAR` | Phonopy supercell structure |
| `FORCE_CONSTANTS` | Phonopy force constant matrices |
| `REFPOS` | Reference site positions (betapy-specific, see below) |

### REFPOS format

```
v_Li              ← label (any string)
  1               ← number of sites
Direct
  0.0807  0.0417  0.1118   ← fractional coordinates, one per line
```

The reference site does not need to coincide with an atom — it can be a vacancy, interstitial, or any point of interest in the cell.

---

## Usage

### Command line

```bash
# Bulk pFC analysis, write results to CSV
betapy --sposcar SPOSCAR --fc FORCE_CONSTANTS --store

# Reference-site projection with 5 Å cutoff
betapy --sposcar SPOSCAR --fc FORCE_CONSTANTS \
       --refsite 5.0 --refpos REFPOS --store

# Using a settings file (recommended for complex runs)
betapy --settings betapy.yaml

# Write a commented template settings file
betapy --write-template

# Launch GUI
betapy-gui
# or
betapy --gui
```

### Settings file

Generate a template with `betapy --write-template`, then edit it:

```yaml
sposcar: SPOSCAR
force_constants: FORCE_CONSTANTS
store: true

refsite:
  file: REFPOS
  cutoff: 5.0

stiffness_shift:
  structure_a:
    sposcar: path/to/deintercalated/SPOSCAR
    force_constants: path/to/deintercalated/FORCE_CONSTANTS
    refpos: path/to/REFPOS
  structure_b:
    sposcar: path/to/intercalated/SPOSCAR
    force_constants: path/to/intercalated/FORCE_CONSTANTS
    refpos: path/to/REFPOS
  cutoff: 6.0
  min_site_dist: 0.1    # excludes site-occupying atom in intercalated structure
```

### GUI

Launch with `betapy-gui` from the directory containing your calculation files. betapy will auto-load `SPOSCAR`, `FORCE_CONSTANTS`, and `unique_pFCs.csv` if they are present in the current directory.

The GUI has two tabs:

**pFC Viewer** — scatter plot of projected force constant vs interatomic distance, coloured by atom-pair species type. Click any data point to highlight the corresponding bond in the 3D structure view. Existing `unique_pFCs.csv` files can be loaded directly without re-running the analysis.

**Reference Site Picker** — 3D structure viewer for placing a reference site. Click an atom to snap the site to it, or type fractional coordinates directly. Export the result as a `REFPOS` file.

---

## Outputs

| File | Description |
|------|-------------|
| `unique_pFCs.csv` | Unique projected force constants per atom pair (eV/Å²) |
| `refsite_pFCs.csv` | Off-site pFCs projected around reference site |
| `refsite_onsite_pFCs.csv` | On-site pFCs around reference site |
| `stiffness_shift.csv` | Per-pair pFC differences between two structures |

---

## Scientific background

Force constants from Phonopy are 3×3 matrices describing the second-order response of the energy to atomic displacements. betapy projects these matrices along the interatomic bond vector (for bulk pFCs) or along the vector from an atom to a reference site (for reference-site pFCs), yielding a scalar that captures the stiffness of each interaction along the relevant direction.

The concept of projecting force constants along interatomic bond vectors to connect lattice dynamics to chemical bonding was first developed by Deringer, Dronskowski, and Wuttig for phase-change materials, where the method was applied to study vibrational properties and bonding in chalcogenides. 

> Deringer, V. L.; Stoffel, R. P.; Wuttig, M.; Dronskowski, R. *Chem. Sci.* **2015**, *6*, 5255–5262. DOI: [10.1039/C5SC00825E](https://doi.org/10.1039/C5SC00825E)

**Projected force constants (pFCs)** were formally introduced with the corrected transpose symmetry treatment — recognising that Φ_p_(κκ′) ≠ Φ_p_(κ′κ) in general, and that the mean of both projections is the physically appropriate scalar — and applied to detect multicenter bonding in GeTe and related IV–VI chalcogenides:

> Hempelmann, J.; Müller, P. C.; Konze, P. M.; Stoffel, R. P.; Steinberg, S.; Dronskowski, R. *Adv. Mater.* **2021**, *33*, 2100163. DOI: [10.1002/adma.202100163](https://doi.org/10.1002/adma.202100163)

Further development of pFCs as a probe for multicenter bonding in phase-change materials:

> Hempelmann, J.; Müller, P. C.; Ertural, C.; Dronskowski, R. *Angew. Chem. Int. Ed.* **2022**, *61*, e202115778. DOI: [10.1002/anie.202115778](https://doi.org/10.1002/anie.202115778)

The **stiffness-shift parameter** implemented in betapy compares the sum of reference-site projected force constants between a deintercalated structure (projection around a vacancy) and an intercalated structure (projection around the occupied site, excluding the site-occupying atom). Atom pairs are matched across structures by fractional coordinate proximity rather than index, making the comparison robust to index reordering between VASP calculations.

Covalent radii used for automatic bond detection are from:

> Cordero et al. *Dalton Trans.* **2008**, 2832. DOI: [10.1039/b801115j](https://doi.org/10.1039/b801115j)

---

## Project structure

```
betapy/
├── betapy/
│   ├── core/
│   │   ├── io.py          # file reading/writing (SPOSCAR, FORCE_CONSTANTS, REFPOS)
│   │   ├── structure.py   # Supercell class, PBC distance calculations
│   │   ├── projection.py  # pFC mathematics, shell identification, stiffness shift
│   │   └── settings.py    # Settings dataclass, YAML loading, CLI parser
│   ├── gui/
│   │   ├── app.py         # main window, auto-loading, tab assembly
│   │   ├── pfc_viewer.py  # scatter plot + 3D view, click-to-highlight
│   │   ├── site_picker.py # reference site placement tool
│   │   └── structure_view.py  # shared PyVista 3D renderer
│   ├── data/
│   │   └── elements.py    # covalent radii, CPK colours, display radii
│   └── cli.py             # command-line entry point
└── tests/
    ├── test_io.py
    └── test_projection.py
```

---

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run tests with coverage
pip install pytest-cov
python -m pytest tests/ --cov=betapy
```

Contributions welcome — please open an issue before submitting a pull request.

---

## Acknowledgements

Development of betapy was supported by the Japan Society for the 
Promotion of Science (JSPS) KAKENHI Grant-in-Aid for Research Activity 
Start-up, Grant Number JP23KF0224. The underlying research work was done in the group of Prof. Fumiyasu Oba at the Institute of Science Tokyo, formerly Tokyo Institute of Technology. Early versions of the code were 
developed during doctoral research funded by the Deutsche 
Forschungsgemeinschaft (DFG) through the Collaborative Research Centre 
SFB 917 "Nanoswitches".

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Authors

Jan Hempelmann  
Original Python 2.7 script: J. Hempelmann (2019) based on a bash script by R. Stoffel
Python 3 conversion and extension (and generally making the code less embarrassing): M. Fecik  
Package restructuring, GUI, and stiffness-shift analysis: J. Hempelmann (2025)
