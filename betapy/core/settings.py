"""
Settings management for betapy.

Settings flow: hardcoded defaults → YAML file → CLI flags.
Each layer only overrides what it explicitly specifies; missing
keys always fall back to the previous layer.

Usage
-----
# From CLI
settings = Settings.from_cli()

# From a settings file only
settings = Settings.from_yaml('betapy.yaml')

# Programmatic (GUI)
settings = Settings()
settings.store = True
settings.refsite = RefsiteSettings(cutoff=4.0)

# Write a template settings file
Settings.write_template('betapy.yaml')
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional
import argparse
import textwrap

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Nested settings sections
# ---------------------------------------------------------------------------

@dataclass
class RefsiteSettings:
    """Settings for reference-site pFC projection."""
    file:                    str   = 'REFPOS'
    cutoff:                  float = 5.0
    exclude_refsite_species: bool  = True


@dataclass
class StructureSettings:
    """File paths for one structure (used in stiffness-shift comparison)."""
    sposcar:         str           = 'SPOSCAR'
    force_constants: str           = 'FORCE_CONSTANTS'
    # Per-structure REFPOS override. If None, falls back to
    # StiffnessShiftSettings.refpos (the shared fallback).
    refpos:          Optional[str] = None


@dataclass
class StiffnessShiftSettings:
    """Settings for the stiffness-shift comparison between two structures."""
    structure_a:    StructureSettings = field(
        default_factory=StructureSettings
    )
    structure_b:    StructureSettings = field(
        default_factory=StructureSettings
    )
    # Shared fallback REFPOS used when a structure does not specify its own.
    refpos:         str            = 'REFPOS'
    # Angstrom cutoff radius around the reference site.
    cutoff:          float = 4.0
    # Minimum distance from ref site for atom inclusion.
    # Excludes the site-occupying atom in the intercalated structure.
    min_site_dist:           float = 0.1
    # Exclude off-site pairs involving the intercalated species.
    # Species is inferred from the nearest atom in structure B (intercalated)
    # and applied consistently to both structures.
    exclude_refsite_species: bool  = True
    # Maximum Cartesian distance (Å) for atom position matching.
    # Atoms further apart than this are considered unmatched and trigger
    # the equal-count fallback for their species pair.
    match_tolerance: float = 1.5


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    Top-level settings object.  All parameters have sensible defaults so
    the object is always in a valid state.
    """
    # --- Input files (single-structure analysis) ---
    sposcar:         str  = 'SPOSCAR'
    force_constants: str  = 'FORCE_CONSTANTS'

    # --- Output ---
    store:           bool = False
    # Display unit for pFC values: 'eV/Ang2' (default) or 'N/m'
    unit:            str  = 'eV/Ang2'

    # --- Analysis modes ---
    # Refsite projection (formerly vacancy analysis)
    refsite:         Optional[RefsiteSettings]       = None
    # Stiffness-shift comparison
    stiffness_shift: Optional[StiffnessShiftSettings] = None

    # --- LOBSTER integration ---
    # Path to LOBSTER output directory.  When None, betapy auto-discovers a
    # sibling *_lobster directory (e.g. ScF3_ph → ScF3_lobster).
    lobster_dir:     Optional[str]                   = None

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Settings':
        """Load settings from a YAML file, falling back to defaults.

        All relative paths in the file are resolved relative to the directory
        containing the YAML, so the file can live anywhere and still find its
        SPOSCAR/FORCE_CONSTANTS siblings without needing CWD to match.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                'pyyaml is required to load settings files. '
                'Install it with: pip install pyyaml'
            )
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f'Settings file not found: {path}')
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        s = cls._from_dict(data)
        s._resolve_relative_paths(path.parent)
        return s

    def _resolve_relative_paths(self, base_dir: Path) -> None:
        """Resolve all file-path fields relative to base_dir in-place."""
        def _r(p: Optional[str]) -> Optional[str]:
            if p is None:
                return None
            pp = Path(p)
            return str((base_dir / pp).resolve()) if not pp.is_absolute() else p

        self.sposcar         = _r(self.sposcar)
        self.force_constants = _r(self.force_constants)

        if self.refsite is not None:
            self.refsite.file = _r(self.refsite.file)

        if self.stiffness_shift is not None:
            ss = self.stiffness_shift
            ss.refpos = _r(ss.refpos)
            for struct in (ss.structure_a, ss.structure_b):
                if struct is not None:
                    struct.sposcar         = _r(struct.sposcar)
                    struct.force_constants = _r(struct.force_constants)
                    struct.refpos          = _r(struct.refpos)

        self.lobster_dir = _r(self.lobster_dir)

    @classmethod
    def from_cli(cls, argv=None) -> 'tuple[Settings, argparse.Namespace]':
        """
        Parse command-line arguments and return a Settings object.
        If --settings is given, that file is loaded first and CLI flags
        override individual values.
        """
        parser = _build_parser()
        args   = parser.parse_args(argv)

        # Start from defaults (or settings file if provided)
        if args.settings:
            settings = cls.from_yaml(args.settings)
        else:
            settings = cls()

        # Apply CLI overrides (only flags that were explicitly set)
        _apply_cli_overrides(settings, args)
        return settings, args

    @classmethod
    def _from_dict(cls, data: dict) -> 'Settings':
        """Recursively build Settings from a plain dict (e.g. from YAML)."""
        s = cls()

        # Scalar top-level keys
        for key in ('sposcar', 'force_constants', 'store', 'unit', 'lobster_dir'):
            if key in data:
                setattr(s, key, data[key])

        # Refsite section
        if 'refsite' in data:
            rd = data['refsite']
            s.refsite = RefsiteSettings(
                file                    = rd.get('file',                    RefsiteSettings.file),
                cutoff                  = rd.get('cutoff',                  RefsiteSettings.cutoff),
                exclude_refsite_species = rd.get('exclude_refsite_species', RefsiteSettings.exclude_refsite_species),
            )

        # Stiffness-shift section
        if 'stiffness_shift' in data:
            sd = data['stiffness_shift']
            s.stiffness_shift = StiffnessShiftSettings(
                structure_a             = _dict_to_structure(sd.get('structure_a', {})),
                structure_b             = _dict_to_structure(sd.get('structure_b', {})),
                refpos                  = sd.get('refpos',                  StiffnessShiftSettings.refpos),
                cutoff                  = sd.get('cutoff',                  4.0),
                min_site_dist           = sd.get('min_site_dist',           0.1),
                exclude_refsite_species = sd.get('exclude_refsite_species', True),
                match_tolerance         = sd.get('match_tolerance',         1.5),
            )

        return s

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for YAML serialisation."""
        d = {
            'sposcar':         self.sposcar,
            'force_constants': self.force_constants,
            'store':           self.store,
            'unit':            self.unit,
        }
        if self.refsite is not None:
            d['refsite'] = {
                'file':                    self.refsite.file,
                'cutoff':                  self.refsite.cutoff,
                'exclude_refsite_species': self.refsite.exclude_refsite_species,
            }
        if self.stiffness_shift is not None:
            ss = self.stiffness_shift
            d['stiffness_shift'] = {
                'structure_a': {
                    'sposcar':         ss.structure_a.sposcar,
                    'force_constants': ss.structure_a.force_constants,
                    'refpos':          ss.structure_a.refpos,
                },
                'structure_b': {
                    'sposcar':         ss.structure_b.sposcar,
                    'force_constants': ss.structure_b.force_constants,
                    'refpos':          ss.structure_b.refpos,
                },
                'refpos':                  ss.refpos,
                'cutoff':                  ss.cutoff,
                'min_site_dist':           ss.min_site_dist,
                'exclude_refsite_species': ss.exclude_refsite_species,
                'match_tolerance':         ss.match_tolerance,
            }
        return d

    def to_yaml(self, path: str | Path):
        """Write current settings to a YAML file."""
        if not _YAML_AVAILABLE:
            raise ImportError('pyyaml required for YAML output.')
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f,
                      default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_template(path: str | Path = 'betapy.yaml'):
        """
        Write a fully-commented template settings file.
        Useful for new users to discover all available options.
        """
        template = textwrap.dedent("""\
            # betapy settings file
            # All values shown are defaults. Remove or comment out any line
            # to use the default. CLI flags always override this file.

            # --- Input files (single-structure analysis) ---
            sposcar: SPOSCAR
            force_constants: FORCE_CONSTANTS

            # --- Output ---
            store: false          # write CSV result files
            # unit: N/m          # display unit for pFC values (default: eV/Ang2)

            # --- Reference-site projection (uncomment to enable) ---
            # refsite:
            #   file: REFPOS
            #   cutoff: 5.0       # Angstrom radius around reference site

            # --- LOBSTER integration (optional) ---
            # lobster_dir:     # path to LOBSTER output directory; auto-discovered if omitted
            #                  # (e.g. ScF3_ph → ScF3_lobster)

            # --- Stiffness-shift comparison (uncomment to enable) ---
            # stiffness_shift:
            #   # Directory form: looks for SPOSCAR, FORCE_CONSTANTS, and REFPOS inside each directory
            #   structure_a: path/to/deintercalated/
            #   structure_b: path/to/intercalated/
            #   # Explicit-file form (use when filenames differ from defaults):
            #   # structure_a:
            #   #   sposcar: path/to/deintercalated/SPOSCAR
            #   #   force_constants: path/to/deintercalated/FORCE_CONSTANTS
            #   #   refpos:           # optional per-structure REFPOS override
            #   # structure_b:
            #   #   sposcar: path/to/intercalated/SPOSCAR
            #   #   force_constants: path/to/intercalated/FORCE_CONSTANTS
            #   #   refpos:           # optional per-structure REFPOS override
            #   cutoff: 4.0         # Angstrom radius around reference site
            #   min_site_dist: 0.1    # exclude atoms closer than this to ref site
            #   match_tolerance: 1.5  # Angstrom tolerance for pair fingerprint matching
        """)
        with open(path, 'w') as f:
            f.write(template)
        return path

    def __repr__(self):
        lines = ['Settings(']
        for f in fields(self):
            lines.append(f'  {f.name}={getattr(self, f.name)!r}')
        lines.append(')')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dict_to_structure(d) -> StructureSettings:
    if isinstance(d, str):
        p = Path(d)
        return StructureSettings(
            sposcar         = str(p / 'SPOSCAR'),
            force_constants = str(p / 'FORCE_CONSTANTS'),
            refpos          = str(p / 'REFPOS'),
        )
    return StructureSettings(
        sposcar         = d.get('sposcar',         StructureSettings.sposcar),
        force_constants = d.get('force_constants', StructureSettings.force_constants),
        refpos          = d.get('refpos',          None),
    )


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=True,
        allow_abbrev=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            betapy: projected force constant analysis for Phonopy supercells.

            Settings hierarchy: defaults → settings file (--settings) → CLI flags.
            CLI flags always win. Any parameter not specified falls back to the
            settings file, then to the built-in default.

            Quick start:
                betapy --settings betapy.yaml
                betapy --write-template          # write a commented template
                betapy --gui                     # open the graphical interface
        '''),
    )

    parser.add_argument(
        '--settings', metavar='FILE',
        help='YAML settings file. CLI flags override individual values.',
    )
    parser.add_argument(
        '--write-template', action='store_true',
        help='Write a commented template settings file (betapy.yaml) and exit.',
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Launch the graphical interface.',
    )
    parser.add_argument(
        '-s', '--store', action='store_true', default=None,
        help='Write CSV result files.',
    )
    parser.add_argument(
        '--unit', choices=['eV/Ang2', 'N/m'], default=None,
        help='Display unit for pFC values (default: eV/Ang2). '
             'N/m = multiply by 16.022.',
    )

    parser.add_argument(
        '--lobster-dir', metavar='DIR',
        help='Path to LOBSTER output directory.  If omitted, betapy looks for '
             'a sibling *_lobster directory next to the phonopy directory.',
    )
    parser.add_argument(
        '--multicenter', action='store_true',
        help='Detect anomalously large pFCs that may indicate multicenter bonding '
             'and generate cobiBetween directives for LOBSTER COBI(N) analysis. '
             'Requires a POSCAR in the LOBSTER directory (auto-discovered or via '
             '--lobster-dir).',
    )
    parser.add_argument(
        '--mc-sigma', type=float, default=2.5, metavar='N',
        help='Detection threshold in standard deviations (default: 2.5). '
             'Lower values flag more shells; raise to reduce false positives.',
    )
    parser.add_argument(
        '--mc-ratio', type=float, default=2.5, metavar='R',
        help='Maximum allowed ratio of pair distance to the species-pair nearest-'
             'neighbour distance (default: 2.5).  Pairs beyond this multiple of '
             'the NN distance are not flagged even if statistically anomalous — '
             'filters zig-zag chain artefacts (e.g. diamond) while passing '
             'genuine multicenter bonds.  Pass 0 to disable.',
    )
    parser.add_argument(
        '--mc-max-order', type=int, default=5, metavar='N',
        help='Maximum number of atoms in a multicenter chain (default: 5).',
    )
    parser.add_argument(
        '--mc-angle', type=float, default=150.0, metavar='DEG',
        help='Minimum bond angle for chain extension in degrees (default: 150). '
             'Use ~130 for slightly off-linear chains such as Sb2Te3.',
    )

    # Single-structure inputs
    io_group = parser.add_argument_group('input files')
    io_group.add_argument(
        '--sposcar', metavar='PATH',
        help='Path to SPOSCAR (default: SPOSCAR)',
    )
    io_group.add_argument(
        '--fc', metavar='PATH', dest='force_constants',
        help='Path to FORCE_CONSTANTS (default: FORCE_CONSTANTS)',
    )

    # Refsite
    rs_group = parser.add_argument_group('reference-site projection')
    rs_group.add_argument(
        '--refsite', type=float, metavar='CUTOFF',
        help='Enable refsite projection with this Angstrom cutoff.',
    )
    rs_group.add_argument(
        '--refpos', metavar='PATH',
        help='Path to REFPOS file (default: REFPOS)',
    )

    # Stiffness shift
    ss_group = parser.add_argument_group('stiffness-shift comparison')
    ss_group.add_argument(
        '--stiffness-shift', action='store_true',
        help='Enable stiffness-shift comparison (requires settings file or '
             '--structure-a / --structure-b).',
    )
    ss_group.add_argument(
        '--structure-a', nargs='+',
        metavar='PATH',
        help=(
            'Structure A for stiffness-shift comparison. Accept either: (1) a directory path containing SPOSCAR and FORCE_CONSTANTS, or (2) two explicit paths: SPOSCAR_A FC_A. Tab-completion works on directory paths.'
        ),
    )
    ss_group.add_argument(
        '--structure-b', nargs='+',
        metavar='PATH',
        help=(
            'Structure B for stiffness-shift comparison. Accept either: (1) a directory path containing SPOSCAR and FORCE_CONSTANTS, or (2) two explicit paths: SPOSCAR_B FC_B.'
        ),
    )
    return parser


def _structure_settings_from_arg(arg_list):
    """
    Resolve --structure-a / --structure-b CLI argument to a StructureSettings.

    Accepts either:
      - A single directory path: looks for SPOSCAR and FORCE_CONSTANTS inside
      - Two explicit file paths: [sposcar_path, fc_path]

    The directory form is the ergonomic default for interactive use since
    tab-completion works naturally on directory paths.
    """
    if len(arg_list) == 1:
        d = Path(arg_list[0])
        return StructureSettings(
            sposcar         = str(d / 'SPOSCAR'),
            force_constants = str(d / 'FORCE_CONSTANTS'),
        )
    elif len(arg_list) == 2:
        return StructureSettings(
            sposcar         = arg_list[0],
            force_constants = arg_list[1],
        )
    else:
        raise ValueError(
            f'--structure-a/b accepts 1 (directory) or 2 (SPOSCAR FC) arguments, got {len(arg_list)}: {arg_list}'
        )


def _apply_cli_overrides(settings: Settings, args: argparse.Namespace):
    """Apply only the CLI flags that were explicitly set."""

    if args.sposcar:
        settings.sposcar = args.sposcar
    if args.force_constants:
        settings.force_constants = args.force_constants
    if args.store:
        settings.store = True
    if args.unit:
        settings.unit = args.unit

    # Refsite
    if args.refsite is not None:
        if settings.refsite is None:
            settings.refsite = RefsiteSettings()
        settings.refsite.cutoff = args.refsite
    if args.refpos:
        if settings.refsite is None:
            settings.refsite = RefsiteSettings()
        settings.refsite.file = args.refpos

    # Stiffness shift
    if args.stiffness_shift:
        if settings.stiffness_shift is None:
            settings.stiffness_shift = StiffnessShiftSettings()
    if args.structure_a:
        if settings.stiffness_shift is None:
            settings.stiffness_shift = StiffnessShiftSettings()
        settings.stiffness_shift.structure_a = _structure_settings_from_arg(
            args.structure_a
        )
    if args.structure_b:
        if settings.stiffness_shift is None:
            settings.stiffness_shift = StiffnessShiftSettings()
        settings.stiffness_shift.structure_b = _structure_settings_from_arg(
            args.structure_b
        )

    if getattr(args, 'lobster_dir', None):
        settings.lobster_dir = args.lobster_dir
