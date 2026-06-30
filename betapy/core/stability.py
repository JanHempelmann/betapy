"""
Opportunistic phonon-stability check using existing Phonopy post-processing
output.

betapy's streamlined workflow only requires SPOSCAR + FORCE_CONSTANTS, which
makes it easy to run a bond-projection analysis on a structure that was never
actually confirmed to be dynamically stable. Rather than diagonalizing a
dynamical matrix ourselves, this module looks for Phonopy post-processing
output (mesh.yaml / band.yaml / qpoints.yaml) that the user is likely to have
generated anyway as part of a normal phonopy workflow, and reads the mode
frequencies already computed there.

Files are checked in order of Brillouin-zone coverage: mesh.yaml samples the
whole zone on a grid, band.yaml only the chosen high-symmetry path (which
usually but not always includes the points where instabilities show up),
qpoints.yaml only whatever the user explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# Order reflects Brillouin-zone coverage, not file popularity.
_CANDIDATE_FILES = ('mesh.yaml', 'band.yaml', 'qpoints.yaml')

# Frequencies more negative than this (THz) are treated as genuine imaginary
# modes. Smaller-magnitude negative values are typically just acoustic-mode
# numerical noise at Gamma, not real instabilities.
DEFAULT_TOLERANCE_THZ = -0.01


@dataclass
class StabilityReport:
    source: Path
    source_kind: str            # 'mesh', 'band', or 'qpoints'
    n_qpoints: int
    n_modes_total: int
    n_imaginary: int
    min_frequency: float        # THz, most negative mode frequency found
    min_frequency_qpoint: Optional[list]

    @property
    def is_stable(self) -> bool:
        return self.n_imaginary == 0


def find_phonopy_yaml(directory) -> Optional[Path]:
    """
    Look for Phonopy post-processing output in *directory*, preferring
    whichever file covers the most of the Brillouin zone. Returns the first
    match, or None if none of the candidate files are present.
    """
    directory = Path(directory)
    for name in _CANDIDATE_FILES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _iter_mode_frequencies(data):
    """Yield (q_position, frequency) for every mode in a parsed Phonopy YAML dict."""
    for entry in data.get('phonon', []) or []:
        q = entry.get('q-position')
        for band in entry.get('band', []) or []:
            freq = band.get('frequency')
            if freq is not None:
                yield q, freq


def check_stability_from_yaml(path, tolerance_thz=DEFAULT_TOLERANCE_THZ) -> Optional[StabilityReport]:
    """
    Parse a Phonopy mesh.yaml / band.yaml / qpoints.yaml file and summarize
    mode stability.

    Returns None if pyyaml is unavailable, the file cannot be parsed, or it
    does not contain the expected 'phonon' section.
    """
    if not _YAML_AVAILABLE:
        return None
    path = Path(path)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict) or 'phonon' not in data:
        return None

    n_qpoints = len(data['phonon'])
    n_modes_total = 0
    n_imaginary = 0
    min_freq = float('inf')
    min_freq_q = None
    for q, freq in _iter_mode_frequencies(data):
        n_modes_total += 1
        if freq < min_freq:
            min_freq = freq
            min_freq_q = q
        if freq < tolerance_thz:
            n_imaginary += 1

    if n_modes_total == 0:
        return None

    return StabilityReport(
        source=path,
        source_kind=path.stem,
        n_qpoints=n_qpoints,
        n_modes_total=n_modes_total,
        n_imaginary=n_imaginary,
        min_frequency=min_freq,
        min_frequency_qpoint=min_freq_q,
    )


def check_stability(directory, tolerance_thz=DEFAULT_TOLERANCE_THZ) -> Optional[StabilityReport]:
    """
    Look for Phonopy post-processing output in *directory* and summarize mode
    stability if found. Returns None if no candidate file is present (i.e.
    there is nothing to check, not that the structure is stable).
    """
    path = find_phonopy_yaml(directory)
    if path is None:
        return None
    return check_stability_from_yaml(path, tolerance_thz=tolerance_thz)


def format_warning(report: StabilityReport) -> str:
    """
    Human-readable warning for an unstable report, '' if the structure is
    stable (so callers can do `if (msg := format_warning(report)): ...`).
    """
    if report.is_stable:
        return ''
    if report.min_frequency_qpoint is not None:
        q = ', '.join(f'{x:.3f}' for x in report.min_frequency_qpoint)
        q_str = f'q = ({q})'
    else:
        q_str = 'unknown q-point'
    return (
        f'{report.n_imaginary} imaginary mode(s) found in {report.source.name} '
        f'({report.source_kind} sampling, {report.n_qpoints} q-point(s), '
        f'{report.n_modes_total} modes total). '
        f'Worst: {report.min_frequency:.3f} THz at {q_str}. '
        f'This structure may not be fully relaxed / dynamically stable — '
        f'pFC results should be interpreted with caution.'
    )
