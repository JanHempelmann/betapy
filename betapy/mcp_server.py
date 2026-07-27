"""
MCP server exposing betapy's read-only analyses as tools for LLM agents.

Thin wrapper: parse tool arguments into betapy.core objects, call
betapy.core.analysis, serialize results to JSON. All scientific logic lives
in betapy.core; this mirrors betapy.cli (which prints text) but returns
structured data instead.

Run with: betapy-mcp   (stdio transport, for use with an MCP-aware client)
"""

from pathlib import Path
from typing import Optional, Union

from mcp.server.fastmcp import FastMCP

from betapy.core.constants import EV_ANG2_TO_N_M
from betapy.core.io import read_SPOSCAR, read_FORCE_CONSTANTS
from betapy.core.structure import Supercell
from betapy.core.settings import (
    Settings, RefsiteSettings, StiffnessShiftSettings, dict_to_structure,
)
from betapy.core.stability import check_stability, format_warning
from betapy.core.analysis import (
    compute_bulk_analysis, compute_refsite_analysis, compute_stiffness_shift,
)

mcp = FastMCP("betapy")


def _stability_to_dict(report):
    if report is None:
        return None
    return {
        'source':               str(report.source),
        'source_kind':          report.source_kind,
        'n_qpoints':            report.n_qpoints,
        'n_modes_total':        report.n_modes_total,
        'n_imaginary':          report.n_imaginary,
        'min_frequency':        report.min_frequency,
        'min_frequency_qpoint': report.min_frequency_qpoint,
        'is_stable':            report.is_stable,
        'warning':              None if report.is_stable else format_warning(report),
    }


@mcp.tool()
def check_phonon_stability(directory: str) -> dict:
    """
    Check for imaginary phonon modes using Phonopy mesh.yaml / band.yaml /
    qpoints.yaml, if present next to a structure.

    A structure with imaginary modes is not dynamically stable, so any pFC
    analysis on it should be interpreted with that caveat.

    Returns {"checked": false} if none of those files exist in `directory`.
    """
    report = check_stability(directory)
    if report is None:
        return {'checked': False}
    return {'checked': True, **_stability_to_dict(report)}


@mcp.tool()
def bulk_pfc_analysis(
    sposcar: str,
    force_constants: str,
    unit: str = 'eV/Ang2',
    lobster_dir: Optional[str] = None,
) -> dict:
    """
    Project every interatomic force constant in a supercell onto its bond
    vector (bond-projected force constants / pFCs), after Deringer &
    Dronskowski 2015 with betapy's forward/transpose symmetrization
    correction.

    Parameters
    ----------
    sposcar, force_constants : paths to a Phonopy SPOSCAR and FORCE_CONSTANTS
        file for the same supercell.
    unit : 'eV/Ang2' (default) or 'N/m'.
    lobster_dir : optional path to a LOBSTER output directory (COHPCAR.lobster
        etc. inputs). When given, ICOBI/ICOHP/ICOOP columns are attached to
        each unique pFC by nearest-distance match.

    Returns dict with n_offsite_pairs, n_onsite_terms, and unique_pfcs (one
    row per distinct species-pair/distance/pFC-value combination).
    """
    supercell = Supercell(read_SPOSCAR(sposcar))
    fc_data   = read_FORCE_CONSTANTS(force_constants)

    lobster_pairs = None
    if lobster_dir:
        from betapy.core.lobster import load_pairs
        lobster_pairs = load_pairs(Path(lobster_dir))

    data = compute_bulk_analysis(supercell, fc_data, lobster_pairs=lobster_pairs)
    df = data['df_unique'].copy()
    if not df.empty and unit == 'N/m':
        df['pFC value'] = df['pFC value'] * EV_ANG2_TO_N_M

    return {
        'unit':            unit,
        'n_offsite_pairs': len(data['results']),
        'n_onsite_terms':  len(data['onsite']),
        'unique_pfcs':     df.to_dict('records'),
    }


@mcp.tool()
def refsite_analysis(
    sposcar: str,
    force_constants: str,
    refpos_file: str,
    cutoff: float = 5.0,
    unit: str = 'eV/Ang2',
    exclude_refsite_species: bool = True,
) -> dict:
    """
    Project force constants toward one or more reference sites (e.g. a
    vacancy or intercalation site) listed in a REFPOS file, considering only
    atoms within `cutoff` Angstrom of each site.

    When REFPOS lists 2 or more sites, also returns a pairwise Sigma-pFC
    comparison between every pair of sites (e.g. to compare symmetry-
    inequivalent Wyckoff sites within the same structure).

    Returns dict with per-site summaries (`sites`), pairwise `comparisons`,
    and the full `offsite_pairs` / `onsite_terms` tables.
    """
    supercell = Supercell(read_SPOSCAR(sposcar))
    fc_data   = read_FORCE_CONSTANTS(force_constants)
    rs = RefsiteSettings(
        file=refpos_file, cutoff=cutoff,
        exclude_refsite_species=exclude_refsite_species,
    )
    data = compute_refsite_analysis(supercell, fc_data, rs, unit=unit)

    return {
        'unit':          data['unit'],
        'sites':         data['sites'],
        'comparisons':   data['comparisons'],
        'offsite_pairs': data['df_offsite'].to_dict('records'),
        'onsite_terms':  data['df_onsite'].to_dict('records'),
    }


@mcp.tool()
def stiffness_shift(
    structure_a: Union[str, dict],
    structure_b: Union[str, dict],
    refpos: Optional[str] = None,
    cutoff: float = 4.0,
    min_site_dist: float = 0.1,
    exclude_refsite_species: bool = True,
    match_tolerance: float = 1.5,
    unit: str = 'eV/Ang2',
) -> dict:
    """
    Compare bond stiffness between two related structures (e.g. host vs.
    intercalated) at corresponding reference sites, matching equivalent
    bonds across structures by fractional-coordinate fingerprint. Structure
    B uses 1.5x cutoff to still find equivalent pairs after cell expansion.

    Parameters
    ----------
    structure_a, structure_b : either a directory containing SPOSCAR,
        FORCE_CONSTANTS, and REFPOS, or a dict
        {"sposcar": ..., "force_constants": ..., "refpos": ...} (refpos
        optional, falls back to `refpos`).
    refpos : shared REFPOS path used when a structure dict has no "refpos".
    cutoff : Angstrom radius around the reference site in structure A.
    min_site_dist : atoms closer than this to the reference site in
        structure B are excluded (excludes the site-occupying atom itself).
    exclude_refsite_species : drop off-site pairs involving the species
        that occupies the reference site in structure B (the intercalant).
    match_tolerance : max fingerprint distance (Angstrom) for two bonds
        across structures to be considered the same bond.
    unit : 'eV/Ang2' (default) or 'N/m'.

    Returns Sigma delta_pFC (B - A), structural disturbance metrics, phonon
    stability of both input structures (if Phonopy mesh/band/qpoints output
    is present next to them), and — when B has an extra species vs. A —
    the total framework <-> intercalant bond contribution in B.
    """
    ss = StiffnessShiftSettings(
        structure_a=dict_to_structure(structure_a),
        structure_b=dict_to_structure(structure_b),
        refpos=refpos or StiffnessShiftSettings.refpos,
        cutoff=cutoff,
        min_site_dist=min_site_dist,
        exclude_refsite_species=exclude_refsite_species,
        match_tolerance=match_tolerance,
    )
    settings = Settings(unit=unit, stiffness_shift=ss)
    data = compute_stiffness_shift(settings)

    return {
        'unit': data['unit'],
        'structure_a': {
            'n_atoms':   data['structure_a']['n_atoms'],
            'n_pairs':   data['structure_a']['n_pairs'],
            'stability': _stability_to_dict(data['structure_a']['stability']),
        },
        'structure_b': {
            'n_atoms':   data['structure_b']['n_atoms'],
            'n_pairs':   data['structure_b']['n_pairs'],
            'stability': _stability_to_dict(data['structure_b']['stability']),
        },
        'n_unmatched_a': data['n_unmatched_a'],
        'n_unmatched_b': data['n_unmatched_b'],
        'total_shift':   data['total_shift_converted'],
        'disturbance': {
            'n_pairs':     data['disturbance']['n_pairs'],
            'total_abs':   data['disturbance']['total_abs_converted'],
            'mean_abs':    data['disturbance']['mean_abs_converted'],
            'min_delta':   data['disturbance']['min_delta_converted'],
            'min_species': data['disturbance']['min_species'],
        },
        'intercalant_species': data['intercalant_species'],
        'intercalant_total':   data['intercalant_total_converted'],
        'matched_pairs':       data['df_matched'].to_dict('records'),
    }


def main():
    mcp.run()


if __name__ == '__main__':
    main()
