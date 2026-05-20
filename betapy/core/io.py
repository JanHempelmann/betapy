"""
File I/O for betapy.

All functions accept explicit file paths rather than assuming the current
working directory. This makes them usable from GUIs and tests, not just
the command line.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def read_SPOSCAR(path='SPOSCAR'):
    """
    Read a Phonopy SPOSCAR file.

    Returns a dict with keys:
        skal        : float, scaling factor
        lattice     : list of 3 lists, lattice vectors in Angstrom
        chem_symbols: list of str, chemical symbols in order
        chem_atoms  : list of int, number of atoms per species
        positions   : list of lists, fractional coordinates
    """
    path = Path(path)
    data = {
        'skal': 0,
        'lattice': [],
        'chem_symbols': [],
        'chem_atoms': [],
        'positions': [],
    }
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 1:
                data['skal'] = float(line.strip())
            elif 1 < i < 5:
                data['lattice'].append([float(x) for x in line.strip().split()])
            elif i == 5:
                data['chem_symbols'] = line.strip().split()
            elif i == 6:
                data['chem_atoms'] = [int(x) for x in line.strip().split()]
            elif i > 7:
                row = line.strip().split()
                if row:
                    data['positions'].append([float(x) for x in row])
    return data


def read_FORCE_CONSTANTS(path='FORCE_CONSTANTS'):
    """
    Read a Phonopy FORCE_CONSTANTS file (both symmetric and asymmetric).

    Returns a dict with keys:
        nats          : [n_first, n_total] from the header line
        atomic_pairs  : list of [i, j] 1-based atom index pairs
        force_matrices: list of 3x3 force constant matrices
    """
    path = Path(path)
    data = {
        'nats': [],
        'atomic_pairs': [],
        'force_matrices': [],
    }
    matrix_rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            tokens = line.strip().split()
            if i == 0:
                data['nats'] = [int(x) for x in tokens]
            elif len(tokens) == 2 and all(t.lstrip('-').isdigit() for t in tokens):
                data['atomic_pairs'].append([int(x) for x in tokens])
            elif tokens:
                matrix_rows.append([float(x) for x in tokens])
                if len(matrix_rows) == 3:
                    data['force_matrices'].append(matrix_rows)
                    matrix_rows = []
    return data


def read_refpos(path='REFPOS'):
    """
    Read a reference-site position file (formerly VACPOS).

    Format:
        Line 0: label string (e.g. 'v_Li', 'interstitial', anything)
        Line 1: number of sites
        Line 2: 'Direct'
        Lines 3+: fractional coordinates, one site per line

    Returns a dict with keys:
        label       : str
        num_sites   : int
        positions   : list of [x, y, z] fractional coordinates
    """
    path = Path(path)
    data = {
        'label': '',
        'num_sites': 0,
        'positions': [],
    }
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                data['label'] = line.strip()
            elif i == 1:
                data['num_sites'] = int(line.strip())
            elif i == 2:
                continue   # 'Direct' header line
            else:
                row = line.strip().split()
                if row:
                    data['positions'].append([float(x) for x in row])
    return data


def write_refpos(label, positions, path='REFPOS'):
    """
    Write a REFPOS file from a label string and list of fractional positions.

    Parameters
    ----------
    label     : str, descriptive label (e.g. 'v_Li', 'custom_site')
    positions : list of [x, y, z]
    path      : output file path
    """
    path = Path(path)
    with open(path, 'w') as f:
        f.write(f'{label}\n')
        f.write(f'  {len(positions)}\n')
        f.write('Direct\n')
        for pos in positions:
            f.write(f'  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}\n')


def write_unique_pfcs(df, path='unique_pFCs.csv'):
    """Write the unique projected force constants dataframe to CSV."""
    df.to_csv(path, sep=' ', header=True, index=False, encoding='utf-8')


def write_refsite_pfcs(df, path='refsite_pFCs.csv'):
    """Write the reference-site offsite projected force constants to CSV."""
    df.to_csv(path, sep=',', header=True, index=False, encoding='utf-8')


def write_refsite_onsite_pfcs(df, path='refsite_onsite_pFCs.csv'):
    """Write the reference-site onsite force constants to CSV."""
    df.to_csv(path, sep=',', header=True, index=False, encoding='utf-8')
