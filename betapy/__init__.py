import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def _source_tree_version():
    """Live version from pyproject.toml, used for editable/dev installs so the
    reported version can't go stale relative to installed package metadata."""
    pyproject = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    if not pyproject.exists():
        return None
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text())
    return match.group(1) if match else None


try:
    __version__ = _source_tree_version() or version("betapy")
except PackageNotFoundError:
    __version__ = "unknown"
