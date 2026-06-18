from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("betapy")
except PackageNotFoundError:
    __version__ = "unknown"
