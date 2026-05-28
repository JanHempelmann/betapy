"""
File-keyed result cache for expensive computations.

Entries are stored in ~/.betapy_cache/ (one .pkl per computation run),
keyed by a hash of the input file fingerprints (resolved path + size +
mtime) and all computation parameters.  A cache hit on identical inputs
makes repeated analyses near-instant.

Usage
-----
    from betapy.core import cache

    key = cache.make_key(path_a, path_b, cutoff=6.0, tol=0.3)
    hit = cache.load(key)
    if hit is not None:
        return hit          # fast path

    result = expensive_computation(...)
    cache.save(key, result)
    return result

The cache is transparent — load() returns None on any miss or read
failure, save() silently ignores write failures.  Stale entries from
old betapy versions are caught by the except-and-delete path in load().
"""

import hashlib
import pickle
from pathlib import Path


def default_dir() -> Path:
    return Path.home() / '.betapy_cache'


def _fingerprint(path) -> str:
    p = Path(path).resolve()
    s = p.stat()
    return f"{p}|{s.st_size}|{s.st_mtime_ns}"


def make_key(*paths, **params) -> str:
    """
    Return a 16-character hex key from file fingerprints + parameters.

    Parameters
    ----------
    *paths  : path-like objects — all input files whose content affects
              the result.  Any mtime or size change invalidates the key.
    **params: scalar computation parameters (cutoff, tol, …).
    """
    parts = [_fingerprint(p) for p in paths]
    parts += [f"{k}={v!r}" for k, v in sorted(params.items())]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load(key: str, cache_dir=None):
    """
    Return the cached object for *key*, or None on miss / read failure.

    A corrupt or incompatible (old betapy version) cache file is deleted
    automatically so the next run recomputes cleanly.
    """
    path = (Path(cache_dir) if cache_dir else default_dir()) / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        path.unlink(missing_ok=True)
        return None


def save(key: str, data, cache_dir=None) -> None:
    """
    Persist *data* under *key*.  Write is atomic (tmp → rename) so an
    interrupted save never leaves a corrupt entry.  All errors are
    silently swallowed — a failed save just means the next run recomputes.
    """
    d = Path(cache_dir) if cache_dir else default_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{key}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(d / f"{key}.pkl")
    except Exception:
        pass


def clear(cache_dir=None) -> int:
    """Delete all cached entries.  Returns the number of files removed."""
    d = Path(cache_dir) if cache_dir else default_dir()
    count = 0
    for f in d.glob("*.pkl"):
        f.unlink(missing_ok=True)
        count += 1
    return count
