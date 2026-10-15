"""Vectorised MinHash + LSH banding (no external deps beyond numpy)."""
from __future__ import annotations

import numpy as np

from common import h32

P = np.uint64(4294967311)  # prime > 2^32; a,b,h < 2^32 keeps a*h+b < 2^64
MAXV = np.iinfo(np.uint64).max


def make_perms(n: int = 128, seed: int = 1):
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**32 - 1, size=n, dtype=np.uint64)
    b = rng.randint(0, 2**32 - 1, size=n, dtype=np.uint64)
    return a, b


def shingles(toks: list, k: int = 3) -> np.ndarray:
    if not toks:
        return np.zeros(0, dtype=np.uint64)
    if len(toks) < k:
        grams = ["\x1f".join(toks)]
    else:
        grams = ["\x1f".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]
    return np.unique(np.fromiter((h32(g) for g in grams), dtype=np.uint64, count=len(grams)))


def signature(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if sh.size == 0:
        return np.full(a.shape, MAXV, dtype=np.uint64)
    v = (np.outer(sh, a) + b) % P
    return v.min(axis=0)


_MULT_CACHE: dict = {}


def band_keys(sigs: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """(n, bands) uint64 keys; equal key in a band => candidate pair."""
    if rows not in _MULT_CACHE:
        _MULT_CACHE[rows] = np.array([(1000003 ** i) % (2**61 - 1) for i in range(rows)], dtype=np.uint64)
    n = sigs.shape[0]
    s = sigs[:, : bands * rows].reshape(n, bands, rows)
    with np.errstate(over="ignore"):
        keys = (s * _MULT_CACHE[rows]).sum(axis=2, dtype=np.uint64)
    # mix in band index so different bands never collide
    with np.errstate(over="ignore"):
        keys = keys ^ (np.arange(bands, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    return keys


def est_jaccard(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return (s1 == s2).mean(axis=-1)
