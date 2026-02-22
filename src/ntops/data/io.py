"""Simple NPZ IO helpers."""
from __future__ import annotations

import numpy as np


def save_npz(path: str, **arrays):
    np.savez_compressed(path, **arrays)


def load_npz(path: str):
    data = np.load(path)
    return {k: data[k] for k in data.files}
