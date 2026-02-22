"""Plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_field(field, title=None):
    plt.figure()
    plt.imshow(field, origin="lower")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    return plt.gcf()
