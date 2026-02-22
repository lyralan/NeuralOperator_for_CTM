"""PyTorch dataset wrappers for transport trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_npz


@dataclass
class TrajectoryData:
    c0: np.ndarray
    u: np.ndarray
    v: np.ndarray
    D: np.ndarray
    S: np.ndarray
    traj: np.ndarray


def _stats(x: np.ndarray):
    return float(np.mean(x)), float(np.std(x) + 1e-8)


class NPZTrajectoryDataset(Dataset):
    def __init__(
        self,
        path: str,
        normalize: bool = True,
        stats: dict | None = None,
        indices: np.ndarray | None = None,
    ):
        data = load_npz(path)
        if indices is not None:
            self.c0 = data["c0"][indices]
            self.u = data["u"][indices]
            self.v = data["v"][indices]
            self.D = data["D"][indices]
            self.S = data["S"][indices]
            self.traj = data["traj"][indices]
        else:
            self.c0 = data["c0"]
            self.u = data["u"]
            self.v = data["v"]
            self.D = data["D"]
            self.S = data["S"]
            self.traj = data["traj"]
        self.normalize = normalize

        if stats is None:
            self.stats = self.compute_stats()
        else:
            self.stats = stats

    def compute_stats(self, indices: np.ndarray | None = None):
        if indices is None:
            c0 = self.c0
            u = self.u
            v = self.v
            S = self.S
            y = self.traj[:, -1]
            D = self.D
        else:
            c0 = self.c0[indices]
            u = self.u[indices]
            v = self.v[indices]
            S = self.S[indices]
            y = self.traj[indices, -1]
            D = self.D[indices]

        stats = {
            "c0": _stats(c0),
            "u": _stats(u),
            "v": _stats(v),
            "S": _stats(S),
            "y": _stats(y),
            "D": _stats(D),
        }
        return stats

    def set_stats(self, stats: dict):
        self.stats = stats

    def set_normalize(self, normalize: bool):
        self.normalize = normalize

    def __len__(self):
        return self.c0.shape[0]

    def __getitem__(self, idx):
        # Inputs: c0, u, v, D, S | Target: final state
        c0 = torch.from_numpy(self.c0[idx]).float()
        u = torch.from_numpy(self.u[idx]).float()
        v = torch.from_numpy(self.v[idx]).float()
        D = torch.tensor(self.D[idx]).float()
        S = torch.from_numpy(self.S[idx]).float()
        y = torch.from_numpy(self.traj[idx, -1]).float()

        if self.normalize:
            m, s = self.stats["c0"]
            c0 = (c0 - m) / s
            m, s = self.stats["u"]
            u = (u - m) / s
            m, s = self.stats["v"]
            v = (v - m) / s
            m, s = self.stats["S"]
            S = (S - m) / s
            m, s = self.stats["y"]
            y = (y - m) / s
            m, s = self.stats["D"]
            D = (D - m) / s
        return {"c0": c0, "u": u, "v": v, "D": D, "S": S, "y": y}
