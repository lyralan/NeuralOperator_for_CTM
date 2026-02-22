"""Checkpoint helpers."""
from __future__ import annotations

import torch


def save(path: str, model, optimizer=None, step: int | None = None):
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = step
    torch.save(payload, path)


def load(path: str, model, optimizer=None):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
