"""Generate datasets for transport models."""
from __future__ import annotations

import argparse
import os
import yaml

from ntops.data.generate import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_path = cfg["output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    generate_dataset(
        path=out_path,
        nx=cfg.get("nx", 64),
        ny=cfg.get("ny", 64),
        lx=cfg.get("lx", 1.0),
        ly=cfg.get("ly", 1.0),
        dt=cfg.get("dt", 1e-3),
        nsteps=cfg.get("nsteps", 200),
        nsamples=cfg.get("nsamples", 100),
        D_range=tuple(cfg.get("D_range", [1e-4, 1e-2])),
        wind_scale=cfg.get("wind_scale", 1.0),
        seed=cfg.get("seed", 0),
        save_every=cfg.get("save_every", 1),
        num_workers=cfg.get("num_workers", 0),
    )

    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
