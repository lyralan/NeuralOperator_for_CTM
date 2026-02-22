"""Rollout evaluation placeholder."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False)
    args = parser.parse_args()
    print("Rollout evaluation is a placeholder. Provide checkpoint path:", args.checkpoint)


if __name__ == "__main__":
    main()
