"""Inverse problem placeholder."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("Inverse solver not yet implemented. Config:", args.config)


if __name__ == "__main__":
    main()
