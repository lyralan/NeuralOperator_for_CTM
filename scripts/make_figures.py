"""Figure generation placeholder."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    print("Figure generation placeholder. Output dir:", args.out)


if __name__ == "__main__":
    main()
