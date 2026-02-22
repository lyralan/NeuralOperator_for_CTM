# Neural Transport Operators

Hybrid neural-operator surrogates for 2D advection–diffusion transport modeling.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/make_data.py --config configs/data.yaml
python scripts/train.py --config configs/train_fno.yaml
```

## Structure

See `code_structure.txt` for the intended layout. The code here is a working scaffold
that implements a baseline PDE solver, dataset generation, and minimal training hooks.
