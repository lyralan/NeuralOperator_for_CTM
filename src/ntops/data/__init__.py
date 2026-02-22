from .generate import generate_dataset

try:
    from .dataset import NPZTrajectoryDataset
    __all__ = ["generate_dataset", "NPZTrajectoryDataset"]
except Exception:
    __all__ = ["generate_dataset"]
