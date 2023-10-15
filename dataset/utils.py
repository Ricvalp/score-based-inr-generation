import numpy as np

from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

from absl import logging


def start_end_idx_from_path(path: str) -> Tuple[int, int]:
    """Get start and end index from path.

    Args:
        path: Path from which to extract the start and end index.

    Returns:
        Tuple with start and end index.
    """
    start_idx = int(Path(path).stem.split("_")[1].split("-")[0])
    end_idx = int(Path(path).stem.split("_")[1].split("-")[1])
    return start_idx, end_idx


def path_from_name_idxs(name: str, start_idx: int, end_idx: int) -> str:
    return f"{name}_{start_idx}-{end_idx}.hdf5"


def data_split_sorting(x):
    if isinstance(x, tuple):
        x = x[0]
    if x == "train":
        return 0
    elif x == "valid" or x == "val":
        return 1
    elif x == "test":
        return 2
    elif x == "all":
        return -1
    else:
        logging.warning(f"Unknown mode {x}, defaulting to lexicographical ordering")
        return x


def find_mode(data_splits, idx):
    for split_name, split_idxs in sorted(data_splits.items(), key=data_split_sorting):
        if idx in split_idxs:
            return split_name

    raise ValueError(f"Index {idx} not found in any split")


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
