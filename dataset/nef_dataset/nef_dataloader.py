# TODO
# Have a dataset class that takes in the path to the data and the split (we define a dataset class for each nef dataset we want)
# define augmentations in the config file.
# Use a function to convert config file of augmentations to a list of transforms
#
import itertools
import json
import os
from glob import glob
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
import torchdata.datapipes as dp
import torchvision.transforms as T
from absl import logging
from ml_collections import ConfigDict
from torch.utils.data.backward_compatibility import (
    worker_init_fn as comp_worker_init_fn,
)

import dataset.nef_dataset.nef_pipe as nef_pipe

from dataset.nef_dataset.augmentations import Augmentation
from dataset.nef_dataset.utils import (
    create_path_start_end_list,
    data_collate_multi_seed,
    numpy_collate,
)
from dataset.utils import data_split_sorting, start_end_idx_from_path


def build_nef_data_loader_group(
    data_config: ConfigDict,
    collate_fn: Callable = data_collate_multi_seed,
    transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            preload (optional): Whether to preload the data.
            split (optional): Split of the data to use. If None (default), all data is used.
            data_prefix (optional): Prefix of the data files.
        collate_fn: Collate function to use for the data loaders.
        transform: Transform to use for the datasets.

    Returns:
        List of data loaders.
    """
    assert transform is None or isinstance(
        transform, (Augmentation, list, tuple)
    ), f"Transform must be a list of transforms is instead {type(transform)}"

    data_split = data_config.get("split", [None])

    num_loaders = len(data_split) if data_split is not None else 1

    # when only one transform is provided, assume that it is the same for all loaders
    if isinstance(transform, Augmentation):
        transform = [transform] + [None for _ in range(num_loaders - 1)]
    elif transform is None:
        transform = [None] * num_loaders
    assert (
        len(transform) == num_loaders
    ), f"there must be a number of transforms ({len(transform)}) equal to the number of loaders ({num_loaders})."

    all_loaders = []

    start_idx = 0
    for loader_idx, split in enumerate(data_split):
        if split is None:
            end_idx = None
        else:
            end_idx = start_idx + split

        loader = build_nef_data_loader(
            data_config,
            start_idx=start_idx,
            end_idx=end_idx,
            is_train=loader_idx == 0,
            collate_fn=collate_fn,
            transform=transform[loader_idx] if transform is not None else None,
        )
        all_loaders.append(loader)
        start_idx = end_idx

    return all_loaders


def build_nef_data_loader(
    data_config: ConfigDict,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    is_train: bool = True,
    collate_fn: Callable = numpy_collate,
    transform: Optional[Callable] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            prelload (optional): Whether to preload the data.
            data_pipe_class (optional): The class of the data pipe to use.
            batch_size (optional): Batch size to use in the data loaders.
            num_workers (optional): Number of workers for each dataset.
            persistent_workers (optional): Whether to use persistent workers.
            seed (optional): Seed to initialize the workers and shuffling with.
        modes: Modes for which data loaders are created. Each mode corresponds to a file prefix.
            If None (default), all files in the data path are used for a single mode.

    Returns:
        List of data loaders.
    """
    files = nef_pipe.FilesDatapipe(
        path=data_config.get("path", ""), start_idx=start_idx, end_idx=end_idx
    )
    if is_train:
        files = files.shuffle()
    num_workers = data_config.get("num_workers", 0)
    shard_over_files = data_config.get("shard_over_files", True)
    if num_workers > 0 and shard_over_files:
        files = files.sharding_filter()
    
    
    pipe_class = getattr(nef_pipe, data_config.get("data_pipe_class", "NeFDatapipe")) # NeFDatapipe
    data_pipe = pipe_class(
        files,
        shuffle_slice=is_train and data_config.get("shuffle_slice", True),
        preload=data_config.get("preload", False),
        transform=transform,
    )
    if num_workers > 0 and not shard_over_files:
        data_pipe = data_pipe.sharding_filter()

    loader = data.DataLoader(
        data_pipe,
        batch_size=data_config.get("batch_size", 128),
        shuffle=is_train,
        drop_last=True,#is_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and data_config.get("persistent_workers", is_train),
        generator=torch.Generator().manual_seed(data_config.get("seed", 42)),
    )
    return loader


def build_multi_seed_nef_data_loader_group(
    data_config: ConfigDict,
    collate_fn: Callable = data_collate_multi_seed,
    transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            preload (optional): Whether to preload the data.
            split (optional): Split of the data to use. If None (default), all data is used.
            data_prefix (optional): Prefix of the data files.
        collate_fn: Collate function to use for the data loaders.
        transform: Transform to use for the datasets.

    Returns:
        List of data loaders.
    """
    assert transform is None or isinstance(
        transform, (Augmentation, list, tuple)
    ), f"Transform must be a list of transforms is instead {type(transform)}"

    data_split = data_config.get("split", [None])

    num_loaders = len(data_split) if data_split is not None else 1

    # when only one transform is provided, assume that it is the same for all loaders
    if isinstance(transform, Augmentation):
        transform = [transform] + [None for _ in range(num_loaders - 1)]
    elif transform is None:
        transform = [None] * num_loaders
    assert (
        len(transform) == num_loaders
    ), f"there must be a number of transforms ({len(transform)}) equal to the number of loaders ({num_loaders})."

    all_loaders = []

    start_idx = 0
    for loader_idx, split in enumerate(data_split):
        end_idx = start_idx + split

        loader = build_multi_seed_nef_data_loader(
            data_config,
            start_idx=start_idx,
            end_idx=end_idx,
            is_train=loader_idx == 0,
            num_seeds=data_config.get("num_seeds", 1),
            collate_fn=collate_fn,
            transform=transform[loader_idx] if transform is not None else None,
        )
        all_loaders.append(loader)
        start_idx = end_idx

    return all_loaders


def build_multi_seed_nef_data_loader(
    data_config: ConfigDict,
    start_idx: int,
    end_idx: int,
    num_seeds: int,
    is_train: bool = True,
    collate_fn: Callable = numpy_collate,
    transform: Optional[Callable] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            prelload (optional): Whether to preload the data.
            data_pipe_class (optional): The class of the data pipe to use.
            batch_size (optional): Batch size to use in the data loaders.
            num_workers (optional): Number of workers for each dataset.
            persistent_workers (optional): Whether to use persistent workers.
            seed (optional): Seed to initialize the workers and shuffling with.
        modes: Modes for which data loaders are created. Each mode corresponds to a file prefix.
            If None (default), all files in the data path are used for a single mode.

    Returns:
        List of data loaders.
    """
    files = nef_pipe.MultiSeedFilesDatapipe(
        path=data_config.get("path", ""),
        start_idx=start_idx,
        end_idx=end_idx,
        dataset_size=data_config.get("dataset_size", None),
        num_seeds=num_seeds,
    )
    if is_train:
        files = files.shuffle()
    num_workers = data_config.get("num_workers", 0)
    shard_over_files = data_config.get("shard_over_files", True)
    if num_workers > 0 and shard_over_files:
        files = files.sharding_filter()
    pipe_class = getattr(nef_pipe, data_config.get("data_pipe_class", "NeFDatapipe"))
    data_pipe = pipe_class(
        files,
        shuffle_slice=is_train and data_config.get("shuffle_slice", True),
        preload=data_config.get("preload", False),
        transform=transform,
        num_views=data_config.get("num_views", 2)
    )
    if num_workers > 0 and not shard_over_files:
        data_pipe = data_pipe.sharding_filter()

    loader = data.DataLoader(
        data_pipe,
        batch_size=data_config.get("batch_size", 128),
        shuffle=is_train,
        drop_last=True, #is_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and data_config.get("persistent_workers", is_train),
        generator=torch.Generator().manual_seed(data_config.get("seed", 42)),
    )
    return loader
