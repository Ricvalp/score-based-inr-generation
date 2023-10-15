import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict

from dataset.nef_dataset import augmentations
from dataset.nef_dataset.augmentations import Data, collate_data_multi_seed, collate_data


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return torch.tensor(batch)
    elif isinstance(batch[0], np.ndarray):
        return torch.tensor(np.stack(batch))
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return torch.tensor(np.array(batch))


def data_collate_multi_seed(batch: List[Data]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    return collate_data_multi_seed(batch, combine_op=torch.stack)


def data_collate(batch: List[Data]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    return collate_data(batch, combine_op=torch.stack)


def splits_to_names(split_sizes: List[int]) -> List[str]:
    start_idx = 0
    split_names = []
    for split_size in split_sizes:
        end_idx = start_idx + split_size
        split_names.append(f"{start_idx}_{end_idx}")
        start_idx = end_idx
    return split_names


def is_range_in_file(start_idx, end_idx, file_start_idx, file_end_idx):
    return start_idx < file_end_idx and end_idx > file_start_idx


def create_path_start_end_list(path_start_end_idxs, start_idx, end_idx):
    used_files = []

    for path, file_start_end in path_start_end_idxs:
        file_start_idx, file_end_idx = file_start_end
        if is_range_in_file(start_idx, end_idx, file_start_idx, file_end_idx):
            file_start_idx, file_end_idx = file_start_end

            used_files.append(
                (
                    path,
                    max(start_idx - file_start_idx, 0),
                    min(end_idx - file_start_idx, file_end_idx - file_start_idx),
                )
            )

    return used_files


def get_normalization_params(data_config):
    if not Path(data_config.path).exists():
        raise RuntimeError(
            f"Data path {data_config.path} does not exist, please download or generate the data first."
        )

    metadata_file = Path(data_config.path) / Path("metadata.json")

    if not metadata_file.exists():
        raise RuntimeError(
            f"Metadata file does not exist, please run `python dataset/nef_dataset/nef_normalization.py --path {data_config.path} --split {','.join([str(x) for x in data_config.split])} --batch_size {data_config.get('batch_size', 256)} --num_workers {data_config.get('num_workers', 0)}`"
        )

    metadata = ConfigDict(json.load(open(metadata_file)))

    split_names = splits_to_names(data_config.get("split", []))

    normalization_params = []

    for split_name in split_names:
        normalization_params.append(
            {"mean": np.array(metadata[split_name]["mean"]), "std": np.array(metadata[split_name]["std"])}
        )

    return normalization_params


def get_param_config(data_config: ConfigDict):
    # find the first parameter hdf5 file
    hdf5_path = glob(os.path.join(data_config.path, "*.hdf5"))[0]
    with h5py.File(hdf5_path, "r") as hdf5_file:
        param_config = json.loads(hdf5_file["param_config"][0])
        return param_config


def get_param_keys(data_config: ConfigDict):
    # find the first parameter hdf5 file
    hdf5_path = glob(os.path.join(data_config.path, "*.hdf5"))[0]
    with h5py.File(hdf5_path, "r") as hdf5_file:
        param_config = json.loads(hdf5_file["param_config"][0])
        param_keys = [key for key, shape in param_config]
        return param_keys


def get_shape_params(data_config: ConfigDict):
    # find the first parameter hdf5 file
    hdf5_path = glob(os.path.join(data_config.path, "*.hdf5"))[0]
    with h5py.File(hdf5_path, "r") as hdf5_file:
        params = hdf5_file["params"][0]
        return params.shape


def create_augmentations(
    cfg_aug_list: Optional[List[ConfigDict]] = None, data_config: Optional[ConfigDict] = None
):
    if cfg_aug_list is None:
        return None
    augm = []
    for aug_dict in cfg_aug_list:
        aug_name, aug_params = aug_dict["name"], aug_dict["params"]

        if aug_name == "Normalize":
            # only select the first normalization parameters because we assume they belong to the training set and that's the only one we ever normalize
            aug_params = get_normalization_params(data_config=data_config)[0]
        elif aug_name == "ParametersToList":
            aug_params = {"param_structure": get_param_config(data_config=data_config)}
        elif aug_name == "RandomMLPWeightPermutation":
            aug_params["param_keys"] = get_param_keys(data_config=data_config)
        elif (
            aug_name == "RandomRotate"
            or aug_name == "RandomScale"
            or aug_name == "RandomTranslate"
        ):
            param_keys = get_param_keys(data_config=data_config)
            aug_params["param_keys"] = param_keys
            include_pattern = aug_params.get("include_pattern", None)
            if include_pattern is not None:
                aug_params["exclude_params"] = [
                    x for i, x in enumerate(param_keys) if re.search(include_pattern, x) is None
                ]
            else:
                raise ValueError(
                    f"{aug_name} augmentation requires an include_pattern to be specified"
                )
            logging.debug(
                f"{aug_name} augmentation will exclude the following parameters: {aug_params['exclude_params']}"
            )
            del aug_params["include_pattern"]
        if aug_name not in augmentations.AVAILABLE_AUGMENTATIONS:
            raise ValueError(
                f"Augmentation {aug_name} not found in augmentations.py, the available elements in augmentations are: {augmentations.AVAILABLE_AUGMENTATIONS}"
            )

        aug_class = getattr(augmentations, aug_name)
        augm.append(aug_class(**aug_params))
    return augmentations.Compose(augm)
