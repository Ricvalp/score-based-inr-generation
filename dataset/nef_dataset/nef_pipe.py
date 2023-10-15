from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torchdata.datapipes as dp

import jax.numpy as jnp
from dataset.nef_dataset.augmentations import Data, Identity, collate_data_multi_seed
from dataset.nef_dataset.utils import create_path_start_end_list
from dataset.utils import data_split_sorting, start_end_idx_from_path

PARAM_KEY = "params"


class NeFDatapipe(dp.iter.IterDataPipe):
    """A datapipe for loading NeF weights.

    Args:
        dp.iter.IterDataPipe: The base datapipe class.
    """

    def __init__(
        self,
        path_start_end_iter: Any,
        data_keys: List[str] = None,
        shuffle_slice: bool = False,
        preload: bool = False,
        transform: Optional[Any] = None,
    ):
        super().__init__()
        if transform is None:
            transform = Identity()
        self.transform = transform
        self.path_start_end_iter = path_start_end_iter

        self.data_keys = data_keys if data_keys is not None else []
        if PARAM_KEY not in self.data_keys:
            self.data_keys.insert(0, PARAM_KEY)
        self.shuffle_slice = shuffle_slice
        self.preload = preload
        self.storage = {}
        if self.preload:
            self.preload_data()
        self._determine_num_elements()

    def preload_data(self):
        for path, _, _ in self.path_start_end_iter:
            with h5py.File(path, "r") as f:
                self.storage[path] = {}
                for key in self.data_keys:
                    assert key in f.keys(), f"Key {key} not found in {path}"
                    self.storage[path][key] = f[key][:]

    def set_path_iter(self, path_start_end_iter: Any):
        self.path_start_end_iter = path_start_end_iter
        self.storage = {}
        if self.preload:
            self.preload_data()
        self._determine_num_elements()

    def _determine_num_elements(self):
        num_elements = 0
        for _, start_idx, end_idx in self.path_start_end_iter:
            num_elements += end_idx - start_idx

        self.num_elements = num_elements

    def __iter__(self):
        for path, start_idx, end_idx in self.path_start_end_iter:
            with h5py.File(path, "r") as f:
                if path in self.storage:
                    data = self.storage[path]
                else:
                    data = f

                slice_idxs = np.arange(start_idx, end_idx)
                if self.shuffle_slice:
                    np.random.shuffle(slice_idxs)

                for idx in slice_idxs:
                    data_element = Data.from_dict(
                        {key: torch.tensor(data[key][idx]) for key in self.data_keys}
                    )

                    val = self.transform(data_element)

                    yield val

    def __len__(self):
        return self.num_elements


class FilesDatapipe(dp.iter.IterDataPipe):
    def __init__(
        self,
        path: Union[str, Path],
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        data_prefix: str = "",
        seed: int = 42,
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"

        rng = np.random.default_rng(seed)

        file_pattern = f"{data_prefix}*.hdf5"
        paths = list(path.glob(f"{data_prefix}*.hdf5"))

        assert (
            len(paths) > 0
        ), f"No files found at `{path.absolute()}` with pattern `{file_pattern}`"

        start_end_idxs = [start_end_idx_from_path(path) for path in paths]
        # this list has (path, (start_idx, end_idx))
        path_start_end_idxs = list(zip(paths, start_end_idxs))

        if end_idx is None:
            # select all nefs in the files selected by the pattern
            end_idx = max([end_idx for _, end_idx in start_end_idxs])

        self.path_start_end_list = create_path_start_end_list(
            path_start_end_idxs, start_idx, end_idx
        )

    def __iter__(self):
        yield from self.path_start_end_list


class MNISTFilesDatapipe(FilesDatapipe):
    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        data_prefix: str = "",
        seed: int = 42,
    ):
        if split == "train":
            start_idx = 0
            end_idx = 45000
        elif split == "val":
            start_idx = 45000
            end_idx = 50000
        elif split == "test":
            start_idx = 50000
            end_idx = 60000
        else:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        super().__init__(path, start_idx, end_idx, data_prefix, seed)


class CIFAR10FilesDatapipe(FilesDatapipe):
    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        data_prefix: str = "",
        seed: int = 42,
    ):
        if split == "train":
            start_idx = 0
            end_idx = 45000
        elif split == "val":
            start_idx = 45000
            end_idx = 50000
        elif split == "test":
            start_idx = 50000
            end_idx = 60000
        else:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        super().__init__(path, start_idx, end_idx, data_prefix, seed)


class NefInspectionDatapipe(NeFDatapipe):
    def __init__(
        self,
        path_start_end_iter: Any,
        shuffle_slice: bool = False,
        preload: bool = False,
        transform: Optional[Any] = None,
    ):
        super().__init__(
            path_start_end_iter,
            [PARAM_KEY, "param_config", "labels"],
            shuffle_slice,
            preload,
            transform,
        )


class NeFClassificationDatapipe(NeFDatapipe):
    """A datapipe for loading NeF weights for classification models.

    Args:
        NeFDatapipe: The base datapipe class.
    """

    def __init__(
        self,
        path_start_end_iter: Any,
        shuffle_slice: bool = False,
        preload: bool = False,
        transform: Optional[Any] = None,
    ):
        super().__init__(
            path_start_end_iter,
            [PARAM_KEY, "labels"],
            shuffle_slice,
            preload,
            transform,
        )


def sort_key_path(path):
    return int(path.stem.split("_")[1].split("-")[0])


class MultiSeedFilesDatapipe(dp.iter.IterDataPipe):
    def __init__(
        self,
        path: Union[str, Path],
        dataset_size: int,
        start_idx: int,
        end_idx: int,
        data_prefix: str = "",
        seed: int = 42,
        num_seeds: int = 1,
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"

        rng = np.random.default_rng(seed)

        file_pattern = f"{data_prefix}*.hdf5"
        paths = list(path.glob(f"{data_prefix}*.hdf5"))

        assert (
            len(paths) > 0
        ), f"No files found at `{path.absolute()}` with pattern `{file_pattern}`"

        start_end_idxs = [start_end_idx_from_path(path) for path in paths]
        # this list has (path, (start_idx, end_idx))
        path_start_end_idxs = list(zip(paths, start_end_idxs))

        # make a list with all the paths for each seed
        partial_path_start_end_list = []
        verifier_length = 0
        cumulative_length = 0
        for seed_idx in range(num_seeds):
            partial_path_start_end_list += create_path_start_end_list(
                path_start_end_idxs,
                start_idx + seed_idx * dataset_size,
                end_idx + seed_idx * dataset_size,
            )
            # check that all seeds have the same number of files in them
            if seed_idx == 0:
                verifier_length = len(partial_path_start_end_list)
            else:
                cur_length = len(partial_path_start_end_list) - cumulative_length
                assert (
                    verifier_length == cur_length
                ), f"Something went wrong with the path list creation, the number of paths for the first seed ({verifier_length}) should be equal to the number of paths for the other seeds ({cur_length})."
            cumulative_length = len(partial_path_start_end_list)

        # calculate the number of paths per seed
        num_paths = len(partial_path_start_end_list) // num_seeds

        partial_path_start_end_list = sorted(
            partial_path_start_end_list, key=lambda x: sort_key_path(x[0])
        )

        # for each path file, make a list with all the paths for each seed
        self.path_start_end_list = []
        for path_idx in range(num_paths):
            self.path_start_end_list.append([])
            for seed_idx in range(num_seeds):
                self.path_start_end_list[path_idx].append(
                    partial_path_start_end_list[path_idx + seed_idx * num_paths]
                )

    def __iter__(self):
        yield from self.path_start_end_list


class NeFMultiSeedDatapipe(dp.iter.IterDataPipe):
    """A datapipe for loading NeF weights.

    Args:
        dp.iter.IterDataPipe: The base datapipe class.
    """

    def __init__(
        self,
        path_start_end_iter: Any,
        num_views: int,
        data_keys: List[str] = None,
        shuffle_slice: bool = False,
        preload: bool = True,
        transform: Optional[Any] = None,
    ):
        super().__init__()
        if transform is None:
            transform = Identity()
        self.transform = transform
        self.num_views = num_views

        self.data_keys = data_keys if data_keys is not None else []
        if PARAM_KEY not in self.data_keys:
            self.data_keys.insert(0, PARAM_KEY)
        self.shuffle_slice = shuffle_slice
        self.preload = preload
        self.set_path_iter(path_start_end_iter)

    def preload_data(self):
        for path_idx, path_list in enumerate(self.path_start_end_iter):
            self.storage.append([])
            for seed_idx, (path, _, _) in enumerate(path_list):
                with h5py.File(path, "r") as f:
                    self.storage[path_idx].append({})
                    for key in self.data_keys:
                        assert key in f.keys(), f"Key {key} not found in {path}"
                        self.storage[path_idx][seed_idx][key] = f[key][:]

    def set_path_iter(self, path_start_end_iter: Any):
        self.path_start_end_iter = path_start_end_iter
        self.storage = []
        if self.preload:
            self.preload_data()
        self._determine_num_elements()
        self.num_seeds = len(next(iter(self.path_start_end_iter)))

    def _determine_num_elements(self):
        idxs = []
        num_elements = 0
        for path_list in self.path_start_end_iter:
            _, start_idx, end_idx = path_list[0]
            num_elements += end_idx - start_idx
            idxs.append(np.arange(start_idx, end_idx))

        self.idxs = np.concatenate(idxs, axis=0)
        self.num_elements = num_elements

    def path_from_idx(self, idx):
        for path, start_idx, end_idx in self.path_start_end_iter[0]:
            if idx >= start_idx and idx < end_idx:
                return path, start_idx, end_idx

    def __iter__(self):
        for path_idx, path_list in enumerate(self.path_start_end_iter):
            path, start_idx, end_idx = path_list[0]
            slice_idxs = np.arange(start_idx, end_idx)
            if self.shuffle_slice:
                np.random.shuffle(slice_idxs)

            # load all seeds in RAM if necessary
            if path_idx in self.storage:
                data = self.storage[path_idx]
            else:
                data = []
                for path, start_idx, end_idx in path_list:
                    data.append({})
                    with h5py.File(path, "r") as f:
                        data[-1] = {}
                        for key in f.keys():
                            data[-1][key] = f[key][:]

            for relative_idx in slice_idxs:
                seed_idxs = np.random.choice(np.arange(0, len(path_list)), self.num_views, replace=False)
                data_element = {
                        key: [torch.tensor(data[seed_idxs[0]][key][relative_idx])]
                        for key in self.data_keys
                    }

                for i in range(self.num_views-1):
                    for key in self.data_keys:
                        data_element[key].append(torch.tensor(data[seed_idxs[i+1]][key][relative_idx]))

                for key in self.data_keys:
                    data_element[key] = torch.stack(data_element[key], dim=0)

                data_element = Data.from_dict(
                    {
                        key: data_element[key]
                        for key in self.data_keys
                    }
                )

                val = self.transform(data_element)

                yield val

    def __len__(self):
        return self.num_elements


# class NeFCelebADatapipe(NeFDatapipe):
#     """A datapipe for loading NeF weights for CelebA dataset.

#     Args:
#         NeFDatapipe: The base datapipe class.
#     """

#     def __init__(self, path_iter: Any, shuffle_slice: bool = False, preload: bool = False):
#         super().__init__(path_iter, [PARAM_KEY], shuffle_slice, preload)

#         # load here all the attributes from the txt files.
