import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from absl import app, flags, logging
from ml_collections import ConfigDict

from dataset.nef_dataset import build_nef_data_loader_group
from dataset.nef_dataset.utils import splits_to_names

from dataset.nef_dataset.utils import collate_data, get_param_keys, get_param_config
from dataset.nef_dataset.param_utils import param_vector_to_list, param_list_to_vector

from functools import partial

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "ImageNet", "Dataset to use")
flags.DEFINE_string("path", "saved_models/best_psnr/CIFAR10/SIREN", "Path to dataset")
flags.DEFINE_string(
    "data_pipe_class", "NeFDatapipe", "Class to use for the data pipe"
)
flags.DEFINE_list("split", [40000, 10000, 10000], "splits present in the dataset")
flags.DEFINE_integer("batch_size", 128, "Batch size to use in the data loaders")
flags.DEFINE_integer("num_workers", 0, "Number of workers for each dataset")
flags.DEFINE_string("data_prefix", "", "Prefix for the nef hdf5 files")


def name_to_split(split_name: str) -> Tuple[int, int]:
    start_idx, end_idx = split_name.split("_")
    return int(start_idx), int(end_idx)


def compute_mean_std_for_nef_dataset(cfg_dataset: ConfigDict):
    logging.info(cfg_dataset)
    collate_fn = partial(collate_data, combine_op=np.stack)
    loaders = build_nef_data_loader_group(
        cfg_dataset, 
        collate_fn=collate_fn
        )

    metadata_file = Path(cfg_dataset.path) / Path("metadata.json")
    if metadata_file.exists():
        # if file is empty, do nothing
        if metadata_file.stat().st_size == 0:
            metadata = ConfigDict()
        else:
            metadata = ConfigDict(json.load(open(metadata_file)))
    else:
        metadata = ConfigDict()

    means = {}
    mean_squares = {}
    sizes = {}

    full_means = {}
    full_stds = {}

    all_split_names = splits_to_names(cfg_dataset.split)
    param_keys = get_param_keys(cfg_dataset)
    param_structure = get_param_config(cfg_dataset)

    logging.debug(all_split_names)
    logging.debug([name_to_split(x) for x in all_split_names])

    for loader, split_name in zip(loaders, all_split_names):
        logging.debug(f"Split: {split_name}")
        start_idx, end_idx = name_to_split(split_name)
        if split_name in metadata:
            logging.info(
                f"Skipping calculation of mean and std for split `{split_name}` because it is already in metadata."
            )
            continue

        logging.debug(f"Size: {end_idx - start_idx}")
        means[split_name] = {}
        mean_squares[split_name] = {}
        sizes[split_name] = {}
        for batch in loader:
            nef_params = batch[0]

            nef_params_list = param_vector_to_list(param=nef_params, param_structure=param_structure)

            for param_key, param in zip(param_keys, nef_params_list):
                if param_key not in means[split_name]:
                    means[split_name][param_key] = []
                    mean_squares[split_name][param_key] = []
                    sizes[split_name][param_key] = []

                means[split_name][param_key].append(np.mean(param))
                mean_squares[split_name][param_key].append(np.mean(param**2))
                sizes[split_name][param_key].append(param.shape[0])

            del nef_params
        logging.info(f"Done with split {split_name}")

        full_means[split_name] = []
        full_stds[split_name] = []

        for param_key, param_shape in param_structure:
            mean = np.full(param_shape, np.average(means[split_name][param_key], weights=sizes[split_name][param_key]))
            std = np.full(param_shape, np.sqrt(np.average(mean_squares[split_name][param_key], weights=sizes[split_name][param_key]) - mean**2))
            full_means[split_name].append(mean)
            full_stds[split_name].append(std)

        metadata[split_name] = {
            "mean": param_list_to_vector(full_means[split_name]).tolist(),
            "std": param_list_to_vector(full_stds[split_name]).tolist(),
        }
        

        # metadata[split_name] = {
        #     "mean": np.average(means[split_name], axis=0, weights=sizes[split_name]).tolist(),
        #     "std": np.sqrt(
        #         np.average(mean_squares[split_name], axis=0, weights=sizes[split_name])
        #         - np.average(means[split_name], axis=0, weights=sizes[split_name]) ** 2
        #     ).tolist(),
        #     "size": sizes[split_name][0],
        # }

    metadata_file.write_text(metadata.to_json())


def main(_):
    dataset_cfg = {
        "path": FLAGS.path,
        "data_pipe_class": FLAGS.data_pipe_class,
        "split": [int(x) for x in FLAGS.split],
        "batch_size": FLAGS.batch_size,
        "num_workers": FLAGS.num_workers,
        "seed": 42,
        "persistent_workers": True,
        "preload": False,
        "shuffle_slice": False,
        "data_prefix": FLAGS.data_prefix,
    }

    cfg_dataset = ConfigDict(dataset_cfg)

    compute_mean_std_for_nef_dataset(cfg_dataset)


if __name__ == "__main__":
    app.run(main)
