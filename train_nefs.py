import functools
import json
import os
from functools import partial
from glob import glob
from pathlib import Path

import h5py
import torch
from absl import app, logging
from ml_collections import config_flags

import wandb
from config import load_cfgs
from dataset.nef_dataset import build_nef_data_loader_group
from dataset.nef_dataset.augmentations import combine_data
from score_based_model.utils import marginal_prob_std
from trainer import TrainerModuleNef

_CFG_FILE = config_flags.DEFINE_config_file("task", default="config/nefs.py:train")


def main(_):
    cfg = load_cfgs(_CFG_FILE)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=cfg.overwite_checkpoint)
    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.model_name}")
    vis_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Storing results in {checkpoint_dir}.")
    logging.info(f"Loaded config: {cfg}")

    wandb.init(project=cfg.wandb.project_name, config=cfg)

    data_config = cfg.dataset

    storage_folder = Path(data_config.path)
    nef_paths = glob(os.path.join(storage_folder, "*.hdf5"))
    with h5py.File(nef_paths[0], "r") as f:
        param_config = json.loads(f["param_config"][0].decode("utf-8"))

    collate_fn = partial(combine_data, combine_op=torch.stack)
    data_loaders = build_nef_data_loader_group(
        data_config,
        collate_fn=collate_fn,
        transform=None,
    )
    train_loader, _, _ = data_loaders

    sigma = cfg.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    trainer = TrainerModuleNef(
        train_loader,
        checkpoint_dir,
        marginal_prob_std_fn=marginal_prob_std_fn,
        lr=cfg.train.lr,
        wandb_log=cfg.wandb.wandb_log,
        param_config=param_config,
        seed=42,
    )

    trainer.train_model(cfg.train.num_epochs)


if __name__ == "__main__":
    app.run(main)
