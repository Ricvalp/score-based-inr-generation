from flax.serialization import to_bytes, from_bytes
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from absl import app
from absl import logging
from ml_collections import config_flags
from pathlib import Path
import functools

from config import load_cfgs

import jax
import jax.numpy as jnp

from trainer import TrainerModulePixel
from score_based_model.utils import diffusion_coeff, marginal_prob_std
from dataset import numpy_collate



_CFG_FILE = config_flags.DEFINE_config_file("task", default="config/config.py:train")


def main(_):
    cfg = load_cfgs(_CFG_FILE)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=cfg.overwite_checkpoint)
    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.model_name}")
    vis_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Storing results in {checkpoint_dir}.")
    logging.info(f"Loaded config: {cfg}")

    dataset = MNIST(cfg.dataset.data_path, train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True) #, collate_fn=numpy_collate) #, num_workers=4)

    sigma =  cfg.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    trainer = TrainerModulePixel(
        data_loader,
        checkpoint_dir,
        marginal_prob_std_fn=marginal_prob_std_fn,
        lr=cfg.train.lr,
        wandb_log=cfg.wandb.wandb_log,
        seed=42,
    )

    trainer.train_model(cfg.train.num_epochs)


if __name__ == "__main__":
    app.run(main)