import jax
import jax.numpy as jnp
import numpy as np
import torch
import functools
import matplotlib.pyplot as plt


from absl import app
from absl import logging
from ml_collections import config_flags
from pathlib import Path

from config import load_cfgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from trainer import TrainerModulePixel
from dataset import numpy_collate


from score_based_model import diffusion_coeff, marginal_prob_std
from torchvision.utils import make_grid
from sampling import Euler_Maruyama_image_sampler


_CFG_FILE = config_flags.DEFINE_config_file("task", default="config/pixels.py:sample_from_last")

def main(_):
    cfg = load_cfgs(_CFG_FILE)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=cfg.overwite_checkpoint)
    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.model_name}")
    vis_folder.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loaded config: {cfg}")
    logging.info(f"Loading checkpoint from {checkpoint_dir}.")

    dataset = MNIST(cfg.dataset.data_path, train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True) #, collate_fn=numpy_collate) #, num_workers=4)

    sigma =  cfg.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    trainer = TrainerModulePixel(
        data_loader,
        checkpoint_dir,
        marginal_prob_std_fn=marginal_prob_std_fn,
        lr=cfg.train.lr,
        wandb_log=cfg.wandb.wandb_log,
        seed=42,
    )

    trainer.load_checkpoint()

    sample_batch_size = cfg.sample.sample_batch_size

    ## Generate samples using the specified sampler.
    rng = jax.random.PRNGKey(cfg.seed)
    samples = Euler_Maruyama_image_sampler(
        rng,
        trainer.model,
        trainer.model_state.params,
        marginal_prob_std_fn,
        diffusion_coeff_fn, 
        sample_batch_size
        )

    ## Sample visualization.
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = jnp.transpose(samples.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
    sample_grid = make_grid(torch.tensor(np.asarray(samples)), nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(vis_folder / Path(f"pixel_sample.png"))
    plt.show()


if __name__ == "__main__":
    app.run(main)