import jax
import jax.numpy as jnp
import numpy as np
import torch
import functools
import matplotlib.pyplot as plt

from absl import app
from absl import logging
from ml_collections import ConfigDict, config_flags
from pathlib import Path
from functools import partial
import json
import h5py
from glob import glob
import os

from config import load_cfgs

from trainer import TrainerModuleNef

from config import load_cfgs

from dataset.nef_dataset import build_nef_data_loader_group
from dataset.nef_dataset.augmentations import combine_data

from score_based_model import diffusion_coeff, marginal_prob_std
from torchvision.utils import make_grid
from sampling import Euler_Maruyama_nef_sampler

from nefs import get_nef, unflatten_params


_CFG_FILE = config_flags.DEFINE_config_file("task", default="config/nefs.py:sample_from_last")

def main(_):
    cfg = load_cfgs(_CFG_FILE)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=cfg.overwite_checkpoint)
    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.model_name}")
    vis_folder.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loaded config: {cfg}")
    logging.info(f"Loading checkpoint from {checkpoint_dir}.")

    sigma =  cfg.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    data_config = cfg.dataset
    
    collate_fn = partial(combine_data, combine_op=torch.stack)
    data_loaders = build_nef_data_loader_group(
        data_config,
        collate_fn=collate_fn,
        transform=None,
    )
    train_loader, _, _ = data_loaders


    trainer = TrainerModuleNef(
        train_loader,
        checkpoint_dir,
        marginal_prob_std_fn=marginal_prob_std_fn,
        lr=cfg.train.lr,
        wandb_log=cfg.wandb.wandb_log,
        seed=42,
    )

    trainer.load_checkpoint()

    sample_batch_size = cfg.sample.sample_batch_size

    ## Sample visualization.
    storage_folder = Path(data_config.path)
    nef_cfg = ConfigDict(json.load(open(storage_folder / "nef.json")))
    nef_paths = glob(os.path.join(storage_folder, "*.hdf5"))
    with h5py.File(nef_paths[0], "r") as f:
        param_config = json.loads(f["param_config"][0].decode("utf-8"))

    # from the dataset
    if cfg.sample.sample_from_dataset:
        params = unflatten_params(param_config, next(iter(train_loader)).params)
        nef = get_nef(nef_cfg=nef_cfg)
        nef_fn = jax.vmap(nef.apply, in_axes=(0, None))    

    else:
        ## Generate samples using the specified sampler.
        rng = jax.random.PRNGKey(cfg.seed)
        nef_shape = next(iter(train_loader)).params.shape[1:]

        samples = Euler_Maruyama_nef_sampler(
            rng=rng,
            score_model=trainer.model,
            params=trainer.model_state.params,
            marginal_prob_std=marginal_prob_std_fn,
            diffusion_coeff=diffusion_coeff_fn,
            nef_shape=nef_shape,
            batch_size=sample_batch_size,
            num_steps=cfg.sample.sample_steps,
            )

        nef = get_nef(nef_cfg=nef_cfg)
        params = unflatten_params(param_config, samples)
        
        nef_fn = jax.vmap(nef.apply, in_axes=(0, None))

    # coordinates
    images_shape = cfg.dataset.image_shape
    
    x = jnp.linspace(-1, 1, images_shape[0])
    y = jnp.linspace(-1, 1, images_shape[1])
    x, y = jnp.meshgrid(x, y)
    coords = jnp.stack([x, y], axis=-1)
    coords = coords.reshape(images_shape[0] * images_shape[1], 2)

    # sampled nefs
    images = nef_fn({'params':params}, coords)

    images = jnp.clip(images, 0.0, 1.0)
    images = jnp.transpose(images.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
    images_grid = make_grid(torch.tensor(np.asarray(images)), nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(images_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(vis_folder / "nef_samples.png")
    plt.show()

if __name__ == "__main__":
    app.run(main)