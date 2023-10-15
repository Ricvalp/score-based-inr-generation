import os
from pathlib import Path
from datetime import datetime
from typing import Literal, Type

from absl import logging
from ml_collections import ConfigDict


def get_config(mode: Literal["train", "test_last", "test_specific"] = None):
    if mode is None:
        mode = "train"
        logging.info(f"No mode provided, using '{mode}' as default")


    cfg = ConfigDict()
    
    cfg.sigma = 25

    cfg.model_name = "model"
    cfg.checkpoint_dir = (
        Path("./checkpoints")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        / Path(cfg.model_name)
    )

    cfg.overwite_checkpoint = True
    cfg.seed = 45

    cfg.wandb = ConfigDict()
    cfg.wandb.wandb_log = False
    cfg.wandb.project_name = "score-based-model-inr"

    cfg.dataset = ConfigDict()
    cfg.dataset.dataset_name = "mnist"
    cfg.dataset.data_path = Path("./data") / Path(cfg.dataset.dataset_name)
    cfg.dataset.batch_size = 256

    cfg.train = ConfigDict()
    cfg.train.lr = 0.001
    cfg.train.num_epochs = 50

    cfg.experiment_dir = Path("visualizations") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    cfg.sample = ConfigDict()
    cfg.sample.sample_batch_size = 64
    cfg.sample.sample_steps = 500


    if mode == "sample_from_last":
        cfg.dataset.batch_size = cfg.sample.sample_batch_size
        date_and_time = max(os.listdir(Path("./checkpoints")))
        cfg.checkpoint_dir = (
            Path("./checkpoints") / date_and_time / Path(cfg.model_name)
        )

    if mode == "sample_from_specific":
        cfg.dataset.batch_size = cfg.sample.sample_batch_size
        cfg.date_and_time = "2023-10-10_19-15-40"
        cfg.checkpoint_dir = (
            Path("./checkpoints")
            / Path(cfg.date_and_time)
            / Path(cfg.model_name)
        )
    
    
    

    logging.debug(f"Loaded config: {cfg}")

    return cfg
