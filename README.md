# 🖼️ **Score-Based Generation of INRs**

*Using score matching to generate implicit neural representations*

Creator: [Riccardo Valperga](https://twitter.com/RValperga).

[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)


##

A JAX-based implementation of score-based generative model adapted from [this](https://huggingface.co/flax-community/NeuralODE_SDE/blame/30104049eb731d33d91363c3277b68b53a7c7376/Score-SDE/sample-from-score-sde.py) implementation.

<!-- TABLE OF CONTENTS

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#driven pendulum">The driven pendulum example</a>
    </li>
    <li>
      <a href="#structure">Structure of the repo</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#running">Running</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details> -->

## Installation

Setup an environment with `python>=3.9` (earlier versions have not been tested).

The following packages are installed very differently based on the system configuration being used. For this reason, they are not included in the `requirements.txt` file:

- `jax >= 0.4.11`. See [Official Jax installation guide](https://github.com/google/jax#installation) for more details.
- `pytorch`. CPU version is suggested because it is only used for loading the data more. See [Official Pytorch installation guide](https://pytorch.org/get-started/locally/).

_After_ having installed the above packages, run:

```bash
pip install -r requirements.txt
```

These are all the packages used to run.


## Structure of the repo

The repository is structured as follows:

- `./config`. Configuration files for the tasks.
- `./dataset`. **Package** that generates the dataset.
- `./nefs`. **Package** of the models. It includes SIREN and MFNs.
- `./trainer` **Package** of the trainer model.
- `train_pixel.py`. Traines score-based model on pixels (MNIST).
- `sample_pixel.py`. Samples images from the trained score-based model.
- `train_nefs.py`. Traines score-based model on nefs (MNIST).
- `sample_nefs.py`. Samples nefs from the trained score-based model.

### Training

To train, run from the root folder run:

```bash
python train_pixels.py --task=config/pixels.py:train
```

```bash
python train_nefs.py --task=config/nefs.py:train --task.dataset.path = "path/to/dataset"
```

To sample using the last checkpoint, run:


```bash
python sample_pixels.py --task=config/pixels.py:sample_from_last
```

```bash
python sample_nefs.py --task=config/nefs.py:sample_from_last
```


To plot images from the INRs dataset run

```bash
python sample_nefs.py --task=config/nefs.py:sample_from_last --task.sample.sample_from_dataset=True
```


<!-- In particular:

- `-config.wandb.wanbb_log=False`: Wandb logging.

- `--config.dataset.train_lines=150`: with this, we use the first 150 lines in the `./pendulum_data` as training points.

- `--config.dataset.batch_size=150`: the traoning batch size. In this case it is full-batch.

- `--config.dataset.num_lines=200`: the total number of points in `./pendulum_data`. In this case 50 points will be used for evaluation.

- `--config.model.num_layers_flow=5 `: number of layers in the flow.

- `--config.model.num_layers=2`: number of layers in the MLP used to construct flow layers. -->

<!-- ### Testing

To test the trained model:

```bash
python test.py --config=config/config.py:test_last --config.model.num_layers_flow=5 --config.model.num_layers=2 --config.model.num_hidden=32 --config.model.d=1
```

Checkpoint from training are saved in `./checkpoints` using date and time as name. `-config=config/config.py:test_last` runs the most recent one. To run a specific one use, for example `-config=config/config.py:test_last --config.date_and_time = "2023-10-10_19-15-40"`. -->

<!-- ## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

[Riccardo Valperga](https://twitter.com/RValperga) -->
