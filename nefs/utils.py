from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import core, dtypes, random
from jax.nn.initializers import Initializer
from jax.random import KeyArray

import flax
from flax.core.frozen_dict import FrozenDict

import numpy as np

from ml_collections import ConfigDict

import nefs

from pathlib import Path
from typing import Any, Sequence, Tuple, Union


def get_nef(nef_cfg: ConfigDict) -> flax.linen.Module:
    """Returns the model for the given config.

    Args:
        nef_cfg (ConfigDict): The config for the model.

    Raises:
        NotImplementedError: If the model is not implemented.

    Returns:
        flax.linen.Module: The model.
    """
    if nef_cfg.name not in dir(nefs):
        raise NotImplementedError(
            f"Model {nef_cfg.name} not implemented. Available are: {dir(nefs)}"
        )
    else:
        model = getattr(nefs, nef_cfg.name)
        return model(**nef_cfg.params)


def flatten_dict(d: Dict, separation: str = "."):
    """Flattens a dictionary.

    Args:
        d (Dict): The dictionary to flatten.

    Returns:
        Dict: The flattened dictionary.
    """
    flat_d = {}
    for key, value in d.items():
        if isinstance(value, (dict, FrozenDict)):
            sub_dict = flatten_dict(value)
            for sub_key, sub_value in sub_dict.items():
                flat_d[key + separation + sub_key] = sub_value
        else:
            flat_d[key] = value
    return flat_d


def unflatten_dict(d: Dict, separation: str = "."):
    """Unflattens a dictionary, inverse to flatten_dict.

    Args:
        d (Dict): The dictionary to unflatten.
        separation (str, optional): The separation character. Defaults to ".".

    Returns:
        Dict: The unflattened dictionary.
    """
    unflat_d = {}
    for key, value in d.items():
        if separation in key:
            sub_keys = key.split(separation)
            sub_dict = unflat_d
            for sub_key in sub_keys[:-1]:
                if sub_key not in sub_dict:
                    sub_dict[sub_key] = {}
                sub_dict = sub_dict[sub_key]
            sub_dict[sub_keys[-1]] = value
        else:
            unflat_d[key] = value
    return unflat_d


def flatten_params(params: Any, num_batch_dims: int = 0):
    """Flattens the parameters of the model.

    Args:
        params (jax.PyTree): The parameters of the model.
        num_batch_dims (int, optional): The number of batch dimensions. Tensors will not be flattened over these dimensions. Defaults to 0.

    Returns:
        List[Tuple[str, List[int]]]: Structure of the flattened parameters.
        jnp.ndarray: The flattened parameters.
    """
    flat_params = flatten_dict(params)
    keys = sorted(list(flat_params.keys()))
    param_config = [(k, flat_params[k].shape[num_batch_dims:]) for k in keys]
    comb_params = jnp.concatenate(
        [flat_params[k].reshape(*flat_params[k].shape[:num_batch_dims], -1) for k in keys], axis=-1
    )
    return param_config, comb_params


def unflatten_params(
    param_config: List[Tuple[str, List[int]]],
    comb_params: jnp.ndarray,
):
    """Unflattens the parameters of the model.

    Args:
        param_config (List[Tuple[str, List[int]]]): Structure of the flattened parameters.
        comb_params (jnp.ndarray): The flattened parameters.

    Returns:
        jax.PyTree: The parameters of the model.
    """
    params = []
    key_dict = {}
    idx = 0
    for key, shape in param_config:
        params.append(
            comb_params[..., idx : idx + np.prod(shape)].reshape(*comb_params.shape[:-1], *shape)
        )
        key_dict[key] = 0
        idx += np.prod(shape)
    key_dict = unflatten_dict(key_dict)
    return FrozenDict(jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(key_dict), params))



def store_cfg(cfg: ConfigDict, storage_folder: Path, cfg_name: str = "cfg.json"):
    """Store the essential information to be able to load the neural field.

    The neural field depends on the information that comes from nef_cfg
    """
    # ml_collections ConfigDict to JSON
    cfg_s = cfg.to_json()
    # nef_cfg_s = json.dumps(nef_cfg)
    cfg_path = storage_folder / Path(cfg_name)
    # Check if the file exists. If cfgs don't match, raise an error
    if cfg_path.exists():
        old_cfg = cfg_path.read_text()
        if cfg_s != old_cfg:
            raise RuntimeError(
                f"You are saving to the same folder as an older, different run."
                f"If you know what you are doing, delete {cfg_path} before proceeding."
                f"The configuration currently in {cfg_path}:\n"
                f"{old_cfg}\n"
                f"The configuration you are trying to save:\n"
                f"{cfg_s}"
            )
        else:
            return
    else:
        # store the json file
        cfg_path.write_text(cfg_s)


def load_model_cfg(storage_folder: Path):
    """Store the essential information to be able to load the neural field.

    The neural field depends on the information that comes from nef_cfg
    """
    nef_cfg_path = storage_folder / Path("nef_cfg.json")
    nef_cfg = ConfigDict.from_json(nef_cfg_path.read_text())

    return nef_cfg