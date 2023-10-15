"""Separate file from utils.py to reduce dependency on e.g. torch or jax."""
from typing import List, Tuple

import numpy as np


def param_vector_to_list(
    param: np.ndarray, param_structure: List[Tuple[str, Tuple[int]]]
) -> List[np.ndarray]:
    """Converts a parameter vector into a list of parameters.

    Args:
        param: Parameter vector.
        param_structure: Structure of the parameter list.

    Returns:
        List of parameters.
    """
    param_list = []
    start_idx = 0
    if param.ndim == 1:
        param = param[None, :]

    for param_name, param_shape in param_structure:
        end_idx = start_idx + np.prod(param_shape)
        reshape_shape = [param.shape[0]] + param_shape
        param_list.append(param[:, start_idx:end_idx].reshape(reshape_shape))
        start_idx = end_idx
    return param_list


def param_vector_to_list_MFN(
    param: np.ndarray, param_structure: List[Tuple[str, Tuple[int]]]
) -> List[np.ndarray]:
    """Converts a parameter vector into a list of parameters.

    Args:
        param: Parameter vector.
        param_structure: Structure of the parameter list.

    Returns:
        List of parameters.
    """
    param_list = []
    start_idx = 0
    if param.ndim == 1:
        param = param[None, :]

    for param_name, param_shape in param_structure:
        end_idx = start_idx + np.prod(param_shape)
        reshape_shape = [param.shape[0]] + param_shape
        param_list.append(param[:, start_idx:end_idx].reshape(reshape_shape))
        start_idx = end_idx
    return param_list


def param_list_to_vector(
    param_list: List[np.ndarray],
) -> np.ndarray:
    """Converts a list of parameters into a vector.

    Args:
        param_list: Parameter list.

    Returns:
        Vector of parameters.
    """
    param = np.concatenate([p.flatten() for p in param_list]).flatten()

    return param
