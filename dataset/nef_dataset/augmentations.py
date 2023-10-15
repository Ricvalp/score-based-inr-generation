import re
from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

Number = Union[int, float, np.ndarray]

ParametersList = List[np.ndarray]
ParameterVector = np.ndarray

try:
    import jax
    import jax.numpy as jnp

    Number = Union[Number, jnp.ndarray]

    ParametersList = Union[ParametersList, List[jnp.ndarray]]
    ParameterVector = Union[ParameterVector, jnp.ndarray]

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
try:
    import torch

    # add torch.Tensor to Number typing
    Number = Union[Number, torch.Tensor]

    ParametersList = Union[ParametersList, List[torch.Tensor]]
    ParameterVector = Union[ParameterVector, torch.Tensor]

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ParameterAny = Union[ParametersList, ParameterVector]

from dataset.nef_dataset.param_utils import param_list_to_vector, param_vector_to_list

AVAILABLE_AUGMENTATIONS = [
    "Compose",
    "ParametersToList",
    "ListToParameters",
    "Normalize",
    "RandomMLPWeightPermutation",
    "RandomQuantileWeightDropout",
    "RandomDropout",
    "RandomGaussianNoise",
    "RandomRotate",
    "RandomScale",
    "RandomTranslateMFN",
]


def select_filters(param_keys):
    return [x for x in param_keys if re.search(r"\S*filter\S*", x) is not None]


def select_kernels(param_keys):
    return [x for x in param_keys if re.search(r"\S*kernel\S*", x) is not None]


def select_bias(param_keys):
    return [x for x in param_keys if re.search(r"\S*bias\S*", x) if not None]

def select_first_layer_SIREN(param_keys):
    return [x for x in param_keys if re.search(r"\S*0\S*.kernel", x) is not None]

def complement_list(param_keys: List[str], include_list: List[str]) -> List[str]:
    """Takes the complement of include_list with respect to universe defined by param_keys.

    Args:
        param_keys: list of strings
        include_list: list of strings

    Returns:
        list of strings
    """
    return [x for x in param_keys if x not in include_list]


class Data:
    def __init__(self, params: ParameterAny, labels: Number = None, batched=True, **kwargs):
        self.key_list = []

        self.params = params

        self.batched = batched

        if labels is not None:
            self.labels = labels

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise AttributeError(f"Object has no attribute {key}")

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name not in self.key_list and name != "key_list" and name != "batched":
            self.key_list.append(name)

    def keys(self):
        return self.key_list

    def slice_data(self, start_idx, end_idx):
        newdata = Data(params=None, labels=None, batched=self.batched)
        for key in self.keys():
            if hasattr(self, key):
                setattr(newdata, key, getattr(self, key)[start_idx:end_idx])
        return newdata

    def __add__(self, other):
        if isinstance(other, Data):
            for key in set(self.keys()).union(set(other.keys())):
                if hasattr(self, key) and hasattr(other, key):
                    if isinstance(getattr(self, key), np.ndarray):
                        if len(getattr(self, key).shape) == 1:
                            setattr(self, key, getattr(self, key)[np.newaxis, ...])

                        if len(getattr(other, key).shape) == 1:
                            setattr(other, key, getattr(other, key)[np.newaxis, ...])

                        setattr(
                            self,
                            key,
                            np.concatenate([getattr(self, key), getattr(other, key)], axis=0),
                        )
                    elif JAX_AVAILABLE:
                        if isinstance(getattr(self, key), jnp.ndarray):
                            if len(getattr(self, key).shape) == 1:
                                setattr(self, key, getattr(self, key)[jnp.newaxis, ...])

                            if len(getattr(other, key).shape) == 1:
                                setattr(other, key, getattr(other, key)[jnp.newaxis, ...])

                            setattr(
                                self,
                                key,
                                jnp.concatenate([getattr(self, key), getattr(other, key)], axis=0),
                            )
                    elif TORCH_AVAILABLE:
                        if isinstance(getattr(self, key), torch.Tensor):
                            if len(getattr(self, key).shape) == 1:
                                setattr(self, key, getattr(self, key)[torch.newaxis, ...])

                            if len(getattr(other, key).shape) == 1:
                                setattr(other, key, getattr(other, key)[torch.newaxis, ...])

                            setattr(
                                self,
                                key,
                                torch.concatenate(
                                    [getattr(self, key), getattr(other, key)], dim=0
                                ),
                            )
                    elif isinstance(getattr(self, key), list):
                        setattr(self, key, getattr(self, key) + getattr(other, key))
                    else:
                        raise NotImplementedError
                elif hasattr(self, key):
                    pass
                elif hasattr(other, key):
                    setattr(self, key, getattr(other, key))
        else:
            raise NotImplementedError

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]):
        return cls(**data_dict)

    def add_list(self, other_list):
        for other in other_list:
            self += other


def collate_data_multi_seed(data_list, combine_op=np.concatenate):
    """Combines a list of Data objects into a single Data object.

    Args:
        data_list: List of Data objects to combine.

    Returns:
        Combined Data object.
    """

    out = []
    for key in data_list[0].keys():
        combined = torch.split(combine_op([data[key] for data in data_list]), 1, dim=1)
        combined_reshaped = torch.cat(combined, dim=0).squeeze(1)
        out.append(combined_reshaped)

    return out

def collate_data(data_list, combine_op=np.concatenate):
    """Combines a list of Data objects into a single Data object.

    Args:
        data_list: List of Data objects to combine.

    Returns:
        Combined Data object.
    """

    out = []
    for key in data_list[0].keys():
        combined = combine_op([data[key] for data in data_list])
        out.append(combined)

    return out


def combine_data(data_list, combine_op=np.concatenate):
    """Combines a list of Data objects into a single Data object.

    Args:
        data_list: List of Data objects to combine.

    Returns:
        Combined Data object.
    """

    newdata = Data(params=None, labels=None, batched=True)
    for key in data_list[0].keys():
        newdata[key] = jnp.array(combine_op([data[key] for data in data_list]))

    return newdata


class Augmentation(ABC):
    """Abstract base class for augmentations."""

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class Compose(Augmentation):
    """Class for composing augmentations.

    Args:
        augmentations: List of augmentations to compose.
    """

    def __init__(self, augmentations: List[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, x: Data) -> Data:
        for augmentation in self.augmentations:
            x = augmentation(x)
        return x

    def __repr__(self) -> str:
        return f"Compose(augmentations={self.augmentations})"


class Identity(Augmentation):
    """Identity augmentation."""

    def __init__(self):
        super().__init__()

    def __call__(self, x: Data) -> Data:
        return x

    def __repr__(self) -> str:
        return "Identity()"


class ContrastiveTransformation(Augmentation):
    def __init__(self, base_transforms, batch_size, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        self.batch_size = batch_size

    def __call__(self, x: Data):
        augmented_views = []
        for i in range(self.n_views):
            x_aug = x.slice_data(i * self.batch_size, (i + 1) * self.batch_size)
            augmented_views.append(self.base_transforms(x_aug))

        return combine_data(augmented_views, combine_op=lambda x: torch.cat(x, dim=0))


class ToPyTorch(Augmentation):
    def __init__(self):
        super().__init__()

    def transform(self, params: Number):
        return torch.from_numpy(params)

    def __call__(self, x):
        x.params = self.transform(x.params)

        return x


class ParametersToList(Augmentation):
    """Converts a parameter vector into a list of parameters.

    Args:
        param_structure: Structure of the parameter list. For example, the one saved
            along with the NeF dataset.
    """

    def __init__(self, param_structure: List[Tuple[str, Tuple[int]]]):
        super().__init__()
        self.param_structure = param_structure

    def transform(self, params: Number):
        return param_vector_to_list(params, self.param_structure)

    def __call__(self, x: Data):
        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return f"ParametersToList(param_structure={self.param_structure})"
    
class ParametersToListMFN(Augmentation):
    """Converts a parameter vector into a list of parameters.

    Args:
        param_structure: Structure of the parameter list. For example, the one saved
            along with the NeF dataset.
    """

    def __init__(self, param_structure: List[Tuple[str, Tuple[int]]]):
        super().__init__()
        self.param_structure = param_structure
        self.param_keys = []
        for key, _ in param_structure:
            self.param_keys.append(key)

        def sorting_order(param_tuple):
            param_key, param_shape = param_tuple
            if param_key.startswith("output_linear."):
                return np.iinfo(np.int32).max
            return int(param_key.split(".")[0].split("_")[-1])
    
        #self.param_keys = sorted(self.param_keys, key=sorting_order)
        self.param_structure = sorted(self.param_structure, key=sorting_order)

    def transform(self, params: ParametersList):

        return param_vector_to_list(params, self.param_structure)

    def __call__(self, x: Data):
        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return f"ParametersToList(param_structure={self.param_structure})"


class Normalize(Augmentation):
    """Normalizes the input by subtracting the mean and dividing by the standard deviation.

    Args:
        mean: Mean to subtract. Can be a float, numpy array, torch tensor, or jax array.
        std: Standard deviation to divide by. Can be a float, numpy array, torch tensor, or jax array.
    """

    def __init__(
        self,
        mean: Number,
        std: Number,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

    def transform(self, params: ParameterVector):
        return (params - self.mean) / self.std

    def __call__(self, x: Data) -> Data:
        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"
    



class TensorAugmentation(Augmentation, ABC):
    """Base augmentation class for operations on tensors. Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def __init__(
        self, platform: str = "numpy", seed: Optional[Any] = None, device=None, batch_size: int = 1
    ):
        self.batch_size = batch_size
        self.platform = platform
        self.seed = seed
        self.device = device

        if hasattr(self, "transform"):
            # If transform is already defined, then we will ignore the platform and seed
            pass
        elif platform == "numpy":
            self._setup_numpy()
        elif platform == "jax":
            self._setup_jax()
        elif platform == "pytorch" or platform == "torch":
            self._setup_pytorch()
        elif platform == "auto":
            self._setup_auto()
        else:
            raise NotImplementedError(f"Platform {platform} not implemented")

    def _setup_numpy(self):
        self._set_transform(self.numpy_transform)
        if self.seed is None:
            self.rng = np.random.random.__self__  # Default global rng
        elif isinstance(self.seed, int):
            self.rng = np.random.default_rng(self.seed)
        elif isinstance(self.seed, np.random.Generator):
            self.rng = self.seed
        else:
            raise ValueError(f"Invalid seed {self.seed}")

    def _setup_jax(self):
        self._set_transform(self.jax_transform)
        if self.seed is None:
            self.rng = jax.random.PRNGKey(0)
        elif isinstance(self.seed, int):
            self.rng = jax.random.PRNGKey(self.seed)
        elif isinstance(self.seed, jax.random.PRNGKeyArray):
            self.rng = self.seed
        else:
            raise ValueError(f"Invalid seed {self.seed}")

    def _setup_pytorch(self):
        self._set_transform(self.pytorch_transform)
        if self.seed is None:
            self.rng = torch.Generator(device=self.device).manual_seed(torch.seed())
        elif isinstance(self.seed, int):
            self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        elif isinstance(self.seed, torch.Generator):
            self.rng = self.seed
        else:
            raise ValueError(f"Invalid seed {self.seed}")

    def _setup_auto(self):
        self._set_transform(self.auto_transform)

    def _set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def numpy_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    @abstractmethod
    def jax_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    @abstractmethod
    def pytorch_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    def auto_transform(self, x: ParameterAny) -> ParameterAny:
        if isinstance(x, np.ndarray):
            self._setup_numpy()
        elif JAX_AVAILABLE:
            if isinstance(x, jnp.ndarray):
                self._setup_jax()
        elif TORCH_AVAILABLE:
            if isinstance(x, torch.Tensor):
                self._setup_pytorch()
        else:
            raise NotImplementedError(
                f"Auto transform failed to find a suitable transformation for type {type(x)}. Platforms available: numpy{', jax' if JAX_AVAILABLE else ', (unable to import jax)'}{', pytorch' if TORCH_AVAILABLE else ', (unable to import pytorch)'}."
            )
        return self.transform(x)

    def __call__(self, x: Data) -> Data:
        """By default, assume that we want to apply the transform only to the parameters."""
        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return f"Augmentation(platform={self.platform})"

    def __str__(self):
        return repr(self)


class ListToParameters(TensorAugmentation):
    """Converts a parameter list into a parameter vector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pytorch_transform(self, params: ParametersList) -> ParameterVector:
        if not self.batched:
            return torch.cat([x.flatten() for x in params], dim=0).flatten()
        else:
            return torch.cat([torch.flatten(x, start_dim=1) for x in params], dim=1)

    def numpy_transform(self, params: ParametersList) -> ParameterVector:
        if not self.batched:
            return np.concatenate([x.flatten() for x in params], axis=0).flatten()
        else:
            return np.concatenate([np.reshape(x, (x.shape[0], -1)) for x in params], axis=1)

    def jax_transform(self, params: ParametersList) -> ParameterVector:
        if not self.batched:
            return jnp.concatenate([x.flatten() for x in params], axis=0).flatten()
        else:
            return jnp.concatenate([jnp.reshape(x, (x.shape[0], -1)) for x in params], axis=1)

    def __call__(self, x: Data) -> Data:
        self.batched = x.batched

        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return "ListToParameters()"


class JointParameterAugmentation(TensorAugmentation, ABC):
    """Base augmentation class for operations on joint parameters, i.e. all parameters being
    augmented together. Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. May not be optional for some augmentations.
    """

    def __init__(
        self,
        platform: str = "numpy",
        seed: Optional[Any] = None,
        param_keys: Optional[List[str]] = None,
        batch_size: int = 1,
        device=None,
    ):
        super().__init__(platform=platform, seed=seed, batch_size=batch_size, device=device)
        self.param_keys = param_keys

    @abstractmethod
    def numpy_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    @abstractmethod
    def jax_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    @abstractmethod
    def pytorch_transform(self, x: ParameterAny) -> ParameterAny:
        raise NotImplementedError

    def __call__(self, x: Data) -> Data:
        """Performs augmentation by passing all parameters. The `transform` function is expecting
        all the parameters of the model at once. This could be a list of parameters or a single
        parameter vector with all the weights and biases flattened together.

        Returns:
            Augmented input.
        """
        self.batched = x.batched

        x.params = self.transform(x.params)

        return x

    def __repr__(self):
        return f"JointParameterAugmentation(platform={self.platform})"


class RandomMLPWeightPermutation(JointParameterAugmentation):
    """Randomly permutes the weights of the network without changing the network structure and
    outputs.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def pre_transform(self, x, transform_func):
        assert isinstance(x, (list)), "Input must be a list of parameters"
        x = copy(x)  # Copy list to avoid modifying original
        idxs = [i for i in range(len(x)) if self.param_keys[i].endswith(".kernel")]
        for layer_idx1, layer_idx2 in zip(idxs[:-1], idxs[1:]):
            bias_idx1 = self.param_keys.index(
                self.param_keys[layer_idx1].replace(".kernel", ".bias")
            )
            output_dim_permute = [x[layer_idx1], x[bias_idx1]]
            input_dim_permute = [x[layer_idx2]]
            assert (
                output_dim_permute[0].shape[-1] == output_dim_permute[1].shape[-1]
            ), f"Output dimensions must match: {[op.shape for op in output_dim_permute]}"
            assert (
                output_dim_permute[0].shape[-1] == input_dim_permute[0].shape[-2]
            ), "Input and output dimensions must match"
            new_layers = transform_func(
                output_dim_permute=output_dim_permute, input_dim_permute=input_dim_permute
            )
            x[layer_idx1] = new_layers[0]
            x[bias_idx1] = new_layers[1]
            x[layer_idx2] = new_layers[2]
        return x

    def _set_transform(self, transform):
        self.transform = lambda x: self.pre_transform(x, transform)

    def numpy_transform(self, output_dim_permute, input_dim_permute):
        # TODO: Consider supporting different permutation per batch element
        perm = self.rng.permutation(output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ]

    def jax_transform(self, output_dim_permute, input_dim_permute):
        self.rng, perm_rng = jax.random.split(self.rng)
        perm = jax.random.permutation(perm_rng, output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ]

    def pytorch_transform(self, output_dim_permute, input_dim_permute):
        perm = torch.randperm(output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ]

def key_sorting_params(param):
    if param.startswith("output_linear."):
        return np.iinfo(np.int32).max
    return int(param.split(".")[0].split("_")[-1])

class RandomFourierNetPermutation(JointParameterAugmentation):
    """Randomly permutes the weights of the network without changing the network structure and
    outputs.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def pre_transform(self, x, transform_func):
        assert isinstance(x, (list)), "Input must be a list of parameters"
        y = x  # Copy list to avoid modifying original

        idxs_filter = [i for i in range(len(y)) if self.param_keys[i].startswith("filters")]
        idxs_filter_kernel = [i for i in idxs_filter if self.param_keys[i].endswith(".kernel")]
        idxs_filter_bias = [i for i in idxs_filter if self.param_keys[i].endswith(".bias")]
        number_filter = [key_sorting_params(self.param_keys[i]) for i in idxs_filter_kernel]
        sorted_filters = np.argsort(number_filter)

        idxs_linears = [i for i in range(len(y)) if self.param_keys[i].startswith("linears")]+[i for i in range(len(x)) if self.param_keys[i].startswith("output_linear")]
        idxs_linears_kernel = [i for i in idxs_linears if self.param_keys[i].endswith(".kernel")]
        idxs_linears_bias = [i for i in idxs_linears if self.param_keys[i].endswith(".bias")]
        number_linear = [key_sorting_params(self.param_keys[i]) for i in idxs_linears_kernel]
        sorted_linears = np.argsort(number_linear)

        perm = torch.stack([torch.randperm(y[idxs_filter_kernel[sorted_filters[0]]].shape[-1]) for _ in range(y[idxs_filter_kernel[sorted_filters[0]]].shape[0])] , dim=0).unsqueeze(1)

        y[idxs_filter_kernel[sorted_filters[0]]] = self.permute_rows(y[idxs_filter_kernel[sorted_filters[0]]], perm)
        y[idxs_filter_bias[sorted_filters[0]]] = self.permute_rows_biases(y[idxs_filter_bias[sorted_filters[0]]], perm)
        y[idxs_linears_kernel[sorted_linears[0]]] = self.permute_cols(y[idxs_linears_kernel[sorted_linears[0]]], perm)

        for i in range(len(sorted_filters)-1):
            perm = torch.stack([torch.randperm(y[idxs_filter_kernel[sorted_filters[0]]].shape[-1]) for _ in range(y[idxs_filter_kernel[sorted_filters[0]]].shape[0])] , dim=0).unsqueeze(1)
            y[idxs_linears_kernel[sorted_linears[i+1]]] = self.permute_cols(y[idxs_linears_kernel[sorted_linears[i+1]]], perm)
            y[idxs_linears_kernel[sorted_linears[i]]] = self.permute_rows(y[idxs_linears_kernel[sorted_linears[i]]], perm)
            y[idxs_linears_bias[sorted_linears[i]]] = self.permute_rows_biases(y[idxs_linears_bias[sorted_linears[i]]], perm)
            y[idxs_filter_kernel[sorted_filters[i+1]]] = self.permute_rows(y[idxs_filter_kernel[sorted_filters[i+1]]], perm)
            y[idxs_filter_bias[sorted_filters[i+1]]] = self.permute_rows_biases(y[idxs_filter_bias[sorted_filters[i+1]]], perm)

        return y

    def _set_transform(self, transform):
        self.transform = lambda x: self.pre_transform(x, transform)

    def numpy_transform(self, row_permute, column_permute):
        # TODO: Consider supporting different permutation per batch element
        perm = self.rng.permutation(row_permute[0].shape[-1])
        return [op[..., perm] for op in row_permute] + [
            ip[..., perm, :] for ip in column_permute
        ]

    def jax_transform(self, row_permute, column_permute):
        self.rng, perm_rng = jax.random.split(self.rng)
        perm = jax.random.permutation(perm_rng, row_permute[0].shape[-1])
        return [op[..., perm] for op in row_permute] + [
            ip[..., perm, :] for ip in column_permute
        ]
    
    def permute_rows(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1), torch.arange(x.shape[1]).unsqueeze(1), perm]
    
    def permute_cols(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1), perm.squeeze()]
    
    def permute_rows_biases(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1), perm].squeeze(1)

    def pytorch_transform(self, row_permute, column_permute, arange_for_sorting_batch,):
        perm = torch.stack([torch.randperm(row_permute[0].shape[-1]) for _ in range(row_permute[0].shape[0])] , dim=0).unsqueeze(1)

        return [
            row_permute[0][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), torch.arange(6).unsqueeze(1), perm],
            row_permute[1][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), perm].squeeze(1),
            row_permute[2][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), torch.arange(2).unsqueeze(1), perm],
            row_permute[3][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), perm].squeeze(1),
            column_permute[0][arange_for_sorting_batch.unsqueeze(1), perm.squeeze()]
        ]

class RandomFourierNetMixup(JointParameterAugmentation):
    """Randomly permutes the weights of the network without changing the network structure and
    outputs.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def pre_transform(self, x, transform_func):
        assert isinstance(x, (list)), "Input must be a list of parameters"
        y = copy(x)  # Copy list to avoid modifying original

        idxs_filter = [i for i in range(len(y)) if self.param_keys[i].startswith("filters")]
        idxs_filter_kernel = [i for i in idxs_filter if self.param_keys[i].endswith(".kernel")]
        idxs_filter_bias = [i for i in idxs_filter if self.param_keys[i].endswith(".bias")]
        number_filter = [key_sorting_params(self.param_keys[i]) for i in idxs_filter_kernel]
        sorted_filters = np.argsort(number_filter)

        idxs_linears = [i for i in range(len(y)) if self.param_keys[i].startswith("linears")]+[i for i in range(len(x)) if self.param_keys[i].startswith("output_linear")]
        idxs_linears_kernel = [i for i in idxs_linears if self.param_keys[i].endswith(".kernel")]
        idxs_linears_bias = [i for i in idxs_linears if self.param_keys[i].endswith(".bias")]
        number_linear = [key_sorting_params(self.param_keys[i]) for i in idxs_linears_kernel]
        sorted_linears = np.argsort(number_linear)

        perm = torch.stack([torch.randperm(y[idxs_filter_kernel[sorted_filters[0]]].shape[-1]) for _ in range(y[idxs_filter_kernel[sorted_filters[0]]].shape[0])] , dim=0).unsqueeze(1)

        y[idxs_filter_kernel[sorted_filters[0]]] = self.permute_rows(y[idxs_filter_kernel[sorted_filters[0]]], perm)
        y[idxs_filter_bias[sorted_filters[0]]] = self.permute_rows_biases(y[idxs_filter_bias[sorted_filters[0]]], perm)
        y[idxs_linears_kernel[sorted_linears[0]]] = self.permute_cols(y[idxs_linears_kernel[sorted_linears[0]]], perm)

        for i in range(len(sorted_filters)-1):
            perm = torch.stack([torch.randperm(y[idxs_filter_kernel[sorted_filters[0]]].shape[-1]) for _ in range(y[idxs_filter_kernel[sorted_filters[0]]].shape[0])] , dim=0).unsqueeze(1)
            y[idxs_linears_kernel[sorted_linears[i+1]]] = self.permute_cols(y[idxs_linears_kernel[sorted_linears[i+1]]], perm)
            y[idxs_linears_kernel[sorted_linears[i]]] = self.permute_rows(y[idxs_linears_kernel[sorted_linears[i]]], perm)
            y[idxs_linears_bias[sorted_linears[i]]] = self.permute_rows_biases(y[idxs_linears_bias[sorted_linears[i]]], perm)
            y[idxs_filter_kernel[sorted_filters[i+1]]] = self.permute_rows(y[idxs_filter_kernel[sorted_filters[i+1]]], perm)
            y[idxs_filter_bias[sorted_filters[i+1]]] = self.permute_rows_biases(y[idxs_filter_bias[sorted_filters[i+1]]], perm)

        return y

    def _set_transform(self, transform):
        self.transform = lambda x: self.pre_transform(x, transform)

    def numpy_transform(self, row_permute, column_permute):
        # TODO: Consider supporting different permutation per batch element
        perm = self.rng.permutation(row_permute[0].shape[-1])
        return [op[..., perm] for op in row_permute] + [
            ip[..., perm, :] for ip in column_permute
        ]

    def jax_transform(self, row_permute, column_permute):
        self.rng, perm_rng = jax.random.split(self.rng)
        perm = jax.random.permutation(perm_rng, row_permute[0].shape[-1])
        return [op[..., perm] for op in row_permute] + [
            ip[..., perm, :] for ip in column_permute
        ]
    
    def permute_rows(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1), torch.arange(x.shape[1]).unsqueeze(1), perm]
    
    def permute_cols(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1), perm.squeeze()]
    
    def permute_rows_biases(self, x, perm):
        return x[torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(1), perm].squeeze(1)

    def pytorch_transform(self, row_permute, column_permute, arange_for_sorting_batch,):
        perm = torch.stack([torch.randperm(row_permute[0].shape[-1]) for _ in range(row_permute[0].shape[0])] , dim=0).unsqueeze(1)

        return [
            row_permute[0][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), torch.arange(6).unsqueeze(1), perm],
            row_permute[1][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), perm].squeeze(1),
            row_permute[2][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), torch.arange(2).unsqueeze(1), perm],
            row_permute[3][arange_for_sorting_batch.unsqueeze(1).unsqueeze(1), perm].squeeze(1),
            column_permute[0][arange_for_sorting_batch.unsqueeze(1), perm.squeeze()]
        ]

class RandomQuantileWeightDropout(JointParameterAugmentation):
    """Randomly masks out weights of the network below a certain quantile. The quantile is
    uniformly sampled between min_quantile and max_quantile.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. Optional, not needed for this augmentation.
        min_quantile: Minimum quantile to use when sampling to quantile to use to mask out weights.
        max_quantile: Maximum quantile to use when sampling to quantile to use to mask out weights.
    """

    def __init__(
        self,
        *args,
        min_quantile: float = 0.0,
        max_quantile: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        assert 0 <= self.min_quantile <= 1, "min_quantile must be between 0 and 1"
        assert 0 <= self.max_quantile <= 1, "max_quantile must be between 0 and 1"
        assert (
            self.min_quantile <= self.max_quantile
        ), "min_quantile must be smaller than max_quantile"

    def numpy_transform(self, x: ParametersList):
        comb_tensor = np.concatenate([t.reshape(self.batch_size, -1) for t in x], axis=1)
        quantile = self.rng.uniform(self.min_quantile, self.max_quantile, size=(self.batch_size,))
        threshold = np.quantile(np.abs(comb_tensor), quantile, axis=1)
        return [t * (np.abs(t) >= threshold) for t in x]

    def jax_transform(self, x):
        comb_tensor = jnp.concatenate([t.reshape(self.batch_size, -1) for t in x], axis=1)
        self.rng, quant_rng = jax.random.split(self.rng)
        quantile = jax.random.uniform(
            key=quant_rng,
            shape=(self.batch_size,),
            minval=self.min_quantile,
            maxval=self.max_quantile,
        )[0]
        threshold = jnp.quantile(jnp.abs(comb_tensor), quantile, axis=1)
        return [t * (jnp.abs(t) >= threshold) for t in x]

    def pytorch_transform(self, x):
        comb_tensor = torch.cat([t.reshape(x[0].shape[0], -1) for t in x], dim=1)

        quantile = (
            torch.rand((x[0].shape[0],), generator=self.rng, device=self.device)
            * (self.max_quantile - self.min_quantile)
            + self.min_quantile
        )
        threshold = torch.quantile(torch.abs(comb_tensor), quantile)

        return [t * (torch.abs(t) >= threshold.reshape([t.shape[0]]+ [1]*(t.ndim-1))) for t in x]


class RandomTranslateMFN(JointParameterAugmentation):
    """"""

    def __init__(
        self,
        *args,
        min_translation: Number = 0.0,
        max_translation: Number = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_translation = min_translation
        self.max_translation = max_translation

        if isinstance(self.min_translation, float):
            self.translation_shape = (1,)
        else:
            self.translation_shape = tuple(self.min_translation.shape)

        self.biases_indices = np.array([
            i
            for i, x in enumerate(self.param_keys)
            if re.search(r"\S*filter\S*linear\S*bias\S*", x)
            if not None
        ])

        self.kernels_indices = np.array([
            i
            for i, x in enumerate(self.param_keys)
            if re.search(r"\S*filter\S*linear\S*kernel\S*", x)
            if not None
        ])

    def numpy_transform(self, params: ParametersList):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        stack_axis = 1

        biases = np.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], axis=stack_axis
        )
        kernels = np.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], axis=stack_axis
        )

        translation_vector = self.rng.uniform(self.min_translation, self.max_translation)

        bias_translation_vector = np.matmul(translation_vector, kernels)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[:, i]

        return params

    def jax_transform(self, params: ParametersList):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        stack_axis = 1

        biases = jnp.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], axis=stack_axis
        )
        kernels = jnp.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], axis=stack_axis
        )

        self.rng, translation_rng = jax.random.split(self.rng)
        translation_vector = jax.random.uniform(
            key=translation_rng,
            shape=self.translation_shape,
            minval=self.min_translation,
            maxval=self.max_translation,
        )

        bias_translation_vector = jnp.matmul(translation_vector, kernels)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[:, i]

        return params

    def pytorch_transform(self, params: ParametersList):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        if self.batch_size == 1:
            stack_dim = 0
        else:
            stack_dim = 1

        biases = torch.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], dim=stack_dim
        )
        kernels = torch.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], dim=stack_dim
        )

        if self.batch_size == 1:
            batched_translation_vector = (
                torch.rand(
                    *self.translation_shape,
                    generator=self.rng,
                    device=self.device,
                )
                * (self.max_translation - self.min_translation)
                + self.min_translation
            )
        else:
            batched_translation_vector = (
                torch.rand(
                    (biases.shape[0], 1, 1, *self.translation_shape),
                    generator=self.rng,
                    device=self.device,
                )
                * (self.max_translation - self.min_translation)
                + self.min_translation
            )

        bias_translation_vector = torch.matmul(batched_translation_vector, kernels).squeeze(2)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[:, i]
            
        return params


class IndividualParameterAugmentation(TensorAugmentation, ABC):
    """Base augmentation class for operations on individual parameters, i.e. each parameter being
    augmented independently of each other. Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from augmentation.
    """

    def __init__(
        self,
        platform: str = "numpy",
        seed: Optional[Any] = None,
        param_keys: Optional[List[str]] = None,
        exclude_params: Optional[List[str]] = None,
        device=None,
        batch_size: int = 1,
    ):
        super().__init__(platform=platform, seed=seed, device=device, batch_size=batch_size)
        if exclude_params is not None:
            assert (
                param_keys is not None
            ), "param_keys must be specified if exclude_params is specified"
        self.param_keys = param_keys
        self.exclude_params = exclude_params if exclude_params is not None else []
        assert all(
            [k in self.param_keys for k in self.exclude_params]
        ), "exclude_params must be a subset of param_keys"

    @abstractmethod
    def numpy_transform(self, x: ParameterAny, param_key: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def jax_transform(self, x: ParameterAny, param_key: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def pytorch_transform(self, x: ParameterAny, param_key: Optional[str] = None):
        raise NotImplementedError

    def __call__(self, x: Data):
        """Perform augmentation.

        Args:
            x: Input to augment. If single parameter, the whole parameter will be
                augmented. If a tuple, the first element is assumed to be the parameters,
                and second or later ones tensors not to be augmented. If parameters are
                a list, ensure that it is passed within a list, i.e. x=[params,...] with
                params = [param1, param2, ...]. The parameters should be in the same order
                as the param_keys.

        Returns:
            Augmented input.
        """
        if isinstance(x.params, (list, tuple)):
            assert (self.param_keys is None) or (
                len(x.params) == len(self.param_keys)
            ), f"Number of parameters ({len(x.params)}) must match number of param keys ({len(self.param_keys)})"

            self.common_transform()

            for i, p in enumerate(x.params):
                if self.param_keys is not None and self.param_keys[i] in self.exclude_params:
                    pass
                else:
                    x.params[i] = (
                        self.transform(
                            p,
                            param_key=self.param_keys[i] if self.param_keys is not None else None,
                        )
                    )
        else:
            x.params = self.transform(x.params)

        return x

    def _set_common_transform(self, transform):
        self.common_transform = transform

    def _setup_pytorch(self):
        super()._setup_pytorch()
        self._set_common_transform(self.pytorch_common_transform)

    def _setup_jax(self):
        super()._setup_jax()
        self._set_common_transform(self.jax_common_transform)

    def _setup_numpy(self):
        super()._setup_numpy()
        self._set_common_transform(self.numpy_common_transform)

    def _setup_auto(self):
        super()._setup_auto()
        self._set_common_transform(self.auto_common_transform)

    def numpy_common_transform(self):
        pass

    def jax_common_transform(self):
        pass

    def pytorch_common_transform(self):
        pass

    def auto_common_transform(self):
        return self.numpy_common_transform()

    def __repr__(self):
        return f"ParameterAugmentation(platform={self.platform})"


class LayerWiseNormalize(IndividualParameterAugmentation):
    """Normalizes the input by subtracting the mean and dividing by the standard deviation.

    Args:
        mean: Mean to subtract. Can be a float, numpy array, torch tensor, or jax array.
        std: Standard deviation to divide by. Can be a float, numpy array, torch tensor, or jax array.
    """

    def __init__(
        self,
        *args,
        mean: Number,
        std: Number,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std

    def all_transform(self, params: ParameterVector, param_key: str):
        return (params - self.mean[param_key]) / self.std[param_key]

    def numpy_transform(self, x, param_key: Optional[str] = None):
        return self.all_transform(x, param_key)

    def jax_transform(self, x, param_key: Optional[str] = None):
        return self.all_transform(x, param_key)

    def pytorch_transform(self, x, param_key: Optional[str] = None):
        return self.all_transform(x, param_key)
    
    def __repr__(self):
        means = ", ".join([f"{k}={v}" for k, v in self.mean.items()])
        stds = ", ".join([f"{k}={v}" for k, v in self.std.items()])
        return f"LayerWiseNormalize(means={means}, stds={stds})"


class RandomDropout(IndividualParameterAugmentation):
    """Randomly sets parameters to zero.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from augmentation.
        p: Probability of setting a parameter to zero. Can be a float or a dictionary
            mapping parameter keys to probabilities.
    """

    def __init__(self, *args, p: Union[float, Dict[str, float]] = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        if isinstance(self.p, dict):
            assert self.param_keys is not None, "param_keys must be specified if p is a dict"
            assert len(self.p) + len(self.exclude_params) == len(
                self.param_keys
            ), f"Number of parameters ({len(self.p)}) must match number of param keys ({len(self.param_keys)}) minus number of excluded parameters ({len(self.exclude_params)})"
            assert all(
                [k in self.param_keys for k in self.p]
            ), "param_keys must contain all keys in p"
            assert all(
                [0 <= prob <= 1 for prob in self.p.values()]
            ), "probabilities must be between 0 and 1"
        else:
            assert isinstance(self.p, float), "p must be a float"
            assert 0 <= self.p <= 1, "p must be between 0 and 1"

    def _get_p(self, param_key: Optional[str] = None):
        if isinstance(self.p, dict):
            assert param_key is not None, "param_key must be specified if p is a dict"
            assert param_key in self.p, f"param_key {param_key} not found in p"
            return self.p[param_key]
        else:
            return self.p

    def numpy_transform(self, x, param_key: Optional[str] = None):
        p = self._get_p(param_key)
        return x * (self.rng.random(x.shape) > p)

    def jax_transform(self, x, param_key: Optional[str] = None):
        p = self._get_p(param_key)
        self.rng, rng = jax.random.split(self.rng)
        return x * (jax.random.uniform(rng, x.shape) > p)

    def pytorch_transform(self, x, param_key: Optional[str] = None):
        p = self._get_p(param_key)
        return x * (torch.rand(x.shape, generator=self.rng, device=self.device) > p)

    def __repr__(self):
        return f"RandomDropout(platform={self.platform}, p={self.p}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"


class RandomGaussianNoise(IndividualParameterAugmentation):
    """Adds Gaussian noise to parameters.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from augmentation.
        sigma: Standard deviation of the Gaussian noise. Can be a float or a dictionary
            mapping parameter keys to standard deviations.
    """

    def __init__(self, *args, sigma: Union[Dict[str, Any], float] = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def _get_sigma(self, param_key: Optional[str] = None):
        if isinstance(self.sigma, dict):
            assert param_key is not None, "param_key must be specified if sigma is a dict"
            assert param_key in self.sigma, f"param_key {param_key} not found in sigma"
            return self.sigma[param_key]
        else:
            return self.sigma

    def numpy_transform(self, x, param_key: Optional[str] = None):
        sigma = self._get_sigma(param_key)
        return x + self.rng.normal(0, sigma, size=x.shape)

    def jax_transform(self, x, param_key: Optional[str] = None):
        sigma = self._get_sigma(param_key)
        self.rng, rng = jax.random.split(self.rng)
        return x + jax.random.normal(rng, x.shape) * sigma

    def pytorch_transform(self, x, param_key: Optional[str] = None):
        sigma = self._get_sigma(param_key)
        return x + x.std() * torch.normal(0, sigma, size=x.shape, generator=self.rng, device=self.device)

    def __repr__(self):
        return f"GaussianNoise(platform={self.platform}, sigma={self.sigma}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"


class RandomRotate(IndividualParameterAugmentation):
    def __init__(
        self,
        *args,
        min_angle: float = 0.0,
        max_angle: float = 3.14,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_angle = min_angle
        self.max_angle = max_angle

        assert (
            self.min_angle <= self.max_angle
        ), f"min_angle must be smaller than max_angle, but got min_angle={self.min_angle} and max_angle={self.max_angle}"

    def numpy_common_transform(self):
        angle = self.rng.uniform(self.min_angle, self.max_angle, size=(self.batch_size,))
        self.rotation_matrix = np.empty((self.batch_size, 2, 2), dtype=np.float32)
        self.rotation_matrix[:, 0, 0] = np.cos(angle)
        self.rotation_matrix[:, 1, 0] = np.sin(angle)
        self.rotation_matrix[:, 0, 1] = -self.rotation_matrix[:, 1, 0]
        self.rotation_matrix[:, 1, 1] = self.rotation_matrix[:, 0, 0]

    def jax_common_transform(self):
        self.rng, rng = jax.random.split(self.rng)
        angle = (
            jnp.random.uniform(rng, (self.batch_size,)) * (self.max_angle - self.min_angle)
            + self.min_angle
        )
        self.rotation_matrix = jnp.empty((self.batch_size, 2, 2), dtype=jnp.float32)
        self.rotation_matrix = jax.ops.index_update(
            self.rotation_matrix, jax.ops.index[:, 0, 0], jnp.cos(angle)
        )
        self.rotation_matrix = jax.ops.index_update(
            self.rotation_matrix, jax.ops.index[:, 1, 0], jnp.sin(angle)
        )
        self.rotation_matrix = jax.ops.index_update(
            self.rotation_matrix, jax.ops.index[:, 0, 1], -self.rotation_matrix[:, 1, 0]
        )
        self.rotation_matrix = jax.ops.index_update(
            self.rotation_matrix, jax.ops.index[:, 1, 1], self.rotation_matrix[:, 0, 0]
        )

    def pytorch_common_transform(self):
        if self.batch_size == 1:
            angle = (
                torch.rand(1, generator=self.rng, device=self.device)
                * (self.max_angle - self.min_angle)
                + self.min_angle
            )
            self.rotation_matrix = torch.empty((2, 2), device=self.device)
            self.rotation_matrix[0, 0] = torch.cos(angle)
            self.rotation_matrix[1, 0] = torch.sin(angle)
            self.rotation_matrix[0, 1] = -self.rotation_matrix[1, 0]
            self.rotation_matrix[1, 1] = self.rotation_matrix[0, 0]
        else:
            angle = (
                torch.rand((self.batch_size,), generator=self.rng, device=self.device)
                * (self.max_angle - self.min_angle)
                + self.min_angle
            )
            self.rotation_matrix = torch.empty((self.batch_size, 2, 2), device=self.device)
            self.rotation_matrix[:, 0, 0] = torch.cos(angle)
            self.rotation_matrix[:, 1, 0] = torch.sin(angle)
            self.rotation_matrix[:, 0, 1] = -self.rotation_matrix[:, 1, 0]
            self.rotation_matrix[:, 1, 1] = self.rotation_matrix[:, 0, 0]

    def numpy_transform(self, x, param_key: Optional[str] = None):
        return np.matmul(self.rotation_matrix, x)

    def jax_transform(self, x, param_key: Optional[str] = None):
        return jnp.matmul(self.rotation_matrix, x)

    def pytorch_transform(self, x, param_key: Optional[str] = None):
        if self.batch_size >1:
            rotation_matrix = self.rotation_matrix[:x.shape[0]]
        else:
            rotation_matrix = self.rotation_matrix

#        return torch.einsum('bca,bawf->bcwf',rotation_matrix, x)
        return torch.matmul(rotation_matrix, x)

    def __repr__(self):
        return f"RandomRotate(platform={self.platform}, min_angle={self.min_angle}, max_angle={self.max_angle}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"


class RandomScale(IndividualParameterAugmentation):
    def __init__(self, *args, min_scale: float = 0.5, max_scale: float = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale

        if isinstance(self.min_scale, float):
            self.scale_shape = (1,1)
        else:
            self.scale_shape = tuple(self.min_scale.shape)

        assert (
            self.min_scale <= self.max_scale
        ), f"min_scale must be smaller than max_scale, but got min_scale={self.min_scale} and max_scale={self.max_scale}"

    def numpy_common_transform(self):
        self.scale = self.rng.uniform(
            self.min_scale, self.max_scale, size=(self.batch_size, *self.scale_shape)
        )

    def jax_common_transform(self):
        self.rng, rng = jax.random.split(self.rng)
        self.scale = (
            jnp.random.uniform(rng, (self.batch_size, *self.scale_shape))
            * (self.max_scale - self.min_scale)
            + self.min_scale
        )

    def pytorch_common_transform(self):
        if self.batch_size == 1:
            self.scale = (
                torch.rand(
                    *self.scale_shape, generator=self.rng, device=self.device
                )
                * (self.max_scale - self.min_scale)
                + self.min_scale
            )
        else:
            self.scale = (
                torch.rand(
                    (self.batch_size, *self.scale_shape), generator=self.rng, device=self.device
                )
                * (self.max_scale - self.min_scale)
                + self.min_scale
            )

    def numpy_transform(self, x, param_key: Optional[str] = None):
        return x * self.scale

    def jax_transform(self, x, param_key: Optional[str] = None):
        return x * self.scale

    def pytorch_transform(self, x, param_key: Optional[str] = None):
        if self.batch_size > 1:
            scale = self.scale[:x.shape[0]]
        else:
            scale = self.scale
        return x * scale

    def __repr__(self):
        return f"RandomScale(platform={self.platform}, min_scale={self.min_scale}, max_scale={self.max_scale}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"
