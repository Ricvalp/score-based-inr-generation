import re
from typing import Any, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import core, dtypes, random
from jax.nn.initializers import Initializer
from jax.random import KeyArray

from nefs.initializers import custom_uniform


class SIREN(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
    omega_0: float

    def setup(self):
        self.kernel_net = [
            SirenLayer(
                output_dim=self.hidden_dim,
                omega_0=self.omega_0,
                is_first_layer=True,
            )
        ] + [
            SirenLayer(
                output_dim=self.hidden_dim,
                omega_0=self.omega_0,
            )
            for _ in range(self.num_layers - 2)
        ]

        self.output_linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=1, mode="fan_in", distribution="normal"),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        for layer in self.kernel_net:
            x = layer(x)

        out = self.output_linear(x)

        return out


class SirenLayer(nn.Module):
    output_dim: int
    omega_0: float
    is_first_layer: bool = False

    def setup(self):
        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"
        self.linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        after_linear = self.omega_0 * self.linear(x)
        return jnp.sin(after_linear)
