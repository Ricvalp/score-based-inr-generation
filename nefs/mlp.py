# Flax
import jax
from flax import linen as nn


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_dim,
                use_bias=True,
                kernel_init=nn.initializers.glorot_normal(),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_normal(),
        )(x)
        return x
