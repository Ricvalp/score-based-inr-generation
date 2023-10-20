import functools
from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from utils import marginal_prob_std


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    embed_dim: int
    scale: float = 30.0

    @nn.compact
    def __call__(self, x):
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        W = self.param("W", jax.nn.initializers.normal(stddev=self.scale), (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class ResMLP(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.array) -> jnp.ndarray:
        y = nn.GroupNorm()(x)
        y = nn.silu(y)
        y = nn.Dense(self.hidden_dim)(y)

        t = nn.silu(t)
        t = nn.Dense(self.hidden_dim)(t)
        y = y + t

        y = nn.GroupNorm()(y)
        y = nn.silu(y)
        y = nn.Dropout(0.3, deterministic=False)(y)
        y = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )(y)
        y = y + x
        return y


class ScoreNet1dDDPM(nn.Module):
    marginal_prob_std: Any
    width: int = 512
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        embed = GaussianFourierProjection(embed_dim=self.embed_dim)(t)

        y = nn.Dense(self.width)(x)

        h1 = ResMLP(self.width)(y, embed)
        h2 = ResMLP(self.width)(h1, embed)
        h3 = ResMLP(self.width)(h2, embed)
        h4 = ResMLP(self.width)(h3, embed)

        h = ResMLP(self.width)(h4 + h3, embed)
        h = ResMLP(self.width)(h + h2, embed)
        h = ResMLP(self.width)(h + h1, embed)
        h = ResMLP(self.width)(h + y, embed)

        h = nn.GroupNorm()(h)
        h = nn.silu(h)
        h = nn.Dense(
            x.shape[-1], kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )(h)

        h = h / self.marginal_prob_std(t)[:, None]
        return h


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    rng, dropout_rng, dropout_rng1 = jax.random.split(rng, 3)
    x = jax.random.normal(rng, (512, 100))
    t = jax.random.normal(rng, (512,))

    sigma = 25
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    model = ScoreNet1dDDPM(marginal_prob_std_fn, width=512)

    init_rngs = {"params": rng, "dropout": dropout_rng}
    params = model.init(
        init_rngs,
        x,
        t,
    )
    y = model.apply(params, x, t, rngs={"dropout": dropout_rng})
    y_1 = model.apply(params, x, t, rngs={"dropout": dropout_rng1})

    assert True
