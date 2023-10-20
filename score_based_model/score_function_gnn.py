import functools
from typing import Any, NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from ml_collections import ConfigDict
from utils import marginal_prob_std

from score_based_model.graph_utils import (
    flatten_params,
    get_nef_graph_lists,
    jraph_tuple_to_nefs,
    nefs_to_jraph_tuple,
)


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


class MLP(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(1e-6),
            )(x)
            # Batchnorm leads to trace leaks.
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)

        # Map to output dimension.
        x = nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(1e-6),
        )(x)
        return x


class ScoreNetGNN(nn.Module):
    param_config: ConfigDict
    nef_graph: NamedTuple
    num_nodes_per_graph: int
    num_edges_per_graph: int
    num_layers: int
    hidden_dim: int
    num_layers_update_fn: int
    hidden_dim_update_fn: int
    marginal_prob_std_fn: Any

    @nn.compact
    def __call__(self, x: jnp.array, t: jnp.array, train: bool = True) -> jnp.ndarray:
        @jax.vmap
        @jraph.concatenated_args
        def update_fn(features):
            update_mlp = MLP(
                hidden_dim=self.hidden_dim_update_fn,
                num_layers=self.num_layers_update_fn,
                out_dim=self.hidden_dim,
            )
            return update_mlp(features)

        # Obtain the Gaussian random feature embedding for t
        time_embed = nn.Dense(self.hidden_dim)(
            GaussianFourierProjection(embed_dim=self.hidden_dim)(t)
        )

        # Convert to jraph tuple
        x = nefs_to_jraph_tuple(
            nef_params=x,
            nef_config=self.param_config,
            nef_graph=self.nef_graph,
        )

        for _ in range(self.num_layers):
            # Perform message passing
            x = jraph.InteractionNetwork(
                update_edge_fn=update_fn,
                update_node_fn=update_fn,
            )(x)

            # Embed time a little bit more
            time_embed = nn.Dense(self.hidden_dim)(
                GaussianFourierProjection(embed_dim=self.hidden_dim)(t)
            )

            # Repeat for each graph
            time_embed_nodes = jnp.repeat(time_embed, self.num_nodes_per_graph, axis=0)
            time_embed_edges = jnp.repeat(time_embed, self.num_edges_per_graph, axis=0)

            # Add time embedding to graph
            x._replace(nodes=x.nodes + time_embed_nodes, edges=x.edges + time_embed_edges)

        # Create weight array from graph
        x = jraph_tuple_to_nefs(x, nef_config=self.param_config)
        x = x / self.marginal_prob_std(t)[:, None]
        return x


if __name__ == "__main__":
    # testing ScoreNet1D

    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (512, 100))
    t = jax.random.normal(rng, (512,))

    sigma = 25
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    model = ScoreNetGNN(marginal_prob_std_fn)
    params = model.init(rng, x, t)
    y = model.apply(params, x, t)

    assert True
