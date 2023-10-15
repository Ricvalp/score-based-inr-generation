import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple
import jax


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x):    
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""  
  output_dim: int  
  
  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.output_dim)(x)


class ScoreNet1D(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture.
  
  Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
  """
  marginal_prob_std: Any
  channels: Tuple[int] = (256, 128, 64, 32)
  embed_dim: int = 256
  
  @nn.compact
  def __call__(self, x, t): 
    # The swish activation function
    act = nn.swish
    # Obtain the Gaussian random feature embedding for t   
    embed = act(nn.Dense(self.embed_dim)(
        GaussianFourierProjection(embed_dim=self.embed_dim)(t)))
        
    # Encoding path
    h1 = nn.Dense(self.channels[0])(x)    
    h1 += Dense(self.channels[0])(embed)
    h1 = nn.GroupNorm(4)(h1)    
    h1 = act(h1)

    h2 = nn.Dense(self.channels[1])(h1)
    h2 += Dense(self.channels[1])(embed)
    h2 = nn.GroupNorm()(h2)        
    h2 = act(h2)

    h3 = nn.Dense(self.channels[2])(h2)
    h3 += Dense(self.channels[2])(embed)
    h3 = nn.GroupNorm()(h3)
    h3 = act(h3)

    h4 = nn.Dense(self.channels[3])(h3)
    h4 += Dense(self.channels[3])(embed)
    h4 = nn.GroupNorm()(h4)    
    h4 = act(h4)

    # Decoding path
    h = nn.Dense(self.channels[2])(h4)    
    h += Dense(self.channels[2])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)

    h = nn.Dense(self.channels[1])(
                      jnp.concatenate([h, h3], axis=-1)
                  )
    h += Dense(self.channels[1])(embed)
    h = nn.GroupNorm()(h)
    h = act(h)

    h = nn.Dense(self.channels[0])(
                      jnp.concatenate([h, h2], axis=-1)
                  )    
    h += Dense(self.channels[0])(embed)    
    h = nn.GroupNorm()(h)  
    h = act(h)
    h = nn.Dense(x.shape[1])(
        jnp.concatenate([h, h1], axis=-1)
    )

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None]
    return h


if __name__=="__main__":

  # testing ScoreNet1D

  import jax.numpy as jnp
  import flax.linen as nn
  from typing import Any, Tuple
  import jax
  import functools
  from utils import marginal_prob_std

  rng = jax.random.PRNGKey(42)
  x = jax.random.normal(rng, (512, 100))
  t = jax.random.normal(rng, (512, ))

  sigma =  25
  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

  model = ScoreNet1D(marginal_prob_std_fn)
  params = model.init(rng, x, t)
  y = model.apply(params, x, t)

  assert True

