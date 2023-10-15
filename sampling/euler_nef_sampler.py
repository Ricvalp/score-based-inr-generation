import jax
import jax.numpy as jnp
from tqdm import tqdm


def score_fn(score_model, params, x, t):
  return score_model.apply(params, x, t)

def Euler_Maruyama_nef_sampler(rng,
                           score_model,
                           params,
                           marginal_prob_std,
                           diffusion_coeff,
                           nef_shape,
                           batch_size=64,
                           num_steps=500,
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    rng: A JAX random state.
    score_model: A `flax.linen.Module` object that represents the architecture
      of a score-based model.
    params: A dictionary that contains the model parameters.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """  
  rng, step_rng = jax.random.split(rng)  
  time_shape = (batch_size,)
  sample_shape = time_shape + nef_shape
  init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
  time_steps = jnp.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  for time_step in tqdm(time_steps):      
    batch_time_step = jnp.ones(time_shape) * time_step
    g = diffusion_coeff(time_step)
    mean_x = x + (g**2) * score_fn(score_model,
                                        params,
                                        x, 
                                        batch_time_step) * step_size
    rng, step_rng = jax.random.split(rng)
    x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)      
  # Do not include any noise in the last sampling step.
  return mean_x