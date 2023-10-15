import jax
import jax.numpy as jnp
import pickle
from pathlib import Path
from flax.training import train_state

import flax.linen as nn
import optax
import wandb
from tqdm import tqdm

from score_based_model import ScoreNet1D


class TrainerModuleNef:
    
    def __init__(
        self,
        train_loader,
        checkpoint_dir,
        marginal_prob_std_fn,
        lr=1e-4,
        wandb_log="no",
        seed=42,
    ):
        super().__init__()

        self.lr = lr
        self.seed = seed
        self.train_loader = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.wandb_log = wandb_log

        # Create model
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.model = ScoreNet1D(marginal_prob_std_fn)

        # Batch from evaluation for shape initialization
        self.fake_input = next(iter(train_loader)).params
        self.fake_time = jnp.ones(self.fake_input.shape[0])

        # Create jitted training and eval functions
        self.create_functions()

        # Initialize model
        self.init_model()


    def create_functions(self):
        # Training function
        def train_step(rng, model_state, batch):
            grad = jax.value_and_grad(loss_fn, argnums=2, has_aux=False)

            loss, grads = grad(rng, model_state, model_state.params, batch, self.marginal_prob_std_fn)
            model_state = model_state.apply_gradients(grads=grads)

            return model_state, loss
        
        self.train_step = jax.jit(train_step)

    def init_model(self):
        # Initialize model

        rng = jax.random.PRNGKey(self.seed)

        model_params = self.model.init({'params': rng}, self.fake_input, self.fake_time)


        # Optimizer

        # EXPONENTIAL DECAY LEARNING RATE
        # init_learning_rate = self.lr # initial learning rate for Adam
        # exponential_decay_scheduler = optax.exponential_decay(init_value=init_learning_rate, transition_steps=self.transition_steps,
        #                                                     decay_rate=self.decay_rate, transition_begin=50,  end_value=self.end_lr,
        #                                                     staircase=False)
        # optimizer = optax.adam(learning_rate=exponential_decay_scheduler)

        # SIMPLE ADAM
        optimizer = optax.adam(learning_rate=self.lr)

        # GRADIENT CLIPPING + ADAM
        # optimizer = optax.chain(
        #     optax.clip(self.clip_at),
        #     optax.adam(learning_rate=self.lr))

        # Initialize training state
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params, tx=optimizer
        )

    def train_model(self, num_epochs=50):
        
        rng = jax.random.PRNGKey(self.seed)

        t = tqdm(range(1, num_epochs + 1), unit="step")
        for epoch_idx in t:
            loss = self.train_epoch(rng)
            t.set_description(f"loss: {loss:.6f}")
            self.save_checkpoint(step=epoch_idx)

    def train_epoch(self, rng):
        plot_count = 0

        for data in self.train_loader:
            batch = data.params
            rng, step_rng = jax.random.split(rng, 2)
            self.model_state, loss = self.train_step(step_rng, self.model_state, batch)

            if self.wandb_log:
                wandb.log(
                    {
                        "train_loss": loss,
                    }
                )

            plot_count += 1

        return loss

    def save_checkpoint(self, step):
        checkpoint = {
            "model_params": self.model_state.params,
            "step": step,
        }
        with open(Path(self.checkpoint_dir) / Path("checkpoint"), "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self):
        with open(self.checkpoint_dir / Path("checkpoint"), "rb") as f:
            checkpoint = pickle.load(f)

        self.model_state = self.model_state.replace(params=checkpoint["model_params"])


def loss_fn(rng, model_state, params, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A `flax.linen.Module` object that represents the structure of 
      the score-based model.
    params: A dictionary that contains all trainable parameters.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  rng, step_rng = jax.random.split(rng)
  random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=1.)
  rng, step_rng = jax.random.split(rng)
  z = jax.random.normal(step_rng, x.shape)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None]
  score = model_state.apply_fn(params, perturbed_x, random_t)
  loss = jnp.mean(jnp.sum((score * std[:, None] + z)**2, 
                          axis=(1,)))
  return loss


