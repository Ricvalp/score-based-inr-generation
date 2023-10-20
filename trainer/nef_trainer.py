import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

import wandb
from score_based_model import ScoreNet1D, ScoreNet1dDDPM, ScoreNetGNN
from score_based_model.graph_utils import (
    flatten_params,
    get_nef_graph_lists,
    nefs_to_jraph_tuple,
)


class TrainerModuleNef:
    def __init__(
        self,
        train_loader,
        checkpoint_dir,
        marginal_prob_std_fn,
        param_config,
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

        # Batch from evaluation for shape initialization
        self.fake_input = next(iter(train_loader)).params
        self.fake_time = jnp.ones(self.fake_input.shape[0])

        # self.model = ScoreNet1D(marginal_prob_std_fn)
        # self.model = ScoreNet1dDDPM(marginal_prob_std_fn, width=512)
        nef_graph = get_nef_graph_lists(
            nef_params=self.fake_input,
            nef_config=param_config,
            nef_name="SIREN",  # Replace at some point
        )
        self.model = ScoreNetGNN(
            num_edges_per_graph=nef_graph.n_edge,
            num_nodes_per_graph=nef_graph.n_node,
            param_config=param_config,
            nef_graph=nef_graph,
            num_layers=6,
            hidden_dim=128,
            num_layers_update_fn=2,
            hidden_dim_update_fn=64,
            marginal_prob_std_fn=marginal_prob_std_fn,
        )

        # Create jitted training and eval functions
        self.create_functions()

        # Initialize model
        self.init_model()

    def create_functions(self):
        # Training function
        def train_step(rng, model_state, batch):
            grad = jax.value_and_grad(loss_fn, argnums=2, has_aux=False)

            loss, grads = grad(
                rng, model_state, model_state.params, batch, self.marginal_prob_std_fn
            )
            model_state = model_state.apply_gradients(grads=grads)

            return model_state, loss

        self.train_step = jax.jit(train_step)

    def init_model(self):
        # Initialize model

        rng = jax.random.PRNGKey(self.seed)
        rng, dropout_rng = jax.random.split(rng, 2)

        # model_params = self.model.init({'params': rng, 'dropout':dropout_rng}, self.fake_input, self.fake_time)
        model_params = self.model.init({"params": rng}, self.fake_input, self.fake_time)

        # Optimizer

        cosine_warmup_decay_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=self.lr, warmup_steps=1000, decay_steps=50000
        )

        optimizer = optax.adam(learning_rate=cosine_warmup_decay_scheduler)

        # SIMPLE ADAM
        # optimizer = optax.adam(learning_rate=self.lr)

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
            loss, rng = self.train_epoch(rng)
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

        return loss, rng

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
    rng, step_rng, dropout_rng = jax.random.split(rng, 3)
    random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=1.0)
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, x.shape)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None]
    score = model_state.apply_fn(params, perturbed_x, random_t, rngs={"dropout": dropout_rng})
    loss = jnp.mean(jnp.sum((score * std[:, None] + z) ** 2, axis=(1,)))
    return loss
