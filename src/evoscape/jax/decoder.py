from typing import Any, Callable, Dict

import optax
import jax.numpy as jnp
from flax import nnx

from evoscape.jax.experiment import Experiment
from evoscape.jax.dynamics import _integrate, state_probs
from evoscape.jax.optim import make_step, clip_dynamic, run_optimization_decoder


class Decoder(Experiment):

    def __init__(self, landscape, dims_decoder):
        # Initializing the landscape
        super().__init__(landscape)

        # Initializing the decoder part
        self.rngs = nnx.Rngs(42)

        self.dims_decoder = dims_decoder
        layers = []
        for in_dim, out_dim in zip(dims_decoder[:-1], dims_decoder[1:]):
            layers.append(nnx.Linear(in_dim, out_dim, rngs=self.rngs))
            layers.append(nnx.relu)

        layers.pop() # last layer must be only linear, not linear + relu

        self.mlp = nnx.Sequential(*layers)

    def _decode_traj_landscape(self, traj, decoder):
        # Our mlp wants (batch, 2), and we have (2, n, nt) so we need to convert the trajectory, decode it, then reconvert it
        traj_permuted = jnp.transpose(traj, (1, 2, 0))
        n, nt, _ = traj_permuted.shape

        traj_permuted_batched = jnp.reshape(traj_permuted, (-1, 2))
        traj_permuted_decoded_batched = decoder(traj_permuted_batched)
        traj_permuted_decoded = jnp.reshape(traj_permuted_decoded_batched, (n, nt, self.dims_decoder[-1]))

        traj_decoded = jnp.transpose(traj_permuted_decoded, (2, 0, 1))

        return traj_decoded

    def forward(self, q_cells):
        key = self.rngs.default()

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_init = q_cells
        dynamic = self.dynamic

        _, traj, states = _integrate(
            key,
            q_init,
            t0,
            tf,
            nt,
            ndt,
            noise,
            dynamic,
            self.static,
            self.regime,
            self.get_cell_states,
        )

        traj_decoded = self._decode_traj_landscape(traj, self.mlp)
        return traj_decoded

    def optimize(
        self,
        data, 
        user_fitness: Callable, 
        fitness_params = None, 
        steps: int = 50, 
        opt_dynamic=optax.adam(0.1), 
        opt_decoder=optax.adam(0.1)
    ):
        """
        data : array of size (nb_genes, cell number, instant t_i)
            For example (4, 200, 50) means there are four genes, 200 cells and 50 timepoints
        """

        key = self.rngs.default()

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_init = self.q_init
        if q_init is None:
            q_init = jnp.asarray(self.dynamic.init_cond)[:, None]

        def loss_fn(dynamic, decoder, subkey):
            _, traj, states = _integrate(
                subkey,
                q_init,
                t0,
                tf,
                nt,
                ndt,
                noise,
                dynamic,
                self.static,
                self.regime,
                self.get_cell_states,
            )

            decoded_traj = self._decode_traj_landscape(traj, decoder)

            # this is a lot of parameters...
            return user_fitness(traj, states, decoded_traj, dynamic, decoder, data, fitness_params)


        dynamic_vals, fitness_vals = run_optimization_decoder(self.dynamic, self.mlp, key, steps, loss_fn, opt_dynamic, opt_decoder)
        self.dynamic = dynamic_vals[-1]


        _, traj, states = _integrate(
            key,
            q_init,
            t0,
            tf,
            nt,
            ndt,
            noise,
            self.dynamic,
            self.static,
            self.regime,
            self.get_cell_states,
        )

        self.trajectories = traj
        self.states = states
        self.fitness_vals = fitness_vals
        self.dynamic_vals = dynamic_vals

        return fitness_vals
