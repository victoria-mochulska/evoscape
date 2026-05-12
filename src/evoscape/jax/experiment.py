from typing import Any, Callable, Dict
import numpy as np

from jax.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jrd
from jax import value_and_grad
import optax

from .converters import landscape_to_pytree, pytree_to_landscape
from .dynamics import _integrate, state_probs
from .optim import make_step, clip_dynamic, run_optimization, run_optimization_optax
from .regimes import wrapped_regime


class Experiment:
    def __init__(self, landscape: Any):
        self.dynamic, self.static = landscape_to_pytree(landscape)

        # Things the user can configure via setters
        self.sim_params: Dict[str, Any] = {}
        self.q_init = None
        self.get_cell_states = None
        self.signal_param = None
        self.regime = None
        self.trajectories = None
        self.states = None
        self.fitness_vals = None
        self.dynamic_vals = None
    # ------------------------------------------------------------------
    # SETTERS (one responsibility each)
    # ------------------------------------------------------------------

    def set_initial_conditions(self, q_init):
        """Initial conditions for the simulation."""
        self.q_init = jnp.asarray(q_init)

    def set_regime_params(self, signal_param=None):
        """Define the regime used during integration."""
        self.signal_param = signal_param
        self.regime = wrapped_regime(self.static, self.signal_param)

    def set_simulation(self, t0: float, tf: float, nt: int, ndt: int, noise: float):
        """Simulation parameters passed to integrate."""
        self.sim_params = dict(
            t0=t0,
            tf=tf,
            nt=nt,
            ndt=ndt,
            noise=noise,
        )

    def set_state_probs(self, get_cell_states):
        self.get_cell_states = get_cell_states

    # ------------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------------

    def get_landscape(self):
        trajectories = None
        cell_coordinates = None
        cell_states = None
        result = None
        fitness = None

        if self.trajectories is not None:
            trajectories = np.asarray(self.trajectories)
            cell_coordinates = trajectories[:, :, -1]

        if self.states is not None:
            cell_states = np.asarray(self.states)

        if self.fitness_vals is not None:
            result = np.asarray(self.fitness_vals)
            if result.size > 0:
                fitness = float(result[-1])

        return pytree_to_landscape(
            self.dynamic,
            self.static,
            trajectories=trajectories,
            cell_coordinates=cell_coordinates,
            cell_states=cell_states,
            fitness=fitness,
            result=result,
        )

    def get_trajectory(self, seed=42):
        key = jrd.PRNGKey(seed)

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_init = self.q_init
        if q_init is None:
            q_init = jnp.asarray(self.dynamic.init_cond)[:, None]

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

        return traj, states, t0, tf, nt

    # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------

    def optimize(self, user_fitness: Callable, fitness_params = None, steps: int = 50, optimizer=optax.adam(0.1), seed: int = 42):
        key = jrd.PRNGKey(seed)

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_init = self.q_init
        if q_init is None:
            q_init = jnp.asarray(self.dynamic.init_cond)[:, None]

        def loss_fn(dynamic, subkey):
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
            return user_fitness(traj, states, dynamic, fitness_params)


        dynamic_vals, fitness_vals = run_optimization_optax(self.dynamic, key, steps, loss_fn, optimizer)
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
        ## self.dynamic_vals = [ tree_map(lambda x: x[t], dynamic_vals) for t in range(len(dynamic_vals[0])) ]
        self.dynamic_vals = dynamic_vals

        return fitness_vals
