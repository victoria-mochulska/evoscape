from typing import Any, Callable, Dict

import jax.numpy as jnp
import jax.random as jrnd
import numpy as np

from .converters import landscape_to_pytree, pytree_to_landscape
from .dynamics import _integrate, state_probs
from .optim import make_step, clip_dynamic, run_optimization
from .regimes import wrapped_regime


class Experiment:
    def __init__(self, landscape: Any):
        self.dynamic, self.static = landscape_to_pytree(landscape)

        # Things the user can configure via setters
        self.sim_params: Dict[str, Any] = {}
        self.q_flat = None
        self.get_cell_states = state_probs
        self.signal_param = None
        self.regime = None

        if self.static.regime_id != 3:
            self.regime = wrapped_regime(self.static, self.signal_param)

        self.trajectories = None
        self.states = None
        self.fitness_vals = None

    # ------------------------------------------------------------------
    # SETTERS (one responsibility each)
    # ------------------------------------------------------------------

    def set_initial_conditions(self, q_flat):
        """Initial conditions for the simulation."""
        self.q_flat = jnp.asarray(q_flat)

    def set_regime(self, regime_type="const", t_list=None, t0=None, tau=None, signal_param=None):
        """Define the regime used during integration."""
        self.signal_param = signal_param

        if regime_type == "const":
            self.static = self.static._replace(regime_id=0, morphogen_times=jnp.array([]))

        elif regime_type == "piecewise":
            assert t_list is not None, "Error : t_list is None"
            self.static = self.static._replace(regime_id=2, morphogen_times=jnp.array(t_list))

        elif regime_type == "sigmoid":
            assert t0 is not None, "Error : t0 is None"
            assert tau is not None, "Error : tau is None"

            module_static = self.static.module._replace(
                tau=jnp.full_like(self.static.module.tau, tau),
                use_tau=jnp.ones_like(self.static.module.use_tau, dtype=bool),
            )
            self.static = self.static._replace(
                regime_id=1,
                morphogen_times=jnp.array([t0]),
                module=module_static,
            )

        elif regime_type == "linear_2signals":
            assert signal_param is not None, "Error : signal_param is None"
            self.static = self.static._replace(regime_id=3)

        else:
            raise ValueError(f"Unknown regime_type: {regime_type}")

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
        key = jrnd.PRNGKey(seed)

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_flat = self.q_flat
        if q_flat is None:
            q_flat = jnp.asarray(self.dynamic.init_cond)[:, None]

        _, traj, states = _integrate(
            key,
            q_flat,
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

    def optimize(self, user_fitness: Callable, steps: int, lr: float = 0.05, seed: int = 42):
        key = jrnd.PRNGKey(seed)

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_flat = self.q_flat
        if q_flat is None:
            q_flat = jnp.asarray(self.dynamic.init_cond)[:, None]

        def loss_fn(dynamic, subkey):
            _, traj, states = _integrate(
                subkey,
                q_flat,
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
            return user_fitness(traj, states)

        step_fn = make_step(loss_fn, lr, clip_dynamic)
        (self.dynamic, key), fitness_vals = run_optimization(
            self.dynamic,
            key,
            step_fn,
            steps,
        )

        _, traj, states = _integrate(
            key,
            q_flat,
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

        return fitness_vals
