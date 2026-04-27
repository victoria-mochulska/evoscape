from typing import Any, Callable, Dict, Tuple
from functools import partial

from jax import jit, grad, tree_util
import jax.numpy as jnp
import jax.random as jrnd
import jax

from evoscape.jax_functions import integrate, get_states_from_probs
from evoscape.mr_jax import mr_sigmoid_jax, mr_const_jax, mr_piecewise_jax

from evoscape.landscapes.landscape_class import Landscape
from evoscape.morphogen_regimes import mr_const, mr_sigmoid, mr_piecewise

class Experiment:
    def __init__(self, landscape: Any):
        self.module_params, self.module_infos = landscape.to_pytree()

        # Things the user can configure via setters 
        self.sim_params: Dict[str, Any] = {}
        self.q_flat = None

        self.regime_type = None
        self.regime = None
        self.t_list = None
        self.t0 = None
        self.tau = None

    # ------------------------------------------------------------------
    # SETTERS (one responsibility each) 
    # ------------------------------------------------------------------

    def set_initial_conditions(self, q_flat):
        """Initial conditions for the simulation."""
        self.q_flat = jnp.asarray(q_flat)

    def set_regime(self, regime_type="const", t_list=None, t0=None, tau=None):
        """Define the regime used during integration."""
        self.regime_type = regime_type
        if regime_type == "const":
            self.regime = mr_const_jax 

        elif regime_type == "piecewise":
            assert t_list is not None, "Error : t_list is None"
            t_list = jnp.array(t_list)

            self.t_list = t_list
            self.regime = partial(mr_piecewise_jax, t_list=t_list)

        elif regime_type == "sigmoid":
            assert t0 is not None, "Error : t0 is None"
            assert tau is not None, "Error : tau is None"

            self.t0 = t0
            self.tau = tau
            self.regime = partial(mr_sigmoid_jax, t0=t0, tau=tau)

    def set_simulation(self, t0: float, tf: float, nt: int, ndt: int, noise: float):
        """Simulation parameters passed to integrate."""
        self.sim_params = dict(
            t0=t0,
            tf=tf,
            nt=nt,
            ndt=ndt,
            noise=noise,
        )

    # ------------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------------

    def get_landscape(self):

        corresponding_mr = {
            "piecewise" : mr_piecewise, 
            "const" : mr_const,
            "sigmoid" : mr_sigmoid
        }

        mr_regime = corresponding_mr[self.regime_type]
        n_regimes = self.module_infos["curl"].shape[0]
        morphogen_times = tuple(self.t_list)
        
        return Landscape.from_pytree(
            self.module_params,
            self.module_infos, 
            mr_regime = mr_regime, 
            n_regimes = n_regimes, 
            morphogen_times = morphogen_times, 
            tau = self.tau
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
        module_infos = self.module_infos 
        module_params = self.module_params

        _, traj, states = integrate(key,  q_flat, t0, tf, nt, ndt, noise, module_infos, module_params, mr_regime=self.regime)

        states = jnp.argmax(states, axis=0) ## states are actually state probabilities
        
        return traj, states, t0, tf, nt

    # ------------------------------------------------------------------
    # OPTIMIZATION 
    # ------------------------------------------------------------------

    def optimize(self, user_fitness : Callable, steps: int, lr: float = 0.05, seed: int = 42):

        ## declaring all the relevant variables for the fitness functions
        key = jrnd.PRNGKey(seed)

        p = self.sim_params

        t0 = p["t0"]
        tf = p["tf"]
        nt = p["nt"]
        ndt = p["ndt"]
        noise = p["noise"]

        q_flat = self.q_flat
        module_infos = self.module_infos 
        module_params = self.module_params

        def fitness(module_params, module_infos, key, q_flat, t0, tf, nt, ndt, noise):
            key_final, traj, states = integrate(key,  q_flat, t0, tf, nt, ndt, noise, module_infos, module_params, mr_regime=self.regime)
            return user_fitness(traj, states)

        grad_fn = jit(grad(fitness), static_argnames=['nt', 'ndt']) # nt and ndt are used as size to create arrays, hence we need to make them static to jit the function
        fitness_vals = []

        for _ in range(steps):
            fitness_vals.append(fitness(module_params, module_infos, key,  q_flat, t0, tf, nt, ndt, noise))
            grad_ = grad_fn(module_params, module_infos, key,  q_flat, t0, tf, nt, ndt, noise)
            module_params = tree_util.tree_map(
                lambda p, g : p - lr * g,
                module_params,
                grad_
            )

        self.module_params = module_params

        return fitness_vals