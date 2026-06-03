from typing import Any, Callable, Dict
import numpy as np

from flax import nnx
import jax.numpy as jnp
import jax.random as jrd

from ..converters import landscape_to_pytree, pytree_to_landscape
from ..dynamics import _integrate
from ..regimes import wrapped_regime
from ..types import LandscapeDynamic, ModuleDynamic


# LandscapeDynamic and ModuleDynamic as flax classes

class ModuleDynamicFlax(nnx.Module):

    def __init__(self, module : ModuleDynamic):

        self.x = nnx.Param(module.x)
        self.y = nnx.Param(module.y)

        self.a = nnx.Param(module.a)
        self.s = nnx.Param(module.s)


class LandscapeDynamicFlax(nnx.Module):

    def __init__(self, dynamic : LandscapeDynamic):
        self.module = ModuleDynamicFlax(dynamic.module)



class LandscapeFlax(nnx.Module):

    def __init__(self, landscape: Any, rngs : nnx.Rngs):

        dynamic, static = landscape_to_pytree(landscape)

        # ==========================================================
        # TRAINABLE PARAMETERS
        # ==========================================================

        self.dynamic = LandscapeDynamicFlax(dynamic)

        # ==========================================================
        # CONFIGURATION STATIQUE
        # ==========================================================

        self.static = nnx.data(static) # cette ligne est tellement aberrante mdr

        self.sim_params: Dict[str, Any] = nnx.static({})

        self.get_cell_states = nnx.static(None)
        self.signal_param = nnx.data(None)
        self.regime = nnx.static(None)

        self.rngs = rngs

    # ==============================================================
    # SETTERS
    # ==============================================================

    def set_regime_params(self, signal_param=None):
        self.signal_param = nnx.data(signal_param)
        self.regime = nnx.static(
            wrapped_regime(
                self.static,
                self.signal_param
            )
        )

    def set_simulation(
        self,
        init_noise,
        t0,
        tf,
        nt,
        ndt,
        noise,
    ):

        self.sim_params = nnx.static(
            dict(
                init_noise=init_noise,
                t0=t0,
                tf=tf,
                nt=nt,
                ndt=ndt,
                noise=noise,
            )
        )

    def set_state_probs(self, get_cell_states):
        self.get_cell_states = nnx.static(get_cell_states)

    # ==============================================================
    # FORWARD
    # ==============================================================

    def __call__(self, q_init):
        """
        q_init : array (2,n)
        """

        key = self.rngs.default()

        p = self.sim_params

        # adding noise to the initial condition
        q_noisy = q_init + jrd.normal(self.rngs.default(), shape=q_init.shape) * p["init_noise"]

        _, traj, states = _integrate(
            key,
            q_noisy,
            p["t0"],
            p["tf"],
            p["nt"],
            p["ndt"],
            p["noise"],
            self.dynamic,
            self.static,
            self.regime,
            self.get_cell_states,
        )

        return traj, states

    # ==============================================================
    # GET LANDSCAPE
    # ==============================================================
    
    def get_landscape(self):
        trajectories = None
        cell_coordinates = None
        cell_states = None
        result = None
        fitness = None

        return pytree_to_landscape(
            self.dynamic,
            self.static,
            trajectories=trajectories,
            cell_coordinates=cell_coordinates,
            cell_states=cell_states,
            fitness=fitness,
            result=result,
        )