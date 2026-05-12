import jax.numpy as jnp

from typing import NamedTuple


class ModuleStatic(NamedTuple):
    #size (n_modules,2,2)
    J: jnp.ndarray

    #size (n_modules,)
    tau: jnp.ndarray
    use_tau: jnp.ndarray


class ModuleDynamic(NamedTuple):
    #size (n_modules,)
    x: jnp.ndarray
    y: jnp.ndarray
    #size (n_modules,n_regimes)
    a: jnp.ndarray
    s: jnp.ndarray


class LandscapeStatic(NamedTuple):
    A0: float
    x0: jnp.ndarray
    n_regimes: int
    regime_id: int #the number associated with the regime
    morphogen_times: jnp.ndarray
    init_cond: jnp.ndarray
    module: ModuleStatic


class LandscapeDynamic(NamedTuple):
    module: ModuleDynamic
