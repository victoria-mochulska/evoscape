import jax.random as jrd
import jax.numpy as jnp
import jax.tree_util as jtu

from jax import value_and_grad, jit
from jax.lax import scan

# Clipping function for non negative value allowed. Clipping values should be chosen wisely 
def clip_dynamic(dynamic):
    def clip_fn(path, x):
        key = path[-1]

        name = getattr(key, "name", None)

        # if name == "a":
        #     return jnp.clip(x, 2, 2)
        if name == "s":
            return jnp.clip(x, 1, 5)
        # if name == "x":
        #     return jnp.clip(x, -5, 5)
        # if name == "y":
        #     return jnp.clip(x, -5, 5)
        # if name == "init_cond":
        #     return jnp.clip(x, 2, 2)
        return x 

    return jtu.tree_map_with_path(clip_fn, dynamic)

# Gradient Descent 
def make_step(loss_fn, η, clip_fn):

    grad_fn = value_and_grad(loss_fn)

    def step(carry, _):
        dynamic, key = carry

        key, subkey = jrd.split(key)

        loss, grads = grad_fn(dynamic, subkey)

        dynamic = jtu.tree_map(
            lambda x, dx: x - η * dx,
            dynamic,
            grads
        )

        dynamic = clip_fn(dynamic)

        return (dynamic, key), loss
    
    return step

# Training function
@jit(static_argnames=('step_fn','iterations'))
def run_optimization(dynamic, key, step_fn, iterations):
    (dynamic, key), losses = scan(
        step_fn,
        (dynamic, key),
        None,
        length=iterations
    )
    return (dynamic, key), losses
