from functools import partial

import jax.random as jrd
import jax.numpy as jnp
import jax.tree_util as jtu

from jax import value_and_grad, jit
from jax.lax import scan

import optax

# Clipping function for non negative value allowed. Clipping values should be chosen wisely
def clip_dynamic(dynamic):
    def clip_fn(path, x):
        key = path[-1]

        name = getattr(key, "name", None)

        if name == "a":
            return jnp.clip(x, 0.5, 20)
        if name == "s":
            return jnp.clip(x, 1, 5)
        if name == "x":
            return jnp.clip(x, -10, 10)
        if name == "y":
            return jnp.clip(x, -10, 10)
        return x

    return jtu.tree_map_with_path(clip_fn, dynamic)


# Gradient Descent
def make_step(loss_fn, lr, clip_fn):
    grad_fn = value_and_grad(loss_fn)

    def step(carry, _):
        dynamic, key = carry

        key, subkey = jrd.split(key)

        loss, grads = grad_fn(dynamic, subkey)

        dynamic = jtu.tree_map(
            lambda x, dx: x - lr * dx,
            dynamic,
            grads
        )

        dynamic = clip_fn(dynamic)

        return (dynamic, key), loss

    return step

# Training function
@partial(jit, static_argnames=("step_fn", "iterations"))
def run_optimization(dynamic, key, step_fn, iterations):
    (dynamic, key), losses = scan(
        step_fn,
        (dynamic, key),
        None,
        length=iterations
    )
    return (dynamic, key), losses


def run_optimization_optax(dynamic, key, steps, loss_fn, optimizer, lr):
    optimizer = optimizer(lr)
    opt_state = optimizer.init(dynamic)
    fitness_vals = []
    dynamic_vals = []

    value_and_grad_fn = jit(value_and_grad(loss_fn))

    for step in range(steps):
        # one optimizer step
        key, subkey = jrd.split(key)

        loss, grads = value_and_grad_fn(dynamic, subkey)
        updates, opt_state = optimizer.update(grads, opt_state)
        dynamic = optax.apply_updates(dynamic, updates)

        # saving losses and current landscape
        fitness_vals.append(loss)
        dynamic_vals.append(dynamic)

        if step%5 == 0:
            print(f"Train step {step} : Loss = {loss}")
        
    return dynamic_vals, fitness_vals
        