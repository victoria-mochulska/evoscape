from functools import partial
import dataclasses
import jax.random as jrd
import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import tqdm
import logging
from jax import value_and_grad, jit
from jax.lax import scan

import optax
from flax import nnx

# Clipping function for non negative value allowed. Clipping values should be chosen wisely
def clip_dynamic(dynamic):
    def clip_fn(path, x):
        names = [k.name for k in path if hasattr(k, "name")]

        if "a" in names:
            return jnp.clip(x, 0.0, 20)
        elif "s" in names:
            return jnp.clip(x, 0.0, 5)
        elif "x" in names:
            return jnp.clip(x, -10, 10)
        elif "y" in names:
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

        return (dynamic, key), (loss, dynamic)

    return step

# Training function
@partial(jit, static_argnames=("step_fn", "iterations"))
def run_optimization(dynamic, key, step_fn, iterations):
    (dynamic, key), (losses, dynamics) = scan(
        step_fn,
        (dynamic, key),
        None,
        length=iterations
    )
    return (dynamic, key), losses, dynamics


def run_optimization_optax(dynamic, key, steps, loss_fn, optimizer):
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
        dynamic = clip_dynamic(dynamic)

        # saving losses and current landscape
        fitness_vals.append(loss)
        dynamic_vals.append(dynamic)

        if step%20 == 0:
            print(f"Train step {step} : Loss = {loss}")
        
    return dynamic_vals, fitness_vals
        
def run_optimization_decoder(dynamic, decoder, key, steps, loss_fn, opt_dynamic, opt_decoder):
    """
    For now, dynamic and decoder are separated models, so we need one optimizer for each.
    Both optimizer are from optax, but we use nnx.Optimizer with opt_decoder to benefit from nnx simplicity

    As a result, the code uses a sort of hybrid approach that may need to be changed later 
    (for example, creating a class landscape that inherits from nnx.Module ?)

    """
    opt_dynamic_state = opt_dynamic.init(dynamic)
    opt_decoder = nnx.Optimizer(decoder, opt_decoder, wrt=nnx.Param) # optax.optimizer -> nnx.Optimizer

    fitness_vals = []
    dynamic_vals = []

    value_and_grad_fn = nnx.jit(nnx.value_and_grad(loss_fn, argnums=(0,1)))

    for step in range(steps):
        # one optimizer step
        key, subkey = jrd.split(key)

        loss, (grads_dynamic, grads_decoder) = value_and_grad_fn(dynamic, decoder, subkey)

        updates_dynamic, opt_dynamic_state = opt_dynamic.update(grads_dynamic, opt_dynamic_state)
        dynamic = optax.apply_updates(dynamic, updates_dynamic)
        dynamic = clip_dynamic(dynamic)

        opt_decoder.update(decoder, grads_decoder)

        # saving losses and current landscape
        fitness_vals.append(loss)
        dynamic_vals.append(dynamic)

        if step%20 == 0:
            print(f"Train step {step} : Loss = {loss}")
        
    return dynamic_vals, fitness_vals




@dataclasses.dataclass
class RunSet:
    """A batch of runs sharing one graph definition.

    params / opt_state carry the run axis on axis 0; losses has shape
    (n_epochs, n_runs); init_losses has shape (n_runs,).
    """
    graphdef: object
    rest: object
    params: object
    opt_state: object
    losses: np.ndarray
    init_losses: np.ndarray

    def model(self, i):
        """Unbatched nnx.AutoEncoder for run i (for analysis / plotting)."""
        params_i = jax.tree.map(lambda v: v[i], self.params)
        return nnx.merge(self.graphdef, params_i, self.rest)


def make_optimizer(
    model,
    learning_rate=1e-3,
    train_encoder=True,
    train_landscape=True,
    train_decoder=True,
):

    filters = []

    if train_encoder:
        filters.append(nnx.PathContains("encoder"))

    if train_landscape:
        filters.append(nnx.PathContains("landscape_flax"))

    if train_decoder:
        filters.append(nnx.PathContains("decoder"))

    if len(filters) == 0:
        raise ValueError("At least one part of the model must be trainable.")

    train_filter = nnx.Any(*filters)

    return nnx.Optimizer(
        model,
        optax.adamw(learning_rate),
        wrt=train_filter,
    )


def make_wrt(
    train_encoder=True,
    train_landscape=True,
    train_decoder=True,
):

    filters = []

    if train_encoder:
        filters.append(nnx.PathContains("encoder"))

    if train_landscape:
        filters.append(nnx.PathContains("landscape_flax"))

    if train_decoder:
        filters.append(nnx.PathContains("decoder"))

    if len(filters) == 0:
        raise ValueError("At least one part of the model must be trainable.")

    train_filter = nnx.Any(*filters)

    return train_filter




def train_runs(models, target, lr, n_epochs, verbose_dt):
    """Train all runs with one vmap'd optimizer step per epoch.

    Epochs are grouped in blocks of `verbose_dt` (one lax.scan each); the
    progress bar and loss collection happen once per block, so there is one
    host-device sync per block instead of one per epoch.
    target: (4, ns, nt) jax array of gene data.
    """
    graphdefs, params_list, rests = zip(*[nnx.split(m, nnx.Param, ...) for m in models])
    batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)
    graphdef, rest = graphdefs[0], rests[0]

    q_init = target[:, :, 0]
    ns, nt = target.shape[1], target.shape[2]

    def loss_fn(params_i):
        model = nnx.merge(graphdef, params_i, rest)
        z = model.encoder(q_init.T)                    # (ns, 4) -> (ns, 2)
        z = z.T                                        # (2, ns)
        traj, _states = model.landscape_flax(z)        # (2, ns, nt)
        decoded = model.decoder(traj.reshape(2, -1).T) # (ns*nt, 2) -> (ns*nt, 4)

        encoded_data = model.encode_traj(target)
        simulated_latent = model.get_latent_trajectory(q_init)
        loss_latent_space = jnp.mean((simulated_latent - encoded_data)**2)

        recon_data = model.forward_no_dynamics(target)
        loss_reconst = jnp.mean((recon_data - target)**2)
        dynamic_param = model.landscape_flax.dynamic.module
        loss_module = jnp.sum(jnp.abs(dynamic_param.a))

        state_autoencoder = nnx.state([model.encoder, model.decoder], nnx.Param)

        L1_norm_autoencoder = sum(
            jnp.sum(jnp.abs(param))
            for param in jax.tree.leaves(state_autoencoder)
        )
        

        """JEb = jax.vmap(lambda v: jax.jacobian(lambda w: model.encoder(w[None])[0])(v))(target.reshape(-1, 4))   # (N, 2, 4)
        JDb = jax.vmap(lambda v: jax.jacobian(lambda w: model.decoder(w[None])[0])(v))(model.encoder(target.reshape(-1, 4)))  # (N, 4, 2)
        G = jnp.einsum("pij,pjk->pik", JEb, JDb)   # (N, 2, 2)
        loss_commutation = jnp.mean(jnp.sum((G - jnp.eye(2))**2, axis=(-2, -1)))"""
        
        return jnp.mean((decoded.T.reshape(4, ns, nt) - target) ** 2) + loss_latent_space + loss_reconst + loss_module + L1_norm_autoencoder

    tx = optax.adam(learning_rate=lr)
    # vmap the init: tx.init(batched) leaves the adam `count` counter scalar
    # (shape ()), which vmap(step) then rejects.
    opt_state = jax.vmap(tx.init, in_axes=0)(batched)

    def step(params_i, opt_i):
        loss, grads = jax.value_and_grad(loss_fn)(params_i)
        updates, opt_new = tx.update(grads, opt_i, params_i)
        return optax.apply_updates(params_i, updates), opt_new, loss

    vmapped_step = jax.jit(jax.vmap(step, in_axes=(0, 0), out_axes=(0, 0, 0)))
    vmapped_loss = jax.jit(jax.vmap(loss_fn))

    def block_scan(carry, _):
        params_i, opt_i, alive = carry
        params_new, opt_new, loss = vmapped_step(params_i, opt_i)

        ok = jnp.isfinite(loss)

        def freeze(new, old):
            return jax.tree.map(
                lambda nw, od: jnp.where(ok[(slice(None),) + (None,) * (nw.ndim - 1)], nw, od),
                new, old)

        params_new = jax.tree.map(freeze, params_new, params_i)
        opt_new = jax.tree.map(freeze, opt_new, opt_i)
        loss = jnp.where(ok, loss, jnp.inf)

        return (params_new, opt_new, alive & ok), loss


    

    init_losses = np.asarray(vmapped_loss(batched))
    params, losses = batched, []
    alive = jnp.ones(len(models), dtype=bool)
    pbar = tqdm.tqdm(total=n_epochs)
    for start in range(0, n_epochs, max(1, verbose_dt)):
        block_len = min(max(1, verbose_dt), n_epochs - start)
        (params, opt_state, alive), block_losses = jax.lax.scan(
            block_scan, (params, opt_state, alive), length=block_len)
        losses.append(np.asarray(block_losses))
        pbar.update(block_len)
        last = block_losses[-1]
        live = np.isfinite(last)
        pbar.set_postfix({"loss": f"{np.mean(last[live]):.4e}", "dead": (~live).sum()}) 


        newly_dead = ~live & alive                # alive before this block, dead after it
        if newly_dead.any():
            logging.warning("run(s) %s discarded at epoch ~%d",
                            np.where(newly_dead)[0].tolist(), start + block_len)
        alive = live                              # update for the next block's transition check
            
    pbar.close()

    return RunSet(graphdef, rest, params, opt_state, np.stack(losses), init_losses)