from evoscape.jax.flax_models.landscape_flax import LandscapeFlax
from evoscape.jax.flax_models.autoencoder_flax import AutoEncoder
from evoscape.jax.flax_models.experiment_manager import ExperimentManager
import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx
from evoscape.landscapes import Landscape
from evoscape.modules import Node
from evoscape.morphogen_regimes import mr_const
from evoscape.jax.dynamics import state_probs
from evoscape.jax.config import DATA_DIR
from evoscape.jax.losses import mmd_traj

# -----------------------------------------------
# FIXED SETUP
# -----------------------------------------------

def create_random_modules(n_modules):
    rng = np.random.default_rng(42)
    x_coord = rng.normal(size=n_modules)
    y_coord = rng.normal(size=n_modules)
    return [Node(x=x, y=y, a=np.array([1.0]), s=np.array([1.0]), tau=1.) for x, y in zip(x_coord, y_coord)]

filepath = DATA_DIR / "synthetic_data" / "synthetic_data.npy"
target_traj = np.load(filepath)
target_traj = np.transpose(target_traj, (2, 0, 1))
indices = np.random.choice(target_traj.shape[1], size=200, replace=False)
target_traj = target_traj[:, indices, ::100]

dims_encoder = [4, 4, 2]
dims_decoder = [2, 4, 4]
n_epochs     = 1000
nt           = 20
ndt          = 100
t0           = 0.0

manager = ExperimentManager("experiments_tf_noise/")

# -----------------------------------------------
# GRID
# -----------------------------------------------

noise_values = np.arange(0.05, 0.55, 0.05).round(2).tolist()  # 0.05 to 0.50, step 0.05
tf_values    = np.geomspace(1, 2000, 10).round(1).tolist()     # ~log scale from 1 to 2000

# -----------------------------------------------
# LOSS
# -----------------------------------------------

def loss_fn(autoencoder, target_traj):
    q_init = target_traj[:, :, 0]
    simulated_traj = autoencoder(q_init)
    loss_dynamics = mmd_traj(simulated_traj, target_traj)
    encoded_decoded_traj = autoencoder.forward_no_dynamics(target_traj)
    loss_encoding = jnp.sum((encoded_decoded_traj - target_traj) ** 2)
    return loss_dynamics + 1e-3 * loss_encoding

@nnx.jit
def train_step(autoencoder, optimizer, target_traj):
    loss, grads = nnx.value_and_grad(loss_fn)(autoencoder, target_traj)
    optimizer.update(autoencoder, grads)
    return loss

# -----------------------------------------------
# GRID SEARCH
# -----------------------------------------------

# Creating the landscape

modules = [
    Node(x=0., y=0., a=np.array([0.5]), s=np.array([0.5]), tau=1.),
    Node(x=-1., y=-1., a=np.array([0.5]), s=np.array([0.5]), tau=1.),
    Node(x=1., y=-1., a=np.array([0.5]), s=np.array([0.5]), tau=1.),
    Node(x=0., y=-2., a=np.array([0.5]), s=np.array([0.5]), tau=1.),
    Node(x=2., y=-2., a=np.array([0.5]), s=np.array([0.5]), tau=1.)
]

landscape = Landscape(module_list=modules, A0=0.05, init_cond=(0.0, 0.0), regime=mr_const, n_regimes=1)

total = len(noise_values) * len(tf_values)
run = 0


for noise in noise_values:
    for tf in tf_values:
        run += 1
        print(f"\n[{run}/{total}] noise={noise:.2f}  tf={tf:.1f}")

        # fresh model for each run
        rngs         = nnx.Rngs(0)
        landscape_flax = LandscapeFlax(landscape, rngs)

        landscape_flax.set_simulation(init_noise=noise, t0=t0, tf=tf, nt=nt, ndt=ndt, noise=noise)
        landscape_flax.set_regime_params(signal_param=None)
        landscape_flax.set_state_probs(state_probs)

        autoencoder = AutoEncoder(landscape_flax, dims_decoder=dims_decoder, dims_encoder=dims_encoder, rngs=rngs)

        tx = optax.adamw(
            learning_rate = 1e-3,
            weight_decay = 1e-4
        )
        optimizer = nnx.Optimizer(autoencoder, tx, wrt=nnx.Param)

        loss_vals = []
        checkpoints = {}
        for epoch in range(n_epochs):
            loss_val = train_step(autoencoder, optimizer, target_traj)
            loss_vals.append(loss_val)

            _, state = nnx.split(autoencoder)
            graphdef, _ = nnx.split(autoencoder)
            checkpoints[epoch] = nnx.merge(graphdef, state)  

            if epoch % 50 == 0:
                print(f"  epoch {epoch:3d} | loss {loss_val:.4f}")

        config = {
            "dims_encoder":        dims_encoder,
            "dims_decoder":        dims_decoder,
            "n_modules":           len(modules),
            "t0":                  t0,
            "tf":                  tf,
            "nt":                  nt,
            "ndt":                 ndt,
            "noise":               noise,
            "init_noise":          noise,
            "optimizer":           "adam",
            "lr":                  1e-2,
            "loss_encoding_weight": 1e-3,
            "data_path":           str(filepath),
            "n_cells":             200,
            "data_seed":           42,
            "n_epochs":            n_epochs,
        }

        name = f"noise{noise:.2f}_tf{tf:.1f}".replace(".", "p")  # e.g. noise0p10_tf200p0

        manager.save(
            model=autoencoder,
            config=config,
            loss_names=["mmd", "encoding"],
            loss_vals=loss_vals,
            name=name,
            notes=f"gridsearch — noise={noise:.2f}, tf={tf:.1f}",
            checkpoints=checkpoints
        )