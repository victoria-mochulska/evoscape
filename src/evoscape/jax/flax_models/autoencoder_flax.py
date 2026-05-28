from flax import nnx
import jax.numpy as jnp
from ..landscape_flax import LandscapeFlax
from .mlp import MLP


class AutoEncoder(nnx.Module):

    def __init__(self, landscape_flax, dims_decoder, dims_encoder, rngs):

        """
        In this first version of the decoder, the architecture is :
        - Input (d) (a cell represented in d dimensions)
        - Encoder MLP(d,2) 
        - Landscape (that gives us the trajectory of the cell)
        - Decoder MLP(2,d) (that is applied on all the trajectory)

        """

        self.encoder = MLP(dims_encoder, rngs)
        self.landscape_flax = landscape_flax
        self.decoder = MLP(dims_decoder, rngs)

        self.rngs = rngs


    def __call__(self, q_init):
        
        # encoding the initial condition
        q_init_encoded = nnx.vmap(lambda x : self.encoder(x), in_axes=1, out_axes=1)(q_init) # (2, n)

        # computing the trajectories with the landscapes
        traj, states = self.landscape_flax(q_init_encoded) # (2, n, nt)

        # decoding the trajectory
        # We need to apply the decoder on the last two dimensions of the traj array
        vdecoder = nnx.vmap(
                nnx.vmap(
                    lambda x : self.decoder(x),
                    in_axes=1,
                    out_axes=1
                ),
                in_axes=2,
                out_axes=2
        )

        traj_decoded = vdecoder(traj) # (d, n, nt)

        return traj_decoded

    # def _decode_traj_landscape(self, traj):
    #     # Our mlp wants (n, 2), and we have (2, n, nt) so we need to convert the trajectory, decode it, then reconvert it
    #     # Maybe this could be done much more efficiently with vmap ? But this works for now

    #     traj_permuted = jnp.transpose(traj, (1, 2, 0)) # (n, nt, 2)
    #     n, nt, _ = traj_permuted.shape

    #     traj_permuted_batched = jnp.reshape(traj_permuted, (-1, 2)) # (n*nt, 2)
    #     traj_permuted_decoded_batched = self.decoder(traj_permuted_batched) # (n*nt, d)
    #     traj_permuted_decoded = jnp.reshape(traj_permuted_decoded_batched, (n, nt, self.dims_decoder[-1])) # (n, nt, d)

    #     traj_decoded = jnp.transpose(traj_permuted_decoded, (2, 0, 1)) # (d, n, nt)

    #     return traj_decoded

    
    # def call_v2(self, q_init):
    #     """
    #     q_init : array (d, n), the initial cells
    #     returns : array(d, n, nt) the trajectory of the initial cells

    #     """

    #     q_init_encoded = self.encoder(q_init.transpose(1,0)).transpose(1,0)

    #     traj, states = self.landscape_flax(q_init_encoded)

    #     traj_decoded = self._decode_traj_landscape(traj)

    #     return traj_decoded





