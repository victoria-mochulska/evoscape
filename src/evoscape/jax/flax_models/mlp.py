from flax import nnx


class MLP(nnx.Module):
    def __init__(self, dims, rngs):

        self.rngs = rngs
        self.dims_encoder = dims

        layers_encoder = [] 
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers_encoder.append(nnx.Linear(in_dim, out_dim, rngs=self.rngs))
            layers_encoder.append(nnx.relu) 
        layers_encoder.pop() # last layer must be only linear, not linear + relu 
        
        self.net = nnx.Sequential(*layers_encoder)

    def __call__(self, x):
        return self.net(x)

