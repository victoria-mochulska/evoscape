import jax.random as jrd

# Initialization function used in the fitness function, we let the model choose where the cells should start in optimization
def init_cell(key,n,init_cond,noise):                    
    key,subkey = jrd.split(key)
    return  key,init_cond[:,None]+noise*jrd.normal(subkey, shape=(2, n))