import jax
from jax import random


class oneLineJaxRNG:
    def __init__(self, init_num=0) -> None:
        self.rng = jax.random.PRNGKey(init_num)
    
    def new_key(self):
        self.rng, key = random.split(self.rng)
        return key
    
def numpify(x):
    return jax.device_get(x)
