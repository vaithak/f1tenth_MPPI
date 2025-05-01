import numpy as np
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from functools import partial
from f1tenth_mppi import jax_utils
from numba import njit
from f1tenth_mppi.jax_dynamics_models import vehicle_dynamics_ks, vehicle_dynamics_st
from f1tenth_mppi.mppi_config import mppi_config
from f1tenth_mppi.env import Environment


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class MPPI():
    """
    An MPPI class that implements the MPPI algorithm for controlling a vehicle.
    """
    def __init__(self):
        self.state = State()
        self.a_cov = None
        self.jrng = jax_utils.oneLineJaxRNG(527787)
        self.n_steps = mppi_config.TK
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps)))
        self.n_samples = mppi_config.NUM_SAMPLES
        self.temperature = 0.01
        self.damping = 0.001


    def init_state(self, env, a_shape):
        """
        Initialize the state of the MPPI algorithm.
        """
        self.a_shape = a_shape
        dim_a = np.prod(a_shape)
        self.env = env
        self.state = State()
        self.a_cov = None
        self.a_opt = 0.0 * jax.random.uniform(self.jrng.new_key(), shape=(self.n_steps,
                                                dim_a))  # [n_steps, dim_a]
        

    def update(self, curr_state, ref_traj):
        """
        Update the MPPI algorithm with the current state and reference trajectory.
        """
        self.a_opt = self.shift_prev_opt(self.a_opt)
        for i in range(mppi_config.N_ITERATIONS):
            # Sample new control inputs
            self.a_opt, self.traj_opt = self.iteration_step(
                self.a_opt, self.jrng.new_key(), curr_state, ref_traj
            )


    @partial(jax.jit, static_argnums=(0))
    def shift_prev_opt(self, a_opt):
        a_opt = jnp.concatenate([a_opt[1:, :],
                                jnp.expand_dims(jnp.zeros((self.a_shape,)),
                                                axis=0)])  # [n_steps, a_shape]
        return a_opt
    
    # @partial(jax.jit, static_argnums=(0))
    # def rollout(self, actions, curr_state, rng_key):
    #     """
    #     Perform a rollout of the MPPI algorithm.
    #     """
    #     def rollout_step(curr_state, actions, rng_key):
    #         actions = jnp.reshape(actions, self.a_shape)
    #         curr_state = self.env.step(curr_state, actions, rng_key)
    #         return curr_state
    #     # Perform the rollout
    #     states = jax.vmap(
    #         rollout_step, in_axes=(None, 0, None))(curr_state, actions, rng_key)
        
    #     return jnp.asarray(states)
    
    @partial(jax.jit, static_argnums=0)
    def rollout(self, actions, curr_state, rng_key):
        """
        # actions: [n_steps, a_shape]
        # env: {.step(states, actions), .reward(states)}
        # env_state: np.float32
        # actions: # a_0, ..., a_{n_steps}. [n_steps, a_shape]
        # states: # s_1, ..., s_{n_steps+1}. [n_steps, env_state_shape]
        """
    
        def rollout_step(curr_state, actions, rng_key):
            actions = jnp.reshape(actions, self.a_shape)
            curr_state = self.env.step(curr_state, actions, rng_key)
            return curr_state
        
        states = []
        for t in range(self.n_steps):
            curr_state = rollout_step(curr_state, actions[t, :], rng_key)
            states.append(curr_state)
            
        return jnp.asarray(states)
    
    @partial(jax.jit, static_argnums=(0))
    def iteration_step(self, a_opt, rng_key, curr_state, ref_traj):
        """
        Perform one iteration of the MPPI algorithm.
        """
        rng_da, rng_da_split1, rng_da_split2 = jax.random.split(rng_key, 3)
        da = jax.random.truncated_normal(
            rng_da,
            -jnp.ones_like(a_opt) * mppi_config.ACTION_STD - a_opt,
            jnp.ones_like(a_opt) * mppi_config.ACTION_STD - a_opt,
            shape=(self.n_samples, self.n_steps, self.a_shape),
            dtype=jnp.float32,
        ) # [n_samples, n_steps, a_shape]

        actions = jnp.expand_dims(a_opt, axis=0) + da  # [n_samples, n_steps, a_shape]
        actions = jnp.clip(da, -1.0, 1.0)
        states = jax.vmap(
            self.rollout,
            in_axes=(0, None, None))(
                actions, curr_state, rng_da_split1)
        
        # Calculate the rewards for each state in the rollout
        rewards = jax.vmap(
            self.env.reward_fn_xy,
            in_axes=(0, None))(
                states, ref_traj)
        
        R = jax.vmap(self.returns)(rewards) # [n_samples, n_steps], pylint: disable=invalid-name
        w = jax.vmap(self.weights, 1, 1)(R)  # [n_samples, n_steps]
        da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [n_steps, dim_a]
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)
        # Calculate the optimal trajectory
        traj_opt = self.rollout(a_opt, curr_state, rng_da_split2)  # [n_steps, dim_a]
        
        return a_opt, traj_opt
        
    
    @partial(jax.jit, static_argnums=(0))
    def returns(self, r):
        # r: [n_steps]
        return jnp.dot(self.accum_matrix, r)  # R: [n_steps]


    @partial(jax.jit, static_argnums=(0))
    def weights(self, R):  # pylint: disable=invalid-name
        # R: [n_samples]
        # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
        # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
        R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)  # pylint: disable=invalid-name
        w = jnp.exp(R_stdzd / self.temperature)  # [n_samples] np.float32
        w = w/jnp.sum(w)  # [n_samples] np.float32
        return w