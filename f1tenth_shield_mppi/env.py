import numpy as np
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from f1tenth_shield_mppi.jax_dynamics_models import vehicle_dynamics_ks, vehicle_dynamics_st
from f1tenth_shield_mppi.mppi_config import mppi_config
from numba import njit


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / (l2s + 1e-8)
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


class Environment:
    def __init__(
            self,
            waypoint_file = None,
        ):
        self.mppi_config = mppi_config()
        self.friction = 0.8

        self.waypoints = np.genfromtxt(waypoint_file, delimiter='; ', skip_header=1)
        waypoint_cols_dict = self.column_numbers_for_waypoints()
        self.yaws = self.waypoints[:, waypoint_cols_dict['psi_racetraj_rad']]
        self.yaws = np.array(self.convert_psi(self.yaws))
        self.speed_profile = self.waypoints[:, waypoint_cols_dict['vx_racetraj_mps']]
        self.waypoints_xs = self.waypoints[:, waypoint_cols_dict['x_ref_m']]
        self.waypoints_ys = self.waypoints[:, waypoint_cols_dict['y_ref_m']]

        self.d_right_array = self.waypoints[:, waypoint_cols_dict['width_right_m']]
        self.d_left_array = self.waypoints[:, waypoint_cols_dict['width_left_m']]
        self.s_array = self.waypoints[:, waypoint_cols_dict['s_racetraj_m']]
        self.state_predictor = 'dynamic_ST'

        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)


        def RK4_fn(x0, u, Ddt, vehicle_dynamics_fn, args):
            # return x0 + vehicle_dynamics_fn(x0, u, *args) * Ddt # Euler integration
            # RK4 integration
            k1 = vehicle_dynamics_fn(x0, u, *args)
            k2 = vehicle_dynamics_fn(x0 + k1 * 0.5 * Ddt, u, *args)
            k3 = vehicle_dynamics_fn(x0 + k2 * 0.5 * Ddt, u, *args)
            k4 = vehicle_dynamics_fn(x0 + k3 * Ddt, u, *args)
            return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
            
        # if self.config.state_predictor == 'dynamic_ST':
        @jax.jit
        def update_fn(x, u):
            x1 = x.copy()
            Ddt = 0.05
            def step_fn(i, x0):
                args = (self.friction,)
                return RK4_fn(x0, u, Ddt, vehicle_dynamics_st, args)
            x1 = jax.lax.fori_loop(0, int(mppi_config.DTK/Ddt), step_fn, x1)
            # return (x1, 0, x1-x)
            return x1
        self.update_fn = update_fn
            
        # elif self.config.state_predictor == 'kinematic_ST':
        #     @jax.jit
        #     def update_fn(x, u):
        #         x_k = x.copy()[:5]
        #         Ddt = 0.05
        #         def step_fn(i, x0):
        #             args = ()
        #             return RK4_fn(x0, u, Ddt, vehicle_dynamics_ks, args)
        #         x_k = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x_k)
        #         x1 = x.at[:5].set(x_k)
        #         return (x1, 0, x1-x)
        #     self.update_fn = update_fn


    @staticmethod
    def column_numbers_for_waypoints():
        """
        Returns the column numbers for the waypoints.
        """
        return {
            "x_ref_m": 0,
            "y_ref_m": 1,
            "width_right_m": 2,
            "width_left_m": 3,
            "x_normvec_m": 4,
            "y_normvec_m": 5,
            "alpha_m": 6,
            "s_racetraj_m": 7,
            "psi_racetraj_rad": 8,
            "kappa_racetraj_radpm": 9,
            "vx_racetraj_mps": 10,
            "ax_racetraj_mps2": 11
        }

    @staticmethod
    def convert_psi(psi):
        """
        Converts the psi angle to the range [-pi, pi].
        """
        new_psi = psi + np.pi / 2

        # Bring the angle to the range [-pi, pi] in a vectorized way
        new_psi[new_psi > np.pi] -= 2 * np.pi

        return new_psi

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, rng_key=None, dyna_norm_param=None):
        """
        Update the state of the vehicle using the given action.
        :param state: Current state of the vehicle.
        :param action: Action to be taken.
        :param rng_key: Random key for JAX operations.
        :return: Updated state of the vehicle.
        """
        return self.update_fn(state, action)

    # def calc_ref_trajectory(self, state):
    #     """
    #     calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
    #     using the current velocity, calc the T points along the reference path
    #     :return: reference trajectory ref_traj
    #     """
    #     cx = self.waypoints_xs
    #     cy = self.waypoints_ys
    #     cyaw = self.yaws
    #     sp = self.speed_profile

    #     # Create placeholder Arrays for the reference trajectory for T steps
    #     ref_traj = np.zeros((self.mppi_config.NXK, self.mppi_config.TK + 1))
    #     ncourse = len(cx)

    #     # Find nearest index/setpoint from where the trajectories are calculated
    #     _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

    #     # Load the initial parameters from the setpoint into the trajectory
    #     ref_traj[0, 0] = cx[ind]
    #     ref_traj[1, 0] = cy[ind]
    #     ref_traj[2, 0] = sp[ind]
    #     ref_traj[3, 0] = cyaw[ind]

    #     # based on current velocity, distance traveled on the ref line between time steps
    #     travel = abs(state.v) * self.mppi_config.DTK
    #     dind = travel / self.mppi_config.dlk
    #     ind_list = int(ind) + np.insert(
    #         np.cumsum(np.repeat(dind, self.mppi_config.TK)), 0, 0
    #     ).astype(int)
    #     ind_list[ind_list >= ncourse] -= ncourse
    #     ref_traj[0, :] = cx[ind_list]
    #     ref_traj[1, :] = cy[ind_list]
    #     ref_traj[2, :] = sp[ind_list]
    #     cyaw[cyaw - state.yaw > 3.15] = cyaw[cyaw - state.yaw > 3.15] - (2 * np.pi)
    #     cyaw[cyaw - state.yaw < -3.15] = cyaw[cyaw - state.yaw < -3.15] + (2 * np.pi)
    #     ref_traj[3, :] = cyaw[ind_list]
    #     return ref_traj


    @partial(jax.jit, static_argnums=(0,))
    def reward_fn_xy(self, state, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        xy_cost = -jnp.linalg.norm(reference[1:, :2] - state[:, :2], ord=1, axis=1)
        vel_cost = -jnp.linalg.norm(reference[1:, 2] - state[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(state[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(state[:, 4]))
            
        # return 20*xy_cost + 15*vel_cost + 1*yaw_cost
        return xy_cost
    
    def get_refernece_traj(self, state, target_speed=None, n_steps=10, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy())
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            # speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(n_steps) * speed
        
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(n_steps),
                                            self.waypoints_distances.copy(), DT=self.mppi_config.DTK)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind


# @njit(cache=True)
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    s_relative = np.zeros((n_steps + 1,))
    s_relative[0] = dist_from_segment_start
    s_relative[1:] = predicted_speeds * DT
    s_relative = np.cumsum(s_relative)

    waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

    index_relative = np.int_(np.ones((n_steps + 1,)))
    for i in range(n_steps + 1):
        index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
    index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

    segment_part = s_relative - (
            waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

    t = (segment_part / waypoints_distances[index_absolute])
    # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

    position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                        waypoints[index_absolute][:, (1, 2)])
    position_diff_s = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 0] -
                        waypoints[index_absolute][:, 0])
    orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                            waypoints[index_absolute][:, 3])
    speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                    waypoints[index_absolute][:, 5])

    interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
    interpolated_s = waypoints[index_absolute][:, 0] + (t * position_diff_s)
    interpolated_s[np.where(interpolated_s > waypoints[-1, 0])] -= waypoints[-1, 0]
    interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
    interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
    interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
    
    reference = np.array([
        # Sort reference trajectory so the order of reference match the order of the states
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speeds,
        interpolated_orientations,
        # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
        interpolated_s,
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds))
    ])
    return reference
