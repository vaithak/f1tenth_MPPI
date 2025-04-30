#!/usr/bin/env python3

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_shield_mppi.mppi_utils import Environment, State
from f1tenth_shield_mppi.jax_utils import numpify
from f1tenth_shield_mppi.mppi_config import mppi_config
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from functools import partial
from f1tenth_shield_mppi.mppi_utils import MPPI


def _numpy_to_multiarray(multiarray_type, np_array):
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(label='dim%d' % i,
                                                 size=np_array.shape[i],
                                                 stride=np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)];
    multiarray.data = np_array.reshape([1, -1])[0].tolist()
    return multiarray

def _multiarray_to_numpy(pytype, dtype, multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)

to_multiarray_f32 = partial(_numpy_to_multiarray, Float32MultiArray)
to_numpy_f32 = partial(_multiarray_to_numpy, float, np.float32)

class MPPI_node(Node):
    def __init__(self):
        super().__init__('mppi_node')
        print("MPPI Node Initialized")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('is_sim', True),
                ('plot_debug', False),
                ('print_debug', False),
                ('num_samples', 100),
                ('num_steps', 10),
                ('dt', 0.1),
                ('max_steering_angle', 0.5),
                ('max_speed', 2.0),
                ('goal_tolerance', 0.1),
                ('waypoint_file', '/home/vaithak/Downloads/UPenn/F1Tenth/sim_ws/src/f1tenth_Shield_MPPI/waypoints/levine-practise-lane-optimal.csv'),
            ]
        )
        qos = rclpy.qos.QoSProfile(history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                                   depth=1,
                                   reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                                   durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)

        # Parameters
        self.is_sim = self.get_parameter('is_sim').get_parameter_value().bool_value
        self.plot_debug = self.get_parameter('plot_debug').get_parameter_value().bool_value
        self.print_debug = self.get_parameter('print_debug').get_parameter_value().bool_value
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        self.num_steps = self.get_parameter('num_steps').get_parameter_value().integer_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.max_steering_angle = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.waypoint_file = self.get_parameter('waypoint_file').get_parameter_value().string_value

        # Subscribers
        if self.is_sim:
            self.pose_sub = self.create_subscription(
                Odometry,
                '/ego_racecar/odom',
                self.pose_callback,
                qos)
        else:
            self.pose_sub = self.create_subscription(
                Odometry,
                '/pf/pose/odom',
                self.pose_callback,
                qos)

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)
        self.reference_pub = self.create_publisher(Float32MultiArray, "/reference_arr", qos)
        self.opt_traj_pub = self.create_publisher(Float32MultiArray, "/opt_traj_arr", qos)

        # Environment
        self.env = Environment(waypoint_file=self.waypoint_file)

        # MPPI Init
        self.mppi = MPPI()
        self.mppi.init_state(self.env, 2)
        self.control = np.zeros(2)


    def pose_callback(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist

        # Extract slip angle and yaw
        slip_angle = np.arctan2(twist.linear.y, twist.linear.x)
        yaw = tf_transformations.euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]


        if twist.linear.x < 1.5:
            self.control = np.array([0.0, 1.5])
            self.publish_control(self.control, msg.header)
            return

        # Calculate current state
        curr_state = State(
            pose.position.x,
            pose.position.y,
            self.control[0], # steering angle
            twist.linear.x,
            yaw,
            twist.angular.z,
            slip_angle
        )
        curr_state_jax = np.array([
            pose.position.x,
            pose.position.y,
            self.control[0],  # steering angle
            twist.linear.x,
            yaw,
            twist.angular.z,
            slip_angle
        ])

        # Calculate reference trajectory
        # ref_trajectory = self.env.get_refernece_traj(curr_state_jax)
        find_waypoint_vel = max(mppi_config.REF_VEL, curr_state.v)
        ref_trajectory, _ = self.env.get_refernece_traj(curr_state_jax.copy(), find_waypoint_vel, mppi_config.TK)
        print("Ref Trajectory: ", ref_trajectory)

        # Compute control
        self.compute_control(curr_state_jax, ref_trajectory)
        print("Control: ", self.control)

        # Publish reference trajectory
        ref_traj_cpu = numpify(ref_trajectory)
        arr_msg = to_multiarray_f32(ref_traj_cpu.astype(np.float32))
        self.reference_pub.publish(arr_msg)

        # Publish optimal trajectory
        opt_traj_cpu = numpify(self.mppi.traj_opt)
        arr_msg = to_multiarray_f32(opt_traj_cpu.astype(np.float32))
        self.opt_traj_pub.publish(arr_msg)

        if self.control is None or np.isnan(self.control).any() or np.isinf(self.control).any():
            self.get_logger().warn("Control is None or NaN")
            self.control = np.array([0.0, 0.0])
            self.mppi.a_opt = np.zeros_like(self.mppi.a_opt)

        # Publish control
        self.publish_control(self.control, msg.header)


    def compute_control(self, curr_state, ref_trajectory):
        # MPPI algorithm implementation goes here
        self.mppi.update(jnp.asarray(curr_state), jnp.asarray(ref_trajectory))
        mppi_control = numpify(self.mppi.a_opt[0])
        print("MPPI Control: ", mppi_control)
        self.control[0] = mppi_control[0] * mppi_config.DTK + self.control[0]
        self.control[1] = mppi_control[1] * mppi_config.DTK + curr_state[3]


    def publish_control(self, control, header):
        drive_msg = AckermannDriveStamped()
        drive_msg.header = header
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = control[0]
        drive_msg.drive.speed = control[1]
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI_node()
    rclpy.spin(mppi_node)
    mppi_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()