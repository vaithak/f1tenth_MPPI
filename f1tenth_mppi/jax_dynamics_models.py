import jax
import jax.numpy as jnp

params_f1tenth = {'mu': 1.0, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                               'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2,
                               'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
                               'width': 0.31, 'length': 0.58}  # F1/10 car

params = params_f1tenth


def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    pos_limit = jax.lax.select(vel > v_switch, a_max*v_switch/vel, a_max)

    # accl limit reached?
    accl = jax.lax.select((vel <= v_min) & (accl <= 0), 0., accl)
    accl = jax.lax.select((vel >= v_max) & (accl >= 0), 0., accl)
    
    accl = jax.lax.select(accl <= -a_max, -a_max, accl)
    accl = jax.lax.select(accl >= pos_limit, pos_limit, accl)

    return accl


def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    steering_velocity = jax.lax.select((steering_angle <= s_min) & (steering_velocity <= 0), 0., steering_velocity)
    steering_velocity = jax.lax.select((steering_angle >= s_max) & (steering_velocity >= 0), 0., steering_velocity)
    steering_velocity = jax.lax.select(steering_velocity <= sv_min, sv_min, steering_velocity)
    steering_velocity = jax.lax.select(steering_velocity >= sv_max, sv_max, steering_velocity)
    
    return steering_velocity

@jax.jit
def vehicle_dynamics_ks(x, u_init, C_Sf=20.898, C_Sr=20.898, 
                        lf=0.88392, lr=1.50876, h=0.59436, m=1225.887, I=1538.853371):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lf = params['lf']  # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params['lr']  # distance from spring mass center of gravity to rear axle [m]  LENB
    lwb = lf + lr
    # steering constraints
    s_min = params['s_min']  # minimum steering angle [rad]
    s_max = params['s_max']  # maximum steering angle [rad]
    # longitudinal constraints
    v_min = params['v_min']  # minimum velocity [m/s]
    v_max = params['v_max'] # minimum velocity [m/s]
    sv_min = params['sv_min'] # minimum steering velocity [rad/s]
    sv_max = params['sv_max'] # maximum steering velocity [rad/s]
    v_switch = params['v_switch']  # switching velocity [m/s]
    a_max = params['a_max'] # maximum absolute acceleration [m/s^2]
    
    
    # constraints
    u = jnp.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = jnp.array([x[3]*jnp.cos(x[4]),
         x[3]*jnp.sin(x[4]), 
         u[0],
         u[1],
         x[3]/lwb*jnp.tan(x[2])])
    return f


@jax.jit
def vehicle_dynamics_st(x, u_init, mu=1.0, C_Sf=20.898, C_Sr=20.898, 
                        lf=0.88392, lr=1.50876, h=0.59436, m=1225.887, I=1538.853371):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # gravity constant m/s^2
    g = 9.81
    
    
    # steering constraints
    lf = params['lf']  # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params['lr']  # distance from spring mass center of gravity to rear axle [m]  LENB
    h = params['h']  # M_s center of gravity above ground [m]  HS
    m = params['m']  # vehicle mass [kg]  MASS
    I = params['I']  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    C_Sf = params['C_Sf']  # front tire cornering stiffness [N/rad]  CF
    C_Sr = params['C_Sr']  # rear tire cornering stiffness [N/rad]  CR
    s_min = params['s_min']  # minimum steering angle [rad]
    s_max = params['s_max']  # maximum steering angle [rad]
    # longitudinal constraints
    v_min = params['v_min']  # minimum velocity [m/s]
    v_max = params['v_max'] # minimum velocity [m/s]
    sv_min = params['sv_min'] # minimum steering velocity [rad/s]
    sv_max = params['sv_max'] # maximum steering velocity [rad/s]
    v_switch = params['v_switch']  # switching velocity [m/s]
    a_max = params['a_max'] # maximum absolute acceleration [m/s^2]

    # constraints
    u = jnp.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])
    # print('u model', u)
    
    # u = u_init
    # steer = jax.lax.select(u_init[0] > 0.4, 0.4, u_init[0])
    # steer = jax.lax.select(u_init[0] < -0.4, -0.4, u_init[0])
    # u = jnp.array([steer, u_init[1]])
    
    # switch to kinematic model for small velocities
    # if abs(x[3]) < 0.5:
    #     # wheelbase
    lwb = lf + lr

    # u = jax.lax.select(jnp.abs(x[3]) < 1, jnp.array([0., 1.]), u)
    #     # system dynamics
    x_ks = x[0:5]
    f_ks = vehicle_dynamics_ks(x_ks, u)
    f_ks = jnp.hstack((f_ks, jnp.array([u[1]/lwb*jnp.tan(x[2])+x[3]/(lwb*jnp.cos(x[2])**2)*u[0],0])))

    # else:
    # system dynamics
    f = jnp.array([x[3]*jnp.cos(x[6] + x[4]),
        x[3]*jnp.sin(x[6] + x[4]),
        u[0],
        u[1],
        x[5],
        -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
            +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
            +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
        (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
            -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
            +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])
    # return f
    return jax.lax.select(jnp.abs(x[3]) < 1, f_ks, f)