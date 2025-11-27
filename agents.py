import numpy as np


class Agent:
    def __init__(self, v, box_dims, dim=2, size=5):
        self.size = size
        self.dim = dim
        self.v = v
        self.x = np.random.uniform(box_dims[0], box_dims[1])
        self.y = np.random.uniform(box_dims[0], box_dims[1])
        self.theta = np.random.uniform(-np.pi, np.pi)

        if dim == 3:
            self.z = np.random.uniform(box_dims[0], box_dims[1])
            self.phi = np.random.uniform(-np.pi, np.pi)

    def get_state(self):
        if self.dim == 2:
            return np.array([self.x,
                             self.y,
                             self.theta])
        else:
            return np.array([self.x,
                             self.y,
                             self.z,
                             self.theta,
                             self.phi])

    def get_state_dict(self):
        if self.dim == 2:
            return {
                "x": self.x,
                "y": self.y,
                "theta": self.theta
            }
        else:
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "theta": self.theta,
                "phi": self.phi
            }


class Point(Agent):
    """
    Holonomic point robot in 2D or 3D.

    2D State: [x, y, theta]
    Control: u = heading angle

        x_dot = v cos(u)
        y_dot = v sin(u)

    3D State: [x, y, z, theta, phi]
    Control: u = [theta, phi]

        x_dot = v cos(theta) cos(phi)
        y_dot = v sin(theta) cos(phi)
        z_dot = v sin(phi)
    """

    def __init__(self, v, box_dims, dim=2, size=5):
        super().__init__(v, box_dims, dim, size)

    def f(self, u):
        if self.dim == 2:
            x_dot = self.v * np.cos(u)
            y_dot = self.v * np.sin(u)
            return np.array([x_dot, y_dot])
        else:
            theta, phi = u
            x_dot = self.v * np.cos(theta) * np.cos(phi)
            y_dot = self.v * np.sin(theta) * np.cos(phi)
            z_dot = self.v * np.sin(phi)

            return np.array([x_dot, y_dot, z_dot])

    def step(self, state, control, dt):
        d = self.f(control)
        self.x += d[0] * dt
        self.y += d[1] * dt

        if self.dim == 2:
            self.theta = control
        else:
            self.z += d[2] * dt
            self.theta, self.phi = control


class Dubin(Agent):
    """
    Dubin's car:
    State: [x, y, theta]
        x and y are in cartesian coordinates
        theta is the orientation relative to the x-axis

    x_dot = v*cos(theta)
    y_dot = v*sin(theta)
    theta_dot = u - Control input

    u must be between [-omega,omega] -> clamped

    Also available in 3D

    """

    def __init__(self, omegas, v, box_dims, dim=2, size=5):
        super().__init__(v, box_dims, dim, size)

        if dim == 2:
            self.omega = omegas
        else:
            self.omega_t, self.omega_p = omegas

    def f(self, state, u):
        # x and y are redundant but we need theta
        if self.dim == 2:
            x, y, theta = state
            x_dot = self.v * np.cos(theta)
            y_dot = self.v * np.sin(theta)
            theta_dot = np.clip(u, -self.omega, self.omega)
            return np.array([x_dot, y_dot, theta_dot])

        else:
            x, y, z, theta, phi = state
            theta_dot, phi_dot = u
            x_dot = self.v * np.cos(theta) * np.cos(phi)
            y_dot = self.v * np.sin(theta) * np.cos(phi)
            z_dot = self.v * np.sin(phi)

            theta_dot = np.clip(theta_dot, -self.omega_t, self.omega_t)
            phi_dot = np.clip(phi_dot, -self.omega_p, self.omega_p)

            return np.array([x_dot, y_dot, z_dot, theta_dot, phi_dot])

    def step(self, state, control, dt):
        """
        Assuming the control policy will be in theta not angular velocity
        """

        if self.dim == 2:
            current_theta = self.theta
            desired_theta = control

            w = (desired_theta - current_theta) / dt

            f = self.f
            k1 = f(state, w) * dt
            k2 = f(state + 0.5 * k1, w) * dt
            k3 = f(state + 0.5 * k2, w) * dt
            k4 = f(state + k3, w) * dt

            new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            self.x = new_state[0]
            self.y = new_state[1]
            self.theta = new_state[2]

        else:
            current_theta = self.theta
            current_phi = self.phi
            desired_theta, desired_phi = control

            w_t = (desired_theta - current_theta) / dt
            w_p = (desired_phi - current_phi) / dt

            control = np.array([w_t, w_p])
            f = self.f
            k1 = f(state, control) * dt
            k2 = f(state + 0.5 * k1, control) * dt
            k3 = f(state + 0.5 * k2, control) * dt
            k4 = f(state + k3, control) * dt

            new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            self.x = new_state[0]
            self.y = new_state[1]
            self.z = new_state[2]
            self.theta = new_state[3]
            self.phi = new_state[4]