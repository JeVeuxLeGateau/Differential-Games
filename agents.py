import numpy as np


class real_world:
    def __init__(self, size):
        self.size = size


class Point(real_world):
    """
    Simple holonomic point:
    State: [x, y, theta]
    Control: u = heading angle (instantaneous turn)

    x_dot = v*cos(u)
    y_dot = v*sin(u)
    """

    def __init__(self, v, box_dims, size=5):
        super().__init__(size)
        self.x = np.random.uniform(box_dims[0], box_dims[1])
        self.y = np.random.uniform(box_dims[0], box_dims[1])
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.v = v

    def f(self, u):
        x_dot = self.v * np.cos(u)
        y_dot = self.v * np.sin(u)

        return np.array([x_dot, y_dot])

    def step(self, state, control, dt):
        dx, dy = self.f(control)
        self.x += dx * dt
        self.y += dy * dt
        self.theta = control

    def get_state(self):
        return np.array([self.x, self.y, self.theta])

    def get_state_dict(self):
        return {"x": self.x,
                "y": self.y,
                "theta": self.theta
                }


class Dubin(real_world):
    """
    Dubin's car:
    State: [x, y, theta]
        x and y are in cartesian coordinates
        theta is the orientation relative to the x-axis

    x_dot = v*cos(theta)
    y_dot = v*sin(theta)
    theta_dot = u - Control input

    u must be between [-omega,omega] -> clamped

    """

    def __init__(self, omega, v, box_dims, size=5):
        super().__init__(size)
        self.x = np.random.uniform(box_dims[0], box_dims[1])
        self.y = np.random.uniform(box_dims[0], box_dims[1])
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.omega = omega
        self.v = v

    def f(self, state, u):
        # x and y are redundant but we need theta
        x, y, theta = state
        x_dot = self.v * np.cos(theta)
        y_dot = self.v * np.sin(theta)
        theta_dot = np.clip(u, -self.omega, self.omega)

        return np.array([x_dot, y_dot, theta_dot])

    def step(self, state, control, dt):
        """
        Assuming the control policy will be in theta not angular velocity
        """
        current_theta = self.theta
        desired_theta = control

        w = (desired_theta - current_theta)/dt

        f = self.f
        k1 = f(state, w) * dt
        k2 = f(state + 0.5 * k1, w) * dt
        k3 = f(state + 0.5 * k2, w) * dt
        k4 = f(state + k3, w) * dt

        new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        self.x = new_state[0]
        self.y = new_state[1]
        self.theta = new_state[2]

    def get_state(self):
        return np.array([self.x, self.y, self.theta])

    def get_state_dict(self):
        return {"x": self.x,
                "y": self.y,
                "theta": self.theta
                }
