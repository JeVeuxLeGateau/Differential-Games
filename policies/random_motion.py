"""
Agent moves in a zig-zag pattern. We choose a random direction and start moving in that direction.
Then every T seconds (defined by the user) we pick a different direction and move that way.
"""
import numpy as np


def random_motion(T, current_time, cur_theta):
    """
    Picks a new direction to move in every T timesteps
    :param T: How often we want to choose a different direction in timesteps
    :param current_time: Current number of timesteps transpired
    :param cur_theta: Current orientation
    :return: Theta (orientation)
    """

    if current_time % T == 0:
        new_vel = np.random.uniform(-np.pi, np.pi)
        return new_vel

    else:
        return cur_theta
    