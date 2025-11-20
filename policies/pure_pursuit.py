import numpy as np


def pure_pursuit(pursuer_dict, evader_dict):
    """
    point our pursuer towards our evaders current position
    :param pursuer_dict:
    :param evader_dict:
    :return:
    """

    x_p = pursuer_dict["x"]
    y_p = pursuer_dict["y"]
    x_e = evader_dict["x"]
    y_e = evader_dict["y"]

    dx = x_e - x_p
    dy = y_e - y_p

    return np.atan2(dy, dx)
