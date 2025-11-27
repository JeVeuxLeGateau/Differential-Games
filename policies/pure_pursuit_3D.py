import numpy as np


def pure_pursuit_3D(pursuer_dict, evader_dict):
    """
    point our pursuer towards our evaders current position
    :param pursuer_dict:
    :param evader_dict:
    :return:
    """

    x_p = pursuer_dict["x"]
    y_p = pursuer_dict["y"]
    z_p = pursuer_dict["z"]
    x_e = evader_dict["x"]
    y_e = evader_dict["y"]
    z_e = evader_dict["z"]

    dx = x_e - x_p
    dy = y_e - y_p
    dz = z_e - z_p

    theta = np.atan2(dy, dx)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    phi = np.atan2(dz, dist)
    return theta, phi
