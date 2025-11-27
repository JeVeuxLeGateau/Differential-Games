"""
@ Author: Matthew Giacovelli

Implementation of the Homicidal Chauffeur Game (HCG)
Goal is to provide an implementation for testing policies to solve the HCG with different vehicles
"""
import numpy as np

import agents
from Games.Game import Game
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Homicidal_Chauffeur_3D(Game):
    """
    The HCG is a two player game.

    The Chauffeur has gone insane and would like to use his vehicle to kill his victim. Naturally he's contstrained
    by the vehicles dynamics and can only control his horizontal angle (theta) limited to a max of omega_t and its.
    vertical angle (phi) limited to a max of omega_p. We limit it to a constant speed v.

    The Victim is trying his not to die. Unfortunately he's much slower, but he is more maneurverable and can move with
    simple motion.

    The game is played across a finite time horizon specified by the user in a 1000x1000 box.

    It ends when t = tf or ||p_c - p_v|| < kill distance where ||*|| is the l_2 norm
    """

    def __init__(self, chauffeur_omegas, chauffeur_speed=1, victim_speed=0.1, box_dimensions=[1000, 1000], dt=0.1):
        ##### Game specs #####
        super().__init__(box_dimensions, dt)
        self.key_names = ["chauffeur, victim"]
        ###########################################

        ##### Create players #####
        self.chauffeur = agents.Dubin(chauffeur_omegas, chauffeur_speed, self.box_dimensions, dim=3)
        self.victim = agents.Point(victim_speed, self.box_dimensions, dim=3)
        ###########################################

        ##### Initialize states for game #####
        self.chauffeur.x = self.box_dimensions[0] * 0.5
        self.chauffeur.y = self.box_dimensions[1]
        self.chauffeur.z = self.box_dimensions[1] * 0.8

        self.victim.x = self.box_dimensions[0] * 0.8
        self.victim.y = self.box_dimensions[1] * 0.2
        self.victim.z = self.box_dimensions[1] * 0.2

        x = np.array(self.victim.x - self.chauffeur.x)
        y = np.array(self.victim.y - self.chauffeur.y)
        z = np.array(self.victim.z - self.chauffeur.z)

        self.chauffeur.theta = np.atan2(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)
        self.chauffeur.phi = np.atan2(z, dist)

        ###########################################

        ##### Store agents #####
        self.all_agents = {"chauffeur": self.chauffeur,
                           "victim": self.victim}
        ###########################################

        self.chauffeur_history = {
            'x': [],
            'y': [],
            'z': []
        }

        self.victim_history = {
            'x': [],
            'y': [],
            'z': []
        }

    def step(self, actions):
        """
        :param actions: Actions is a dictionary. Keys are the names of players, Values are the actions
        """

        chauffeur_dict = self.chauffeur.get_state_dict()
        victim_dict = self.victim.get_state_dict()

        self.chauffeur_history['x'].append(chauffeur_dict['x'])
        self.chauffeur_history['y'].append(chauffeur_dict['y'])
        self.chauffeur_history['z'].append(chauffeur_dict['z'])

        self.victim_history['x'].append(victim_dict['x'])
        self.victim_history['y'].append(victim_dict['y'])
        self.victim_history['z'].append(victim_dict['z'])

        for agent in self.all_agents:
            cur_agent = self.all_agents[agent]

            state = cur_agent.get_state()
            control = actions[agent]

            cur_agent.step(state, control, self.dt)

    def check_win(self):
        c_state = self.chauffeur.get_state_dict()
        v_state = self.victim.get_state_dict()

        c_pos = np.array([c_state["x"], c_state["y"], c_state["z"]])
        v_pos = np.array([v_state["x"], v_state["y"], v_state["z"]])

        dist = np.linalg.norm(c_pos - v_pos)
        print(dist)

        if dist <= 10:
            return -1
        else:
            return 0

    def render(self):

        # 3D trajectory plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.chauffeur_history['x'], self.chauffeur_history['y'], self.chauffeur_history['z'],
                'b-', label='Chauffeur', linewidth=2)
        ax.plot(self.victim_history['x'], self.victim_history['y'], self.victim_history['z'],
                'r-', label='Victim', linewidth=2)

        # Mark start and end points
        ax.scatter(self.chauffeur_history['x'][0], self.chauffeur_history['y'][0], self.chauffeur_history['z'][0],
                   c='blue', marker='o', s=100, label='Chauffeur Start')
        ax.scatter(self.victim_history['x'][0], self.victim_history['y'][0], self.victim_history['z'][0],
                   c='red', marker='o', s=100, label='Victim Start')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
