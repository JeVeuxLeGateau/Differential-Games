"""
@ Author: Matthew Giacovelli

Implementation of the Homicidal Chauffeur Game (HCG)
Goal is to provide an implementation for testing policies to solve the HCG with different vehicles
"""
import numpy as np

import agents
import pygame


class Homicidal_Chauffeur:
    """
    The HCG is a two player game.

    The Chauffeur has gone insane and would like to use his vehicle to kill his victim. Naturally he's contstrained
    by the vehicles dynamics and can only control his steering angle (theta) limited to a max of omega. To model this we
    give it dubin dynamics meaning he moves with constant speed, v, and controls its orientation with theta.

    The Victim is trying his not to die. Unfortunately he's much slower, but he is more maneurverable and can move with
    simple motion.

    The game is played across a finite time horizon specified by the user in a 1000x1000 box.

    It ends when t = tf or ||p_c - p_v|| < kill distance where ||*|| is the l_2 norm
    """

    def __init__(self, chauffeur_omega, chauffeur_speed=1, victim_speed=0.1, box_dimensions=[1000, 1000], dt=0.1):
        ##### Game specs #####
        self.box_dimensions = box_dimensions
        self.dt = dt
        self.key_names = ["chauffeur, victim"]
        ###########################################

        ##### Create pygame box for rendering #####
        pygame.init()
        self.screen = pygame.display.set_mode([self.box_dimensions[0], self.box_dimensions[1]])
        pygame.display.set_caption("Homicidal Chauffeur")
        # Fill the screen with white background
        self.screen.fill((255, 255, 255))
        ###########################################

        ##### Create players #####
        self.chauffeur = agents.Dubin(chauffeur_omega, chauffeur_speed, self.box_dimensions)
        self.victim = agents.Point(victim_speed, self.box_dimensions)
        ###########################################

        ##### Initialize states for game #####
        self.chauffeur.x = self.box_dimensions[0] / 2
        self.chauffeur.y = self.box_dimensions[1]

        self.victim.x = self.box_dimensions[0] / 2
        self.victim.y = self.box_dimensions[1] * 0.2

        x = np.array(self.victim.x - self.chauffeur.x)
        y = np.array(self.victim.y - self.chauffeur.y)
        self.chauffeur.theta = np.atan2(y, x)
        ###########################################

        ##### Store agents #####
        self.all_agents = {"chauffeur": self.chauffeur,
                           "victim": self.victim}
        ###########################################

    def step(self, actions):
        """
        :param actions: Actions is a dictionary. Keys are the names of players, Values are the actions
        """

        for agent in self.all_agents:
            cur_agent = self.all_agents[agent]

            state = cur_agent.get_state()
            control = actions[agent]

            cur_agent.step(state, control, self.dt)

    def check_win(self):
        c_state = self.chauffeur.get_state()
        c_state = np.array([c_state[0], c_state[1]])
        v_state = self.victim.get_state()

        dist = np.linalg.norm(c_state - v_state)

        if dist <= 10:
            return -1
        else:
            return 0

    def render(self):

        # Draw each agent
        for agent in self.all_agents:
            cur_agent = self.all_agents[agent]

            state = cur_agent.get_state()

            if agent == "chauffeur":
                pygame.draw.circle(self.screen, "orange",
                                   (int(state[0]), int(state[1])), int(cur_agent.size))
            else:
                pygame.draw.circle(self.screen, "black",
                                   (int(state[0]), int(state[1])), int(cur_agent.size))


        # Update the full display surface to the screen
        pygame.display.flip()