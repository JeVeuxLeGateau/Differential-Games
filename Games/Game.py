import numpy as np


class Game:
    def __init__(self, box_dimensions=[1000, 1000], dt=0.1):
        self.box_dimensions = box_dimensions
        self.dt = dt

    def enforce_bounds(self):
        for agent in self.all_agents:
            self.all_agents[agent].x = np.clip(self.all_agents[agent].x, 0, self.box_dimensions[0])
            self.all_agents[agent].y = np.clip(self.all_agents[agent].y, 0, self.box_dimensions[1])