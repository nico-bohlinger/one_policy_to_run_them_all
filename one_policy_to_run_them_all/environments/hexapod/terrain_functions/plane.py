import os
import numpy as np


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.xml_file_name = [file for file in os.listdir(os.path.join(os.path.dirname(__file__), "../data")) if file.endswith(".xml")][0]
        self.center_height = 0.0
        self.nr_sampled_heights = 1
        self.current_difficulty_level = 0.0
        self.sampled_heights = np.zeros(self.nr_sampled_heights)
    
    def step(self, obs, reward, absorbing, info):
        return
    
    def sample(self):
        return
