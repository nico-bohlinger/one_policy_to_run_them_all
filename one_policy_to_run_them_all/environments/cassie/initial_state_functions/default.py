import numpy as np


class DefaultInitialState:
    def __init__(self, env):
        self.env = env

    def step(self, obs, reward, absorbing, info):
        return

    def setup(self):
        qpos = self.env.complete_nominal_joint_position.tolist()
        qpos[2] += self.env.terrain_function.center_height

        qvel = np.zeros(self.env.data.qvel.shape)
        
        return qpos, qvel
