import numpy as np


class RandomCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-1.0, max_yaw_velocity=1.0):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity

    def get_next_command(self):
        goal_x_velocity = self.env.np_rng.uniform(self.min_x_velocity, self.max_x_velocity)
        goal_y_velocity = self.env.np_rng.uniform(self.min_y_velocity, self.max_y_velocity)
        goal_yaw_velocity = self.env.np_rng.uniform(self.min_yaw_velocity, self.max_yaw_velocity)

        return goal_x_velocity, goal_y_velocity, goal_yaw_velocity

    def setup(self):
        return

    def step(self, obs, reward, absorbing, info):
        return
