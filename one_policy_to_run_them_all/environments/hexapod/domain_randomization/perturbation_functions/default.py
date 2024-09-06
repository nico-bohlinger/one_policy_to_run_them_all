import numpy as np


class DefaultDomainPerturbation:
    def __init__(self, env,
                 push_velocity_x_min=-0.3, push_velocity_x_max=0.3,
                 push_velocity_y_min=-0.3, push_velocity_y_max=0.3,
                 push_velocity_z_min=-0.3, push_velocity_z_max=0.3,
                 set_chance=0.5, additive_multiplier=1.0):
        self.env = env
        self.push_velocity_x_min = push_velocity_x_min
        self.push_velocity_x_max = push_velocity_x_max
        self.push_velocity_y_min = push_velocity_y_min
        self.push_velocity_y_max = push_velocity_y_max
        self.push_velocity_z_min = push_velocity_z_min
        self.push_velocity_z_max = push_velocity_z_max
        self.set_chance = set_chance
        self.additive_multiplier = additive_multiplier
        self.sampled_push_velocity_x = 0.0
        self.sampled_push_velocity_y = 0.0
        self.sampled_push_velocity_z = 0.0

    def sample(self):
        self.sampled_push_velocity_x = self.env.np_rng.uniform(self.push_velocity_x_min, self.push_velocity_x_max)
        self.sampled_push_velocity_y = self.env.np_rng.uniform(self.push_velocity_y_min, self.push_velocity_y_max)
        self.sampled_push_velocity_z = self.env.np_rng.uniform(self.push_velocity_z_min, self.push_velocity_z_max)
        if self.env.np_rng.uniform() < self.set_chance:
            self.env.data.qvel[0:3] = np.array([self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z])
        else:
            self.env.data.qvel[0:3] += np.array([self.sampled_push_velocity_x, self.sampled_push_velocity_y, self.sampled_push_velocity_z]) * self.additive_multiplier
