import numpy as np


class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability

    def setup(self):
        return False

    def step(self):
        return self.env.np_rng.uniform() < self.probability
