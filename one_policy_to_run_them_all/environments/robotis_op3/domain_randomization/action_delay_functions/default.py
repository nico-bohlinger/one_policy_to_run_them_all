import numpy as np


class DefaultActionDelay:
    def __init__(self, env, max_nr_delay_steps=1, mixed_chance=0.01):
        self.env = env
        self.max_nr_delay_steps = max_nr_delay_steps
        self.mixed_chance = mixed_chance

        self.current_mixed = False
        self.current_nr_delay_steps = 0

    def setup(self):
        self.action_history = np.zeros((self.max_nr_delay_steps + 1, self.env.model.nu))

    def sample(self):
        self.current_mixed = self.env.np_rng.uniform() < self.mixed_chance
        self.current_nr_delay_steps = 0

    def delay_action(self, action):
        if self.current_mixed:
            self.current_nr_delay_steps = self.env.np_rng.integers(self.max_nr_delay_steps+1)

        self.action_history = np.roll(self.action_history, -1, axis=0)
        self.action_history[-1] = action

        chosen_action = self.action_history[-1-self.current_nr_delay_steps]

        return chosen_action
