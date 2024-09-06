class NoneActionDelay:
    def __init__(self, env):
        self.env = env

    def setup(self):
        pass

    def sample(self):
        pass

    def delay_action(self, action):
        return action
