class NoneTerminationFunction:
    def __init__(self, env):
        self.env = env

    def should_terminate(self, obs):
        return False
