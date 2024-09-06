class NoneObservationDropout:
    def __init__(self, env):
        self.env = env

    def init(self):
        pass

    def modify_observation(self, obs):
        return obs
