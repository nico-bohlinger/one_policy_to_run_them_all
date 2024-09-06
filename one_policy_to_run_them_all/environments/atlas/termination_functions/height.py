class HeightTermination:
    def __init__(self, env):
        self.env = env

    def should_terminate(self, obs):
        return self.env.data.geom("middle_chest").xpos[2] < 0.85