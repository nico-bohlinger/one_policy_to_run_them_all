class HeightTermination:
    def __init__(self, env):
        self.env = env

    def should_terminate(self, obs):
        return self.env.data.qpos[2] < 0.4