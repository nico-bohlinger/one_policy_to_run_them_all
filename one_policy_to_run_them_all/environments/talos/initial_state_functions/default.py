class DefaultInitialState:
    def __init__(self, env):
        self.env = env

    def step(self, obs, reward, absorbing, info):
        return

    def setup(self):
        qpos = [
            0.0, 0.0, self.env.initial_drop_height + self.env.terrain_function.center_height,
            1.0, 0.0, 0.0, 0.0,
            *self.env.nominal_joint_positions
        ]

        qvel = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            *([0.0] * self.env.model.nu)
        ]
        
        return qpos, qvel
