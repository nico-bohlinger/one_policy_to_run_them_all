class DefaultDomainMuJoCoModel:
    def __init__(self, env,
                 friction_tangential_min=0.8, friction_tangential_max=1.2,
                 friction_torsional_min=0.003, friction_torsional_max=0.007,
                 friction_rolling_min=0.00008, friction_rolling_max=0.00012,
                 damping_min=72, damping_max=88,
                 stiffness_min=900, stiffness_max=1100,
                 gravity_min=9.51, gravity_max=10.11,
        ):
        self.env = env
        self.friction_tangential_min = friction_tangential_min
        self.friction_tangential_max = friction_tangential_max
        self.friction_torsional_min = friction_torsional_min
        self.friction_torsional_max = friction_torsional_max
        self.friction_rolling_min = friction_rolling_min
        self.friction_rolling_max = friction_rolling_max
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.stiffness_min = stiffness_min
        self.stiffness_max = stiffness_max
        self.gravity_min = gravity_min
        self.gravity_max = gravity_max
    
    def init(self):
        self.sampled_friction_tangential = self.env.model.geom_friction[0, 0]
        self.sampled_friction_torsional = self.env.model.geom_friction[0, 1]
        self.sampled_friction_rolling = self.env.model.geom_friction[0, 2]
        self.sampled_damping = -self.env.model.geom_solref[0, 1]
        self.sampled_stiffness = -self.env.model.geom_solref[0, 0]
        self.sampled_gravity = -self.env.model.opt.gravity[2]

    def sample(self):
        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_friction_tangential = self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) * interpolation
        self.sampled_friction_torsional = self.friction_torsional_min + (self.friction_torsional_max - self.friction_torsional_min) * interpolation
        self.sampled_friction_rolling = self.friction_rolling_min + (self.friction_rolling_max - self.friction_rolling_min) * interpolation
        self.env.model.geom_friction[:,] = [self.sampled_friction_tangential, self.sampled_friction_torsional, self.sampled_friction_rolling]

        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        self.sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        self.env.model.geom_solref[:, 0] = -self.sampled_stiffness
        self.env.model.geom_solref[:, 1] = -self.sampled_damping

        self.sampled_gravity = self.env.np_rng.uniform(self.gravity_min, self.gravity_max)
        self.env.model.opt.gravity[2] = -self.sampled_gravity
