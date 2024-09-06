class DefaultDomainMuJoCoModel:
    def __init__(self, env,
                 friction_tangential_min=0.8, friction_tangential_max=1.2,
                 friction_torsional_ground_min=0.003, friction_torsional_ground_max=0.007,
                 friction_torsional_feet_min=0.003, friction_torsional_feet_max=0.007,
                 friction_rolling_ground_min=0.00008, friction_rolling_ground_max=0.00012,
                 friction_rolling_feet_min=0.00008, friction_rolling_feet_max=0.00012,
                 damping_min=72, damping_max=88,
                 stiffness_min=900, stiffness_max=1100,
                 gravity_min=9.51, gravity_max=10.11,
        ):
        self.env = env
        self.friction_tangential_min = friction_tangential_min
        self.friction_tangential_max = friction_tangential_max
        self.friction_torsional_ground_min = friction_torsional_ground_min
        self.friction_torsional_ground_max = friction_torsional_ground_max
        self.friction_torsional_feet_min = friction_torsional_feet_min
        self.friction_torsional_feet_max = friction_torsional_feet_max
        self.friction_rolling_ground_min = friction_rolling_ground_min
        self.friction_rolling_ground_max = friction_rolling_ground_max
        self.friction_rolling_feet_min = friction_rolling_feet_min
        self.friction_rolling_feet_max = friction_rolling_feet_max
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.stiffness_min = stiffness_min
        self.stiffness_max = stiffness_max
        self.gravity_min = gravity_min
        self.gravity_max = gravity_max
    
    def init(self):
        self.sampled_friction_tangential = self.env.model.geom_friction[0, 0]
        self.sampled_friction_torsional_ground = self.env.model.geom_friction[0, 1]
        self.sampled_friction_torsional_feet = self.env.model.geom_friction[21, 1]
        self.sampled_friction_rolling_ground = self.env.model.geom_friction[0, 2]
        self.sampled_friction_rolling_feet = self.env.model.geom_friction[21, 2]
        self.sampled_damping = -self.env.model.geom_solref[0, 1]
        self.sampled_stiffness = -self.env.model.geom_solref[0, 0]
        self.sampled_gravity = -self.env.model.opt.gravity[2]

    def sample(self):
        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_friction_tangential = self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) * interpolation
        self.sampled_friction_torsional_ground = self.friction_torsional_ground_min + (self.friction_torsional_ground_max - self.friction_torsional_ground_min) * interpolation
        self.sampled_friction_torsional_feet = self.friction_torsional_feet_min + (self.friction_torsional_feet_max - self.friction_torsional_feet_min) * interpolation
        self.sampled_friction_rolling_ground = self.friction_rolling_ground_min + (self.friction_rolling_ground_max - self.friction_rolling_ground_min) * interpolation
        self.sampled_friction_rolling_feet = self.friction_rolling_feet_min + (self.friction_rolling_feet_max - self.friction_rolling_feet_min) * interpolation
        self.env.model.geom_friction[0] = [self.sampled_friction_tangential, self.sampled_friction_torsional_ground, self.sampled_friction_rolling_ground]
        self.env.model.geom_friction[[9, 13, 17, 21]] = [self.sampled_friction_tangential, self.sampled_friction_torsional_feet, self.sampled_friction_rolling_feet]

        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        self.sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        self.env.model.geom_solref[:, 0] = -self.sampled_stiffness
        self.env.model.geom_solref[:, 1] = -self.sampled_damping

        self.sampled_gravity = self.env.np_rng.uniform(self.gravity_min, self.gravity_max)
        self.env.model.opt.gravity[2] = -self.sampled_gravity
