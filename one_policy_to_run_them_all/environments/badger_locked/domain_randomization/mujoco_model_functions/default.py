class DefaultDomainMuJoCoModel:
    def __init__(self, env,
                 friction_tangential_min=0.001, friction_tangential_max=2.0,
                 friction_torsional_ground_min=0.00001, friction_torsional_ground_max=0.01,
                 friction_torsional_feet_min=0.00001, friction_torsional_feet_max=0.01,
                 friction_rolling_ground_min=0.00001, friction_rolling_ground_max=0.0002,
                 friction_rolling_feet_min=0.00001, friction_rolling_feet_max=0.0002,
                 damping_min=30, damping_max=130,
                 stiffness_min=500, stiffness_max=1500,
                 gravity_min=8.81, gravity_max=10.81,
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
        self.sampled_friction_torsional_feet = self.env.model.geom_friction[38, 1]
        self.sampled_friction_rolling_ground = self.env.model.geom_friction[0, 2]
        self.sampled_friction_rolling_feet = self.env.model.geom_friction[38, 2]
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
        self.env.model.geom_friction[[14, 22, 30, 38]] = [self.sampled_friction_tangential, self.sampled_friction_torsional_feet, self.sampled_friction_rolling_feet]

        interpolation = self.env.np_rng.uniform(0, 1)
        self.sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        self.sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        self.env.model.geom_solref[:, 0] = -self.sampled_stiffness
        self.env.model.geom_solref[:, 1] = -self.sampled_damping

        self.sampled_gravity = self.env.np_rng.uniform(self.gravity_min, self.gravity_max)
        self.env.model.opt.gravity[2] = -self.sampled_gravity
