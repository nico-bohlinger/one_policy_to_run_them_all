class DefaultObservationNoise:
    def __init__(self, env,
                 joint_position_noise=0.003, joint_velocity_noise=0.08,
                 trunk_angular_velocity_noise=0.02,
                 gravity_vector_noise=0.015):
        self.env = env
        self.joint_position_noise = joint_position_noise
        self.joint_velocity_noise = joint_velocity_noise
        self.trunk_angular_velocity_noise = trunk_angular_velocity_noise
        self.gravity_vector_noise = gravity_vector_noise

    def init(self):
        joint_position_names = [joint_name + "_position" for joint_name in self.env.joint_names]
        joint_velocity_names = [joint_name + "_velocity" for joint_name in self.env.joint_names]
        trunk_angular_velocity_names = ["trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity"]
        gravity_vector_names = ["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"]

        self.joint_position_ids = [self.env.observation_name_to_id[name] for name in joint_position_names]
        self.joint_velocity_ids = [self.env.observation_name_to_id[name] for name in joint_velocity_names]
        self.trunk_angular_velocity_ids = [self.env.observation_name_to_id[name] for name in trunk_angular_velocity_names]
        self.gravity_vector_ids = [self.env.observation_name_to_id[name] for name in gravity_vector_names]

    def modify_observation(self, obs):
        obs[self.joint_position_ids] += self.env.np_rng.uniform(-self.joint_position_noise, self.joint_position_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.trunk_angular_velocity_ids] += self.env.np_rng.uniform(-self.trunk_angular_velocity_noise, self.trunk_angular_velocity_noise, 3)
        obs[self.joint_velocity_ids] += self.env.np_rng.uniform(-self.joint_velocity_noise, self.joint_velocity_noise, self.env.nominal_joint_positions.shape[0])
        obs[self.gravity_vector_ids] += self.env.np_rng.uniform(-self.gravity_vector_noise, self.gravity_vector_noise, 3)

        return obs
