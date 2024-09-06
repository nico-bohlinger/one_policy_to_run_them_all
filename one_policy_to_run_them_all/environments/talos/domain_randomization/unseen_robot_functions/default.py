class DefaultDomainUnseenRobotFunction:
    def __init__(self, env,
                 trunk_mass_factor=0.1,
                 com_displacement_factor=0.1,
                 foot_size_factor=0.03,
                 joint_damping_factor=0.1,
                 joint_armature_factor=0.1,
                 joint_stiffness_factor=0.1,
                 joint_friction_loss_factor=0.1,
                 motor_strength_factor=0.1,
                 p_gain_factor=0.1,
                 d_gain_factor=0.1,
                 position_offset=0.01
        ):
        self.env = env
        self.trunk_mass_factor = trunk_mass_factor
        self.com_displacement_factor = com_displacement_factor
        self.foot_size_factor = foot_size_factor
        self.joint_damping_factor = joint_damping_factor
        self.joint_armature_factor = joint_armature_factor
        self.joint_stiffness_factor = joint_stiffness_factor
        self.joint_friction_loss_factor = joint_friction_loss_factor
        self.motor_strength_factor = motor_strength_factor
        self.p_gain_factor = p_gain_factor
        self.d_gain_factor = d_gain_factor
        self.position_offset = position_offset

    def sample(self):
        self.env.domain_randomization_seen_robot_function.trunk_mass_noise_factor = self.env.np_rng.uniform(1 - self.trunk_mass_factor, 1 + self.trunk_mass_factor)
        self.env.domain_randomization_seen_robot_function.com_displacement_noise_factor = self.env.np_rng.uniform(1 - self.com_displacement_factor, 1 + self.com_displacement_factor)
        self.env.domain_randomization_seen_robot_function.foot_size_noise_factor = self.env.np_rng.uniform(1 - self.foot_size_factor, 1 + self.foot_size_factor)
        self.env.domain_randomization_seen_robot_function.joint_damping_noise_factor = self.env.np_rng.uniform(1 - self.joint_damping_factor, 1 + self.joint_damping_factor)
        self.env.domain_randomization_seen_robot_function.joint_armature_noise_factor = self.env.np_rng.uniform(1 - self.joint_armature_factor, 1 + self.joint_armature_factor)
        self.env.domain_randomization_seen_robot_function.joint_stiffness_noise_factor = self.env.np_rng.uniform(1 - self.joint_stiffness_factor, 1 + self.joint_stiffness_factor)
        self.env.domain_randomization_seen_robot_function.joint_friction_loss_noise_factor = self.env.np_rng.uniform(1 - self.joint_friction_loss_factor, 1 + self.joint_friction_loss_factor)

        self.env.control_function.p_gain_noise_factor = self.env.np_rng.uniform(1 - self.p_gain_factor, 1 + self.p_gain_factor)
        self.env.control_function.d_gain_noise_factor = self.env.np_rng.uniform(1 - self.d_gain_factor, 1 + self.d_gain_factor)
        self.env.control_function.motor_strength_noise_factor = self.env.np_rng.uniform(1 - self.motor_strength_factor, 1 + self.motor_strength_factor)
        self.env.control_function.joint_position_offset = self.env.np_rng.uniform(-self.position_offset, self.position_offset, self.env.model.nu)
