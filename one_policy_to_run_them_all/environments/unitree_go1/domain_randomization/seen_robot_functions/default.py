class DefaultDomainSeenRobotFunction:
    def __init__(
        self, env,
        stay_at_default_for_asymmetric_percentage=0.3,
        add_trunk_mass_min=-0.8, add_trunk_mass_max=0.8,
        add_com_displacement_min=-0.0025, add_com_displacement_max=0.0025,
        foot_size_min=0.0225, foot_size_max=0.0235,
        torque_limit_factor=0.5,
        add_joint_nominal_position_min=-0.01, add_joint_nominal_position_max=0.01,
        joint_velocity_factor=0.15,
        add_joint_range_min=-0.05, add_joint_range_max=0.05,
        joint_damping_min=0.0, joint_damping_max=0.2,
        joint_armature_min=0.009, joint_armature_max=0.02,
        joint_stiffness_min=0.0, joint_stiffness_max=0.5,
        joint_friction_loss_min=0.0, joint_friction_loss_max=0.65,
        add_p_gain_min=-3.0, add_p_gain_max=3.0,
        add_d_gain_min=-0.1, add_d_gain_max=0.1,  
        add_scaling_factor_min=-0.03, add_scaling_factor_max=0.03,     
    ):
        self.env = env
        self.stay_at_default_for_asymmetric_percentage = stay_at_default_for_asymmetric_percentage

        self.add_trunk_mass_min = add_trunk_mass_min
        self.add_trunk_mass_max = add_trunk_mass_max
        self.add_com_displacement_min = add_com_displacement_min
        self.add_com_displacement_max = add_com_displacement_max
        self.foot_size_min = foot_size_min
        self.foot_size_max = foot_size_max
        self.torque_limit_factor = torque_limit_factor
        self.add_joint_nominal_position_min = add_joint_nominal_position_min
        self.add_joint_nominal_position_max = add_joint_nominal_position_max
        self.joint_velocity_factor = joint_velocity_factor
        self.add_joint_range_min = add_joint_range_min
        self.add_joint_range_max = add_joint_range_max
        self.joint_damping_min = joint_damping_min
        self.joint_damping_max = joint_damping_max
        self.joint_armature_min = joint_armature_min
        self.joint_armature_max = joint_armature_max
        self.joint_stiffness_min = joint_stiffness_min
        self.joint_stiffness_max = joint_stiffness_max
        self.joint_friction_loss_min = joint_friction_loss_min
        self.joint_friction_loss_max = joint_friction_loss_max

        self.add_p_gain_min = add_p_gain_min
        self.add_p_gain_max = add_p_gain_max
        self.add_d_gain_min = add_d_gain_min
        self.add_d_gain_max = add_d_gain_max
        self.add_scaling_factor_min = add_scaling_factor_min
        self.add_scaling_factor_max = add_scaling_factor_max

        self.trunk_mass_noise_factor = 1.0  # Set by unseen robot function
        self.com_displacement_noise_factor = 1.0  # Set by unseen robot function
        self.foot_size_noise_factor = 1.0  # Set by unseen robot function
        self.joint_damping_noise_factor = 1.0  # Set by unseen robot function
        self.joint_armature_noise_factor = 1.0  # Set by unseen robot function
        self.joint_stiffness_noise_factor = 1.0  # Set by unseen robot function
        self.joint_friction_loss_noise_factor = 1.0  # Set by unseen robot function

    def init(self):
        self.default_trunk_mass = self.env.model.body_mass[1]
        self.default_trunk_inertia = self.env.model.body_inertia[1].copy()
        self.default_trunk_com = self.env.model.body_ipos[1].copy()
        self.default_torque_limit = self.env.model.actuator_ctrlrange[:, 1].copy()
        self.default_joint_nominal_position = self.env.nominal_joint_positions.copy()
        self.default_joint_velocity = self.env.max_joint_velocities.copy()
        self.default_joint_range = self.env.model.jnt_range[1:].copy()
        self.default_joint_damping = self.env.model.dof_damping[6:].copy()
        self.default_joint_armature = self.env.model.dof_armature[6:].copy()
        self.default_joint_stiffness = self.env.model.jnt_stiffness[1:].copy()
        self.default_joint_frictionloss = self.env.model.dof_frictionloss[6:].copy()

        # For environment observation
        self.seen_body_mass = self.env.model.body_mass.copy()
        self.seen_torque_limit = self.default_torque_limit
        self.seen_joint_nominal_position = self.default_joint_nominal_position
        self.seen_joint_velocity = self.default_joint_velocity
        self.seen_joint_range = self.default_joint_range
        self.seen_joint_damping = self.default_joint_damping
        self.seen_joint_armature = self.default_joint_armature
        self.seen_joint_stiffness = self.default_joint_stiffness
        self.seen_joint_frictionloss = self.default_joint_frictionloss


    def sample(self):
        stay_at_default_for_asymmatric = self.env.np_rng.uniform(0, 1) < self.stay_at_default_for_asymmetric_percentage

        seen_trunk_mass = self.default_trunk_mass + self.env.np_rng.uniform(self.add_trunk_mass_min, self.add_trunk_mass_max)
        self.seen_body_mass[1] = seen_trunk_mass
        self.env.model.body_mass[1] = seen_trunk_mass * self.trunk_mass_noise_factor
        self.env.model.body_inertia[1] = self.default_trunk_inertia * (self.env.model.body_mass[1] / self.default_trunk_mass)

        self.env.model.body_ipos[1] = (self.default_trunk_com + self.env.np_rng.uniform(self.add_com_displacement_min, self.add_com_displacement_max)) * self.com_displacement_noise_factor

        self.env.model.geom_size[[19, 31, 43, 55],0] = self.env.np_rng.uniform(self.foot_size_min, self.foot_size_max) * self.foot_size_noise_factor

        self.seen_joint_nominal_position = self.default_joint_nominal_position + self.env.np_rng.uniform(self.add_joint_nominal_position_min, self.add_joint_nominal_position_max, size=self.default_joint_nominal_position.shape)
        self.env.nominal_joint_positions = self.seen_joint_nominal_position

        self.seen_torque_limit = self.default_torque_limit * (1 + self.env.np_rng.uniform(-self.torque_limit_factor, self.torque_limit_factor, size=self.default_torque_limit.shape))
        self.env.model.actuator_ctrlrange[:, 1] = self.seen_torque_limit
        self.env.model.actuator_ctrlrange[:, 0] = -self.seen_torque_limit

        self.seen_joint_velocity = self.default_joint_velocity * (1 + self.env.np_rng.uniform(-self.joint_velocity_factor, self.joint_velocity_factor, size=self.default_joint_velocity.shape))
        self.env.max_joint_velocities = self.seen_joint_velocity

        self.seen_joint_range = self.default_joint_range + self.env.np_rng.uniform(self.add_joint_range_min, self.add_joint_range_max, size=self.default_joint_range.shape)
        self.env.model.jnt_range[1:] = self.seen_joint_range

        if stay_at_default_for_asymmatric:
            self.seen_joint_damping = self.default_joint_damping
            self.seen_armature = self.default_joint_armature
            self.seen_joint_stiffness = self.default_joint_stiffness
            self.seen_joint_frictionloss = self.default_joint_frictionloss
        else:
            self.seen_joint_damping = self.env.np_rng.uniform(self.joint_damping_min, self.joint_damping_max, size=self.default_joint_damping.shape)
            self.seen_joint_armature = self.env.np_rng.uniform(self.joint_armature_min, self.joint_armature_max, size=self.default_joint_armature.shape)
            self.seen_joint_stiffness = self.env.np_rng.uniform(self.joint_stiffness_min, self.joint_stiffness_max, size=self.default_joint_stiffness.shape)
            self.seen_joint_frictionloss = self.env.np_rng.uniform(self.joint_friction_loss_min, self.joint_friction_loss_max, size=self.default_joint_frictionloss.shape)
        self.env.model.dof_damping[6:] = self.seen_joint_damping * self.joint_damping_noise_factor
        self.env.model.dof_armature[6:] = self.seen_joint_armature * self.joint_armature_noise_factor
        self.env.model.jnt_stiffness[1:] = self.seen_joint_stiffness * self.joint_stiffness_noise_factor
        self.env.model.dof_frictionloss[6:] = self.seen_joint_frictionloss * self.joint_friction_loss_noise_factor

        self.env.control_function.add_p_gain = self.env.np_rng.uniform(self.add_p_gain_min, self.add_p_gain_max)
        self.env.control_function.add_d_gain = self.env.np_rng.uniform(self.add_d_gain_min, self.add_d_gain_max)
        self.env.control_function.add_scaling_factor = self.env.np_rng.uniform(self.add_scaling_factor_min, self.add_scaling_factor_max)
