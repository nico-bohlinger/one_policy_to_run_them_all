class Rudin2022Control:
    def __init__(self, env, control_frequency_hz=50, p_gain=100, d_gain=2.0, scaling_factor=0.75):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.scaling_factor = scaling_factor
        self.add_p_gain = 0.0  # Set by seen robot function
        self.add_d_gain = 0.0  # Set by seen robot function
        self.add_scaling_factor = 0.0  # Set by seen robot function
        self.p_gain_noise_factor = 1.0  # Set by unseen robot function
        self.d_gain_noise_factor = 1.0  # Set by unseen robot function
        self.motor_strength_noise_factor = 1.0  # Set by unseen robot function
        self.joint_position_offset = 0.0 # Set by unseen robot function
        self.seen_p_gain = p_gain + self.add_p_gain
        self.seen_d_gain = d_gain + self.add_d_gain
        self.seen_scaling_factor = scaling_factor + self.add_scaling_factor

    def process_action(self, action):
        self.seen_p_gain = self.p_gain + self.add_p_gain
        self.seen_d_gain = self.d_gain + self.add_d_gain
        self.seen_scaling_factor = self.scaling_factor + self.add_scaling_factor

        scaled_action = action * self.seen_scaling_factor
        target_joint_positions = self.env.nominal_joint_positions + scaled_action
        torques = self.seen_p_gain * self.p_gain_noise_factor * (target_joint_positions - self.env.data.qpos[7:] + self.joint_position_offset) \
                  - self.seen_d_gain * self.d_gain_noise_factor * self.env.data.qvel[6:]
        
        return torques * self.motor_strength_noise_factor
