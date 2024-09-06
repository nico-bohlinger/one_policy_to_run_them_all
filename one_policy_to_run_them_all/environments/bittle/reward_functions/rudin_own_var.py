import numpy as np


class RudinOwnVarReward:
    def __init__(self, env,
                 curriculum_steps=40e6,
                 tracking_xy_velocity_command_coeff=5.0, tracking_yaw_velocity_command_coeff=2.5,
                 xy_tracking_temperature=0.25, yaw_tracking_temperature=0.25,
                 z_velocity_coeff=2e0, pitch_roll_vel_coeff=5e-2, pitch_roll_pos_coeff=2e-1, joint_nominal_diff_coeff=0.0,
                 joint_position_limit_coeff=1e1, soft_joint_position_limit=0.9,
                 joint_velocity_coeff=0.0, joint_acceleration_coeff=2.5e-7, joint_torque_coeff=2e-4, action_rate_coeff=1e-2,
                 collision_coeff=1e0, base_height_coeff=3e1, nominal_trunk_z=0.094, air_time_coeff=1e-1, air_time_max=0.5, symmetry_air_coeff=0.5):
        self.env = env
        self.tracking_xy_velocity_command_coeff = tracking_xy_velocity_command_coeff * self.env.dt
        self.tracking_yaw_velocity_command_coeff = tracking_yaw_velocity_command_coeff * self.env.dt
        self.curriculum_steps = curriculum_steps
        self.xy_tracking_temperature = xy_tracking_temperature
        self.yaw_tracking_temperature = yaw_tracking_temperature
        self.z_velocity_coeff = z_velocity_coeff * self.env.dt
        self.pitch_roll_vel_coeff = pitch_roll_vel_coeff * self.env.dt
        self.pitch_roll_pos_coeff = pitch_roll_pos_coeff * self.env.dt
        self.joint_nominal_diff_coeff = joint_nominal_diff_coeff * self.env.dt
        self.joint_position_limit_coeff = joint_position_limit_coeff * self.env.dt
        self.soft_joint_position_limit = soft_joint_position_limit
        self.joint_velocity_coeff = joint_velocity_coeff * self.env.dt
        self.joint_acceleration_coeff = joint_acceleration_coeff * self.env.dt
        self.joint_torque_coeff = joint_torque_coeff * self.env.dt
        self.action_rate_coeff = action_rate_coeff * self.env.dt
        self.collision_coeff = collision_coeff * self.env.dt
        self.base_height_coeff = base_height_coeff * self.env.dt
        self.nominal_trunk_z = nominal_trunk_z
        self.air_time_coeff = air_time_coeff * self.env.dt
        self.air_time_max = air_time_max
        self.symmetry_air_coeff = symmetry_air_coeff * self.env.dt

        self.time_since_last_touchdown_lb = 0
        self.time_since_last_touchdown_lf = 0
        self.time_since_last_touchdown_rb = 0
        self.time_since_last_touchdown_rf = 0
        self._prev_joint_vel = None

    def init(self):
        self.joint_limits = self.env.model.jnt_range[1:].copy()
        joint_limits_midpoint = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        joint_limits_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.joint_limits[:, 0] = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        self.joint_limits[:, 1] = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit

    def setup(self):
        self.time_since_last_touchdown_lb = 0
        self.time_since_last_touchdown_lf = 0
        self.time_since_last_touchdown_rb = 0
        self.time_since_last_touchdown_rf = 0
        self._prev_joint_vel = np.zeros(self.env.model.nu)
        self.sum_tracking_performance_percentage = 0.0

    def step(self, action):
        self.time_since_last_touchdown_lb = 0 if self.env.check_collision("floor", "LB_foot") else self.time_since_last_touchdown_lb + self.env.dt
        self.time_since_last_touchdown_lf = 0 if self.env.check_collision("floor", "LF_foot") else self.time_since_last_touchdown_lf + self.env.dt
        self.time_since_last_touchdown_rb = 0 if self.env.check_collision("floor", "RB_foot") else self.time_since_last_touchdown_rb + self.env.dt
        self.time_since_last_touchdown_rf = 0 if self.env.check_collision("floor", "RF_foot") else self.time_since_last_touchdown_rf + self.env.dt
        self._prev_joint_vel = np.array(self.env.data.qvel[6:])

    def reward_and_info(self, info, done):
        total_timesteps = self.env.total_timesteps * self.env.nr_envs_per_type
        curriculum_coeff = min(total_timesteps / self.curriculum_steps, 1.0)
        if self.env.eval or self.env.mode == "test":
            curriculum_coeff = 1.0

        # Tracking velocity command reward
        current_global_linear_velocity = self.env.data.qvel[:3]
        current_local_linear_velocity = self.env.orientation_quat_inv.apply(current_global_linear_velocity)
        desired_local_linear_velocity_xy = np.array([self.env.goal_x_velocity, self.env.goal_y_velocity])
        xy_velocity_difference_norm = np.sum(np.square(desired_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * np.exp(-xy_velocity_difference_norm / self.xy_tracking_temperature)

        # Tracking angular velocity command reward
        current_local_angular_velocity = self.env.data.qvel[3:6]
        desired_local_yaw_velocity = self.env.goal_yaw_velocity
        yaw_velocity_difference_norm = np.sum(np.square(current_local_angular_velocity[2] - desired_local_yaw_velocity))
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * np.exp(-yaw_velocity_difference_norm / self.yaw_tracking_temperature)

        # Linear velocity reward
        z_velocity_squared = current_local_linear_velocity[2] ** 2
        linear_velocity_reward = curriculum_coeff * self.z_velocity_coeff * -z_velocity_squared

        # Angular velocity reward
        angular_velocity_norm = np.sum(np.square(current_local_angular_velocity[:2]))
        angular_velocity_reward = curriculum_coeff * self.pitch_roll_vel_coeff * -angular_velocity_norm

        # Angular position reward
        pitch_roll_position_norm = np.sum(np.square(self.env.orientation_euler[:2]))
        angular_position_reward = curriculum_coeff * self.pitch_roll_pos_coeff * -pitch_roll_position_norm

        # Joint nominal position difference reward
        joint_nominal_diff_norm = 0.0
        joint_nominal_diff_reward = curriculum_coeff * self.joint_nominal_diff_coeff * -joint_nominal_diff_norm

        # Joint position limit reward
        joint_positions = np.array(self.env.data.qpos[7:])
        lower_limit_penalty = -np.minimum(joint_positions - self.joint_limits[:, 0], 0.0).sum()
        upper_limit_penalty = np.maximum(joint_positions - self.joint_limits[:, 1], 0.0).sum()
        joint_position_limit_reward = curriculum_coeff * self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Joint velocity reward
        velocity_norm = np.sum(np.square(np.array(self.env.data.qvel[6:])))
        velocity_reward = curriculum_coeff * self.joint_velocity_coeff * -velocity_norm

        # Joint acceleration reward
        acceleration_norm = np.sum(np.square(self._prev_joint_vel - np.array(self.env.data.qvel[6:]) / self.env.dt))
        acceleration_reward = curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = np.sum(np.square(np.array(self.env.data.qfrc_actuator[6:])))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # Action rate reward
        action_rate_norm = np.sum(np.square(self.env.current_action - self.env.last_action))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm

        # Collision reward
        collisions = self.env.check_any_collision_for_all([
            "LB_thigh_1", "LF_thigh_1", "RB_thigh_1", "RF_thigh_1",
            "LB_thigh_2", "LF_thigh_2", "RB_thigh_2", "RF_thigh_2",
        ])
        trunk_collision = 1 if self.env.check_any_collision(["trunk_1", "trunk_2", "trunk_3", "trunk_4", "trunk_5", "trunk_6"]) else 0
        nr_collisions = sum(collisions.values()) + trunk_collision
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions

        # Walking height
        trunk_z = self.env.data.qpos[2] - self.env.terrain_function.center_height
        height_difference_squared = (trunk_z - self.nominal_trunk_z) ** 2
        base_height_reward = curriculum_coeff * self.base_height_coeff * -height_difference_squared

        # Air time reward
        air_time_reward = 0.0
        foot_lb_on_ground = self.env.check_collision("floor", "LB_foot")
        foot_lf_on_ground = self.env.check_collision("floor", "LF_foot")
        foot_rb_on_ground = self.env.check_collision("floor", "RB_foot")
        foot_rf_on_ground = self.env.check_collision("floor", "RF_foot")
        if foot_lb_on_ground:
            air_time_reward += self.time_since_last_touchdown_lb - self.air_time_max
        if foot_lf_on_ground:
            air_time_reward += self.time_since_last_touchdown_lf - self.air_time_max
        if foot_rb_on_ground:
            air_time_reward += self.time_since_last_touchdown_rb - self.air_time_max
        if foot_rf_on_ground:
            air_time_reward += self.time_since_last_touchdown_rf - self.air_time_max
        air_time_reward = curriculum_coeff * self.air_time_coeff * air_time_reward

        # Symmetry reward
        symmetry_air_violations = 0.0
        if not foot_rf_on_ground and not foot_lf_on_ground:
            symmetry_air_violations += 1
        if not foot_rb_on_ground and not foot_lb_on_ground:
            symmetry_air_violations += 1
        symmetry_air_reward = curriculum_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        # Total reward
        tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
        reward_penalty = linear_velocity_reward + angular_velocity_reward + angular_position_reward + joint_nominal_diff_reward + \
                         joint_position_limit_reward + velocity_reward + acceleration_reward + torque_reward + action_rate_reward + \
                         collision_reward + base_height_reward + air_time_reward + symmetry_air_reward
        reward = tracking_reward + reward_penalty
        reward = max(reward, 0.0)

        # More logging metrics
        # power = np.sum(abs(self.env.current_torques) * abs(self.env.data.qvel[6:]))
        # mass_of_robot = np.sum(self.env.model.body_mass)
        # gravity = -self.env.model.opt.gravity[2]
        # velocity = np.linalg.norm(current_local_linear_velocity)
        # cost_of_transport = power / (mass_of_robot * gravity * velocity)
        # froude_number = velocity ** 2 / (gravity * trunk_z)
        current_global_velocities = np.array([current_local_linear_velocity[0], current_local_linear_velocity[1], current_local_angular_velocity[2]])
        desired_global_velocities = np.array([desired_local_linear_velocity_xy[0], desired_local_linear_velocity_xy[1], desired_local_yaw_velocity])
        tracking_performance_percentage = max(np.mean(1 - (np.abs(current_global_velocities - desired_global_velocities) / np.abs(desired_global_velocities))), 0.0)
        self.sum_tracking_performance_percentage += tracking_performance_percentage
        if done:
            episode_tracking_performance_percentage = self.sum_tracking_performance_percentage / self.env.horizon

        # Info
        # info[f"reward/{self.env.SHORT_NAME}/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        # info[f"reward/{self.env.SHORT_NAME}/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        # info[f"reward/{self.env.SHORT_NAME}/linear_velocity"] = linear_velocity_reward
        # info[f"reward/{self.env.SHORT_NAME}/angular_velocity"] = angular_velocity_reward
        # info[f"reward/{self.env.SHORT_NAME}/angular_position"] = angular_position_reward
        # info[f"reward/{self.env.SHORT_NAME}/joint_nominal_diff"] = joint_nominal_diff_reward
        # info[f"reward/{self.env.SHORT_NAME}/joint_position_limit"] = joint_position_limit_reward
        # info[f"reward/{self.env.SHORT_NAME}/torque"] = torque_reward
        # info[f"reward/{self.env.SHORT_NAME}/acceleration"] = acceleration_reward
        # info[f"reward/{self.env.SHORT_NAME}/velocity"] = velocity_reward
        # info[f"reward/{self.env.SHORT_NAME}/action_rate"] = action_rate_reward
        # info[f"reward/{self.env.SHORT_NAME}/collision"] = collision_reward
        # info[f"reward/{self.env.SHORT_NAME}/base_height"] = base_height_reward
        # info[f"reward/{self.env.SHORT_NAME}/air_time"] = air_time_reward
        # info[f"reward/{self.env.SHORT_NAME}/symmetry_air"] = symmetry_air_reward
        # info["env_info/target_x_vel"] = desired_local_velocity_x
        # info["env_info/target_y_vel"] = desired_local_velocity_y
        # info["env_info/current_x_vel"] = current_trunk_velocity[0]
        # info["env_info/current_y_vel"] = current_trunk_velocity[1]
        info[f"env_info/track_perf_perc/{self.env.SHORT_NAME}"] = tracking_performance_percentage
        if done:
            info[f"env_info/eps_track_perf_perc/{self.env.SHORT_NAME}"] = episode_tracking_performance_percentage
        # info["env_info/symmetry_violations"] = symmetry_air_violations
        # info["env_info/walk_height"] = trunk_z
        # info["env_info/xy_vel_diff_norm"] = xy_velocity_difference_norm
        # info["env_info/yaw_vel_diff_norm"] = yaw_velocity_difference_norm
        # info["env_info/torque_norm"] = torque_norm
        # info["env_info/acceleration_norm"] = acceleration_norm
        # info["env_info/velocity_norm"] = velocity_norm
        # info["env_info/action_rate_norm"] = action_rate_norm
        # info["env_info/power"] = power
        # info["env_info/cost_of_transport"] = cost_of_transport
        # info["env_info/froude_number"] = froude_number

        return reward, info
