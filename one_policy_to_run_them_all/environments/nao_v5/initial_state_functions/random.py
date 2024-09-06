import numpy as np
from scipy.spatial.transform import Rotation as R


class RandomInitialState:
    def __init__(self, env,
                 roll_angle_factor=0.01,
                 pitch_angle_factor=0.01,
                 yaw_angle_factor=1.0,
                 nominal_joint_position_factor=0.1,
                 joint_velocity_factor=0.1,
                 max_linear_velocity=0.1,
                 max_angular_velocity=0.1,
        ):
        self.env = env
        self.roll_angle_factor = roll_angle_factor
        self.pitch_angle_factor = pitch_angle_factor
        self.yaw_angle_factor = yaw_angle_factor
        self.nominal_joint_position_factor = nominal_joint_position_factor
        self.joint_velocity_factor = joint_velocity_factor
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

    def step(self, obs, reward, absorbing, info):
        return

    def setup(self):
        roll_angle = self.env.np_rng.uniform(-np.pi * self.roll_angle_factor, np.pi * self.roll_angle_factor)
        pitch_angle = self.env.np_rng.uniform(-np.pi * self.pitch_angle_factor, np.pi * self.pitch_angle_factor)
        yaw_angle = self.env.np_rng.uniform(-np.pi * self.yaw_angle_factor, np.pi * self.yaw_angle_factor)
        quaternion = R.from_euler('xyz', [roll_angle, pitch_angle, yaw_angle]).as_quat()
        quaternion = quaternion[[3, 0, 1, 2]]
        joint_positions = self.env.nominal_joint_positions * self.env.np_rng.uniform(self.nominal_joint_position_factor, 1 + self.nominal_joint_position_factor, size=(self.env.nominal_joint_positions.shape[0],))
        joint_velocities = self.env.max_joint_velocities * self.env.np_rng.uniform(-self.joint_velocity_factor, self.joint_velocity_factor)
        linear_velocities = self.env.np_rng.uniform(-self.max_linear_velocity, self.max_linear_velocity, size=(3,))
        angular_velocities = self.env.np_rng.uniform(-self.max_angular_velocity, self.max_angular_velocity, size=(3,))

        qpos = self.env.complete_nominal_joint_position.copy()
        qpos[2] += self.env.terrain_function.center_height
        qpos[[3, 4, 5, 6]] = quaternion
        qpos[self.env.joint_mask_qpos] = joint_positions

        qvel = np.zeros(self.env.data.qvel.shape)
        qvel[[0, 1, 2]] = linear_velocities
        qvel[[3, 4, 5]] = angular_velocities
        qvel[self.env.joint_mask_qvel] = joint_velocities
        
        return qpos, qvel
