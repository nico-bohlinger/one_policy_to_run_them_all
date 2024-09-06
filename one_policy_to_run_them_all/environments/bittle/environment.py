from pathlib import Path
import psutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from dm_control import mjcf
import pygame

from one_policy_to_run_them_all.environments.bittle.viewer import MujocoViewer
from one_policy_to_run_them_all.environments.bittle.control_functions.handler import get_control_function
from one_policy_to_run_them_all.environments.bittle.command_functions.handler import get_command_function
from one_policy_to_run_them_all.environments.bittle.sampling_functions.handler import get_sampling_function
from one_policy_to_run_them_all.environments.bittle.initial_state_functions.handler import get_initial_state_function
from one_policy_to_run_them_all.environments.bittle.reward_functions.handler import get_reward_function
from one_policy_to_run_them_all.environments.bittle.termination_functions.handler import get_termination_function
from one_policy_to_run_them_all.environments.bittle.domain_randomization.action_delay_functions.handler import get_get_domain_randomization_action_delay_function
from one_policy_to_run_them_all.environments.bittle.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from one_policy_to_run_them_all.environments.bittle.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from one_policy_to_run_them_all.environments.bittle.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from one_policy_to_run_them_all.environments.bittle.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from one_policy_to_run_them_all.environments.bittle.observation_noise_functions.handler import get_observation_noise_function
from one_policy_to_run_them_all.environments.bittle.observation_dropout_functions.handler import get_observation_dropout_function
from one_policy_to_run_them_all.environments.bittle.terrain_functions.handler import get_terrain_function


class Bittle(gym.Env):
    LONG_NAME = "bittle"
    SHORT_NAME = "bit"

    def __init__(self, seed, render,
                 mode,
                 control_type, command_type, command_sampling_type, initial_state_type,
                 reward_type, termination_type,
                 domain_randomization_sampling_type,
                 domain_randomization_action_delay_type,
                 domain_randomization_mujoco_model_type,
                 domain_randomization_seen_robot_type, domain_randomization_unseen_robot_type,
                 domain_randomization_perturbation_type, domain_randomization_perturbation_sampling_type,
                 observation_noise_type, observation_dropout_type, terrain_type,
                 missing_value,
                 add_goal_arrow, timestep, episode_length_in_seconds, total_nr_envs,
                 multi_robot_max_observation_size=-1, multi_robot_max_action_size=-1, nr_envs_per_type=1,
                 cpu_id=None):
        
        if cpu_id is not None:
            p = psutil.Process()
            p.cpu_affinity([cpu_id,])

        self.seed = seed
        self.mode = mode
        self.missing_value = missing_value
        self.add_goal_arrow = add_goal_arrow
        self.total_nr_envs = total_nr_envs
        self.nr_envs_per_type = nr_envs_per_type
        self.eval = False
        self.eval_at_last_setup = self.eval
        self.np_rng = np.random.default_rng(self.seed)
        self.nominal_joint_positions = np.array([
            0.6, 0.5,
            0.6, 0.5,
            -0.6, -0.5,
            -0.6, -0.5,
        ])
        self.max_joint_velocities = np.array([14.96] * 8)
        self.initial_drop_height = 0.094

        self.total_timesteps = 0
        self.goal_x_velocity = 0
        self.goal_y_velocity = 0
        self.goal_yaw_velocity = 0

        if mode == "test":
            initial_state_type = "default"
            domain_randomization_sampling_type = "none"
            domain_randomization_perturbation_sampling_type = "none"
            observation_noise_type = "none"
            observation_dropout_type = "none"

        self.control_function = get_control_function(control_type, self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / timestep))
        self.nr_intermediate_steps = 1
        self.dt = timestep * self.nr_substeps * self.nr_intermediate_steps
        self.horizon = int(round(episode_length_in_seconds * self.control_frequency_hz))
        self.command_function = get_command_function(command_type, self)
        self.command_sampling_function = get_sampling_function(command_sampling_type, self)
        self.initial_state_function = get_initial_state_function(initial_state_type, self)
        self.reward_function = get_reward_function(reward_type, self)
        self.termination_function = get_termination_function(termination_type, self)
        self.domain_randomization_sampling_function = get_sampling_function(domain_randomization_sampling_type, self)
        self.domain_randomization_action_delay_function = get_get_domain_randomization_action_delay_function(domain_randomization_action_delay_type, self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(domain_randomization_mujoco_model_type, self)
        self.domain_randomization_seen_robot_function = get_domain_randomization_seen_robot_function(domain_randomization_seen_robot_type, self)
        self.domain_randomization_unseen_robot_function = get_domain_randomization_unseen_robot_function(domain_randomization_unseen_robot_type, self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(domain_randomization_perturbation_type, self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(domain_randomization_perturbation_sampling_type, self)
        self.observation_noise_function = get_observation_noise_function(observation_noise_type, self)
        self.observation_dropout_function = get_observation_dropout_function(observation_dropout_type, self)
        self.terrain_function = get_terrain_function(terrain_type, self)

        xml_file_name = self.terrain_function.xml_file_name
        xml_path = (Path(__file__).resolve().parent / "data" / xml_file_name).as_posix()
        if self.add_goal_arrow:
            # Add goal arrow
            xml_handle = mjcf.from_path(xml_path)
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.08")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
            self.model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)

        collision_groups = [("floor", ["floor"]),
                            ("feet", ["LB_foot_1", "LB_foot_2", "LF_foot_1", "LF_foot_2", "RB_foot_1", "RB_foot_2", "RF_foot_1", "RF_foot_2"]),
                            ("LB_foot", ["LB_foot_1", "LB_foot_2"]), ("LF_foot", ["LF_foot_1", "LF_foot_2"]), ("RB_foot", ["RB_foot_1", "RB_foot_2"]), ("RF_foot", ["RF_foot_1", "RF_foot_2"]),
                            ("LB_thigh_1", ["LB_thigh_1"]), ("LF_thigh_1", ["LF_thigh_1"]), ("RB_thigh_1", ["RB_thigh_1"]), ("RF_thigh_1", ["RF_thigh_1"]),
                            ("LB_thigh_2", ["LB_thigh_2"]), ("LF_thigh_2", ["LF_thigh_2"]), ("RB_thigh_2", ["RB_thigh_2"]), ("RF_thigh_2", ["RF_thigh_2"]),
                            ("trunk", ["trunk_1", "trunk_2", "trunk_3", "trunk_4", "trunk_5", "trunk_6"]),
                            ("trunk_1", ["trunk_1"]), ("trunk_2", ["trunk_2"]), ("trunk_3", ["trunk_3"]), ("trunk_4", ["trunk_4"]), ("trunk_5", ["trunk_5"]), ("trunk_6", ["trunk_6"])]
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                self.collision_groups[name] = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) for geom_name in geom_names}

        self.viewer = None if not render else MujocoViewer(self.model, self.dt)

        action_space_size = self.model.nu
        self.missing_nr_of_actions = 0
        if multi_robot_max_action_size != -1:
            self.multi_robot_max_observation_size = multi_robot_max_observation_size
            self.missing_nr_of_actions = multi_robot_max_action_size - action_space_size
            action_space_size = multi_robot_max_action_size
        action_space_low = -np.ones(action_space_size) * np.Inf
        action_space_high = np.ones(action_space_size) * np.Inf
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        self.foot_names = ["LB_foot", "LF_foot", "RB_foot", "RF_foot"]
        self.joint_names = [
            "left_back_shoulder_joint", "left_back_knee_joint",
            "left_front_shoulder_joint", "left_front_knee_joint",
            "right_back_shoulder_joint", "right_back_knee_joint",
            "right_front_shoulder_joint", "right_front_knee_joint"
        ]
        self.joint_nr_direct_child_joints = [
            1, 0,
            1, 0,
            1, 0,
            1, 0
        ]
        
        qpos, qvel = get_initial_state_function("default", self).setup()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()

        self.domain_randomization_seen_robot_function.init()
        
        self.name_to_description_vector = self.get_name_to_description_vector()

        self.observation_space, self.observation_name_to_id = self.get_observation_space(multi_robot_max_observation_size)

        self.current_action = np.zeros(self.model.nu)
        self.initial_observation = self.get_initial_observation(multi_robot_max_observation_size)
        
        # Idx that need to be updated every step
        self.joint_positions_update_obs_idx = [self.observation_name_to_id[joint_name + "_position"] for joint_name in self.joint_names]
        self.joint_velocities_update_obs_idx = [self.observation_name_to_id[joint_name + "_velocity"] for joint_name in self.joint_names]
        self.joint_previous_actions_update_obs_idx = [self.observation_name_to_id[joint_name + "_previous_action"] for joint_name in self.joint_names]
        self.foot_ground_contact_update_obs_idx = [self.observation_name_to_id[foot_name + "_ground_contact"] for foot_name in self.foot_names]
        self.foot_time_since_last_ground_contact_update_obs_idx = [self.observation_name_to_id[foot_name + "_cycles_since_last_ground_contact"] for foot_name in self.foot_names]
        self.trunk_linear_vel_update_obs_idx = [self.observation_name_to_id["trunk_" + observation_name] for observation_name in ["x_velocity", "y_velocity", "z_velocity"]]
        self.trunk_angular_vel_update_obs_idx = [self.observation_name_to_id["trunk_" + observation_name] for observation_name in ["roll_velocity", "pitch_velocity", "yaw_velocity"]]
        self.goal_velocity_update_obs_idx = [self.observation_name_to_id["goal_" + observation_name] for observation_name in ["x_velocity", "y_velocity", "yaw_velocity"]]
        self.projected_gravity_update_obs_idx = [self.observation_name_to_id["projected_gravity_" + observation_name] for observation_name in ["x", "y", "z"]]
        self.height_update_obs_idx = [self.observation_name_to_id["height_0"]]

        self.reward_function.init()
        self.domain_randomization_mujoco_model_function.init()
        self.observation_noise_function.init()
        self.observation_dropout_function.init()

        if self.mode == "test":
            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True


    def reset(self, seed=None):
        self.episode_step = 0
        self.current_action = np.zeros(self.model.nu)
        self.last_action = np.zeros(self.model.nu)
        self.current_torques = np.zeros(self.model.nu)

        self.terrain_function.sample()
        self.command_function.setup()
        self.reward_function.setup()
        self.handle_domain_randomization(function="setup")

        qpos, qvel = self.initial_state_function.setup()

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.update_orientation_attributes()

        if self.viewer:
            self.viewer.render(self.data)

        return self.get_observation(), {}


    def step(self, action):
        explicit_commands = False
        if self.mode == "test":
            if self.joystick_present:
                pygame.event.pump()
                self.goal_x_velocity = -self.joystick.get_axis(1)
                self.goal_y_velocity = -self.joystick.get_axis(0)
                self.goal_yaw_velocity = -self.joystick.get_axis(3)
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    self.goal_x_velocity = float(commands[0])
                    self.goal_y_velocity = float(commands[1])
                    self.goal_yaw_velocity = float(commands[2])
                    explicit_commands = True

        if not explicit_commands:
            should_sample_commands = self.command_sampling_function.step()
            if should_sample_commands or self.total_timesteps == 0:
                self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity = self.command_function.get_next_command()

        action = self.domain_randomization_action_delay_function.delay_action(action[:self.model.nu])

        torques = self.control_function.process_action(action)

        for _ in range(self.nr_intermediate_steps):
            self.data.ctrl = torques
            mujoco.mj_step(self.model, self.data, self.nr_substeps)
            self.data.qvel[6:] = np.clip(self.data.qvel[6:], -self.max_joint_velocities, self.max_joint_velocities)

        self.update_orientation_attributes()

        if self.add_goal_arrow:
            trunk_rotation = self.orientation_euler[2]
            desired_angle = trunk_rotation + np.arctan2(self.goal_y_velocity, self.goal_x_velocity)
            rot_mat = R.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([self.goal_x_velocity, self.goal_y_velocity])))
            self.model.site_size[1, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.data.site("dir_arrow_ball").xpos = self.data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]

        if self.viewer:
            self.viewer.render(self.data)
        
        self.current_action = action.copy()
        self.current_torques = torques

        self.handle_domain_randomization(function="step")

        next_observation = self.get_observation()
        terminated = self.termination_function.should_terminate(next_observation)
        truncated = self.episode_step + 1 >= self.horizon
        done = terminated | truncated
        reward, r_info = self.get_reward(done)
        info = {**r_info}

        self.reward_function.step(action)
        self.command_function.step(next_observation, reward, done, info)
        self.initial_state_function.step(next_observation, reward, done, info)
        self.terrain_function.step(next_observation, reward, done, info)
        
        self.last_action = action.copy()
        self.episode_step += 1
        if not self.eval:
            self.total_timesteps += 1

        return next_observation, reward, terminated, truncated, info
    

    def update_orientation_attributes(self):
        self.orientation_quat = R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        self.orientation_euler = self.orientation_quat.as_euler("xyz")
        self.orientation_quat_inv = self.orientation_quat.inv()


    def handle_domain_randomization(self, function="setup"):
        if function == "setup":
            if self.eval_at_last_setup != self.eval:
                self.should_randomize_domain = True
                self.should_randomize_domain_perturbation = True
                self.eval_at_last_setup = self.eval
            else:
                self.should_randomize_domain = self.domain_randomization_sampling_function.setup()
                self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.setup()
            self.domain_randomization_action_delay_function.setup()
        elif function == "step":
            self.should_randomize_domain = self.domain_randomization_sampling_function.step()
            self.should_randomize_domain_perturbation = self.domain_randomization_perturbation_sampling_function.step()
        if self.should_randomize_domain:
            self.domain_randomization_unseen_robot_function.sample()
            self.domain_randomization_seen_robot_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.domain_randomization_action_delay_function.sample()
            self.name_to_description_vector = self.get_name_to_description_vector()
            self.initial_observation = self.get_initial_observation(self.multi_robot_max_observation_size)
            self.reward_function.init()
        if self.should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()


    def get_name_to_description_vector(self):
        name_to_description_vector = {}

        # Save initial state
        qpos_old, qvel_old = self.data.qpos.copy(), self.data.qvel.copy()

        # Move to initial position
        qpos, qvel = get_initial_state_function("default", self).setup()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        trunk_pos = self.data.body("trunk")

        # Calculate robot width, length and height
        relative_geom_positions = self.data.geom_xpos - trunk_pos.xpos
        for i in range(len(relative_geom_positions)):
            relative_geom_positions[i] = np.matmul(trunk_pos.xmat.reshape(3, 3).T, relative_geom_positions[i])

        relative_joint_positions = np.zeros((len(self.joint_names), 3))
        for i, joint in enumerate(self.joint_names):
            relative_joint_positions[i] = self.data.joint(joint).xanchor - trunk_pos.xpos
            relative_joint_positions[i] = np.matmul(trunk_pos.xmat.reshape(3, 3).T, relative_joint_positions[i])

        rotated_geom_sizes = np.zeros_like(self.model.geom_size)
        for i in range(len(rotated_geom_sizes)):
            global_frame_geom_size = np.matmul(self.data.geom_xmat[i].reshape(3, 3), self.model.geom_size[i])
            trunk_frame_geom_size = np.matmul(trunk_pos.xmat.reshape(3, 3).T, global_frame_geom_size)
            rotated_geom_sizes[i] = np.abs(trunk_frame_geom_size)

        relative_geom_positions_minus_geom_size = relative_geom_positions.copy()
        relative_geom_positions_plus_geom_size = relative_geom_positions.copy()
        for i in range(len(relative_geom_positions)):
            relative_geom_positions_minus_geom_size[i] -= rotated_geom_sizes[i]
            relative_geom_positions_plus_geom_size[i] += rotated_geom_sizes[i]

        # Ignore the first geom (floor)
        min_x = min(min(relative_geom_positions_minus_geom_size[1:, 0]), min(relative_geom_positions_plus_geom_size[1:, 0]), min(relative_joint_positions[:, 0]))
        min_y = min(min(relative_geom_positions_minus_geom_size[1:, 1]), min(relative_geom_positions_plus_geom_size[1:, 1]), min(relative_joint_positions[:, 1]))
        min_z = min(min(relative_geom_positions_minus_geom_size[1:, 2]), min(relative_geom_positions_plus_geom_size[1:, 2]), min(relative_joint_positions[:, 2]))
        max_x = max(max(relative_geom_positions_minus_geom_size[1:, 0]), max(relative_geom_positions_plus_geom_size[1:, 0]), max(relative_joint_positions[:, 0]))
        max_y = max(max(relative_geom_positions_minus_geom_size[1:, 1]), max(relative_geom_positions_plus_geom_size[1:, 1]), max(relative_joint_positions[:, 1]))
        max_z = max(max(relative_geom_positions_minus_geom_size[1:, 2]), max(relative_geom_positions_plus_geom_size[1:, 2]), max(relative_joint_positions[:, 2]))
        mins = np.array([min_x, min_y, min_z])
        self.robot_length = max_x - min_x
        self.robot_width = max_y - min_y
        self.robot_height = max_z - min_z
        self.robot_dimensions = np.array([self.robot_length, self.robot_width, self.robot_height])

        self.gains_and_action_scaling_factor = np.array([self.control_function.seen_p_gain, self.control_function.seen_d_gain, self.control_function.seen_scaling_factor])
        self.mass = np.array([np.sum(self.domain_randomization_seen_robot_function.seen_body_mass)])
        
        for foot_name in self.foot_names:
            foot = self.data.geom(foot_name + "_1")
            relative_foot_pos = foot.xpos - trunk_pos.xpos
            relative_foot_pos = np.matmul(trunk_pos.xmat.reshape(3, 3).T, relative_foot_pos)
            relative_foot_pos_normalized = (relative_foot_pos - mins) / self.robot_dimensions
            name_to_description_vector[foot_name] = np.concatenate([
                (relative_foot_pos_normalized / 0.5) - 1.0,
                (self.gains_and_action_scaling_factor / [100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0,
                (self.mass / (170.0 / 2)) - 1.0,
                (self.robot_dimensions / (2.0 / 2)) - 1.0
            ])

        for i, joint_name in enumerate(self.joint_names):
            joint = self.data.joint(joint_name)
            relative_joint_pos = joint.xanchor - trunk_pos.xpos
            relative_joint_pos = np.matmul(trunk_pos.xmat.reshape(3, 3).T, relative_joint_pos)
            relative_joint_pos_normalized = (relative_joint_pos - mins) / self.robot_dimensions
            joint_axis = joint.xaxis
            relative_joint_axis = np.matmul(trunk_pos.xmat.reshape(3, 3).T, joint_axis)
            name_to_description_vector[joint_name] = np.concatenate([
                (relative_joint_pos_normalized / 0.5) - 1.0,
                relative_joint_axis,
                np.array([self.joint_nr_direct_child_joints[i] - 1.0]),
                np.array([self.domain_randomization_seen_robot_function.seen_joint_nominal_position[i] / 4.6]),
                np.array([(self.domain_randomization_seen_robot_function.seen_torque_limit[i] / (1000.0 / 2)) - 1.0]),
                np.array([(self.domain_randomization_seen_robot_function.seen_joint_velocity[i] / (35.0 / 2)) - 1.0]),
                np.array([(self.domain_randomization_seen_robot_function.seen_joint_damping[i] / (10.0 / 2)) - 1.0]),
                np.array([(self.domain_randomization_seen_robot_function.seen_joint_armature[i] / (0.2 / 2)) - 1.0]),
                np.array([(self.domain_randomization_seen_robot_function.seen_joint_stiffness[i] / (30.0 / 2)) - 1.0]),
                np.array([(self.domain_randomization_seen_robot_function.seen_joint_frictionloss[i] / (1.2 / 2)) - 1.0]),
                self.domain_randomization_seen_robot_function.seen_joint_range[i] / 4.6,
                (self.gains_and_action_scaling_factor / [100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0,
                (self.mass / (170.0 / 2)) - 1.0,
                (self.robot_dimensions / (2.0 / 2)) - 1.0
            ])

        self.dynamic_joint_description_size = name_to_description_vector[self.joint_names[0]].shape[0]
        self.dynamic_foot_description_size = name_to_description_vector[self.foot_names[0]].shape[0]

        # Restore initial state
        self.data.qpos[:] = qpos_old
        self.data.qvel[:] = qvel_old
        mujoco.mj_forward(self.model, self.data)

        return name_to_description_vector


    def get_observation_space(self, multi_robot_max_observation_size):
        observation_names = []
        space_low, space_high = [], []

        # Dynamic observations
        self.nr_dynamic_joint_observations = len(self.joint_names)
        self.single_dynamic_joint_observation_length = self.dynamic_joint_description_size + 3
        self.dynamic_joint_observation_length = self.single_dynamic_joint_observation_length * self.nr_dynamic_joint_observations
        for joint_name in self.joint_names:
            observation_names.extend([joint_name + "_description_" + str(i) for i in range(self.dynamic_joint_description_size)])
            observation_names.extend([
                joint_name + "_position", joint_name + "_velocity", joint_name + "_previous_action",
            ])
            space_low.extend([-np.inf] * (self.single_dynamic_joint_observation_length))
            space_high.extend([np.inf] * (self.single_dynamic_joint_observation_length))

        self.nr_dynamic_foot_observations = len(self.foot_names)
        self.single_dynamic_foot_observation_length = self.dynamic_foot_description_size + 2
        self.dynamic_foot_observation_length = self.single_dynamic_foot_observation_length * self.nr_dynamic_foot_observations
        for foot_name in self.foot_names:
            observation_names.extend([foot_name + "_description_" + str(i) for i in range(self.dynamic_foot_description_size)])
            observation_names.extend([
                foot_name + "_ground_contact", foot_name + "_cycles_since_last_ground_contact",
            ])
            space_low.extend([-np.inf] * (self.single_dynamic_foot_observation_length))
            space_high.extend([np.inf] * (self.single_dynamic_foot_observation_length))
        
        # General observations
        observation_names.extend([
            "trunk_x_velocity", "trunk_y_velocity", "trunk_z_velocity",
            "trunk_roll_velocity", "trunk_pitch_velocity", "trunk_yaw_velocity",
        ])
        space_low.extend([-np.inf] * 6)
        space_high.extend([np.inf] * 6)

        observation_names.extend(["goal_x_velocity", "goal_y_velocity", "goal_yaw_velocity"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        observation_names.extend(["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        observation_names.append("height_0")
        space_low.append(-np.inf)
        space_high.append(np.inf)

        # General robot context
        observation_names.extend(["p_gain", "d_gain", "action_scaling_factor"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        observation_names.append("mass")
        space_low.append(-np.inf)
        space_high.append(np.inf)

        observation_names.extend(["robot_length", "robot_width", "robot_height"])
        space_low.extend([-np.inf] * 3)
        space_high.extend([np.inf] * 3)

        self.missing_nr_of_observations = 0
        if multi_robot_max_observation_size != -1:
            self.missing_nr_of_observations = multi_robot_max_observation_size - len(observation_names)
            observation_names.extend(["padding_" + str(i) for i in range(self.missing_nr_of_observations)])
            space_low.extend([-np.inf] * self.missing_nr_of_observations)
            space_high.extend([np.inf] * self.missing_nr_of_observations)

        space_low = np.array(space_low, dtype=np.float32)
        space_high = np.array(space_high, dtype=np.float32)

        name_to_idx = {name: idx for idx, name in enumerate(observation_names)}

        return gym.spaces.Box(low=space_low, high=space_high, shape=space_low.shape, dtype=np.float32), name_to_idx


    def get_initial_observation(self, multi_robot_max_observation_size):
        # Dynamic observations
        dynamic_joint_observations = []
        for i, joint_name in enumerate(self.joint_names):
            dynamic_joint_observations.extend(self.name_to_description_vector[joint_name])
            dynamic_joint_observations.append(self.data.qpos[i+7] - self.domain_randomization_seen_robot_function.seen_joint_nominal_position[i])
            dynamic_joint_observations.append(self.data.qvel[i+6])
            dynamic_joint_observations.append(self.current_action[i])
        dynamic_joint_observations = np.array(dynamic_joint_observations, dtype=np.float32)

        dynamic_foot_observations = []
        for foot_name in self.foot_names:
            dynamic_foot_observations.extend(self.name_to_description_vector[foot_name])
            dynamic_foot_observations.append(self.check_collision("floor", foot_name))
            if foot_name == "LB_foot":
                dynamic_foot_observations.append(self.reward_function.time_since_last_touchdown_lb)
            elif foot_name == "LF_foot":
                dynamic_foot_observations.append(self.reward_function.time_since_last_touchdown_lf)
            elif foot_name == "RB_foot":
                dynamic_foot_observations.append(self.reward_function.time_since_last_touchdown_rb)
            elif foot_name == "RF_foot":
                dynamic_foot_observations.append(self.reward_function.time_since_last_touchdown_rf)
        dynamic_foot_observations = np.array(dynamic_foot_observations, dtype=np.float32)

        # General observations
        trunk_linear_velocity = self.orientation_quat_inv.apply(self.data.qvel[:3])
        trunk_angular_velocity = self.data.qvel[3:6]
        goal_velocity = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        projected_gravity_vector = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        height = np.array([(self.data.qpos[2] - self.terrain_function.center_height) / self.robot_height])

        # General robot context
        gains_and_action_scaling_factor = (self.gains_and_action_scaling_factor / [100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0
        mass = (self.mass / (170.0 / 2)) - 1.0
        robot_dimensions = (self.robot_dimensions / (2.0 / 2)) - 1.0

        # Padding
        padding = np.array([])
        if multi_robot_max_observation_size != -1:
            padding = np.zeros(self.missing_nr_of_observations, dtype=np.float32)

        observation = np.concatenate([
            dynamic_joint_observations,
            dynamic_foot_observations,
            trunk_linear_velocity,
            trunk_angular_velocity,
            goal_velocity,
            projected_gravity_vector,
            height,
            gains_and_action_scaling_factor,
            mass,
            robot_dimensions,
            padding
        ])
        
        return observation
    

    def get_observation(self):
        observation = self.initial_observation.copy()

        # Update observations every step
        observation[self.joint_positions_update_obs_idx] = self.data.qpos[7:] - self.domain_randomization_seen_robot_function.seen_joint_nominal_position
        observation[self.joint_velocities_update_obs_idx] = self.data.qvel[6:]
        observation[self.joint_previous_actions_update_obs_idx] = self.current_action
        observation[self.foot_ground_contact_update_obs_idx] = np.array([self.check_collision("floor", foot_name) for foot_name in self.foot_names])
        observation[self.foot_time_since_last_ground_contact_update_obs_idx] = np.array([self.reward_function.time_since_last_touchdown_lb, self.reward_function.time_since_last_touchdown_lf, self.reward_function.time_since_last_touchdown_rb, self.reward_function.time_since_last_touchdown_rf])
        observation[self.trunk_linear_vel_update_obs_idx] = self.orientation_quat_inv.apply(self.data.qvel[:3])
        observation[self.trunk_angular_vel_update_obs_idx] = self.data.qvel[3:6]
        observation[self.goal_velocity_update_obs_idx] = np.array([self.goal_x_velocity, self.goal_y_velocity, self.goal_yaw_velocity])
        observation[self.projected_gravity_update_obs_idx] = self.orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))
        observation[self.height_update_obs_idx] = np.array([self.data.qpos[2] - self.terrain_function.center_height])

        # Add noise
        observation = self.observation_noise_function.modify_observation(observation)

        # Dropout
        observation = self.observation_dropout_function.modify_observation(observation)

        # Normalize and clip
        observation[self.joint_positions_update_obs_idx] /= 4.6
        observation[self.joint_velocities_update_obs_idx] /= 35.0
        observation[self.joint_previous_actions_update_obs_idx] /= 10.0
        observation[self.foot_ground_contact_update_obs_idx] = (observation[self.foot_ground_contact_update_obs_idx] / 0.5) - 1.0
        observation[self.foot_time_since_last_ground_contact_update_obs_idx] = np.clip((observation[self.foot_time_since_last_ground_contact_update_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.trunk_linear_vel_update_obs_idx] = np.clip(observation[self.trunk_linear_vel_update_obs_idx] / 10.0, -1.0, 1.0)
        observation[self.trunk_angular_vel_update_obs_idx] = np.clip(observation[self.trunk_angular_vel_update_obs_idx] / 50.0, -1.0, 1.0)
        observation[self.height_update_obs_idx] = np.clip((observation[self.height_update_obs_idx] / (2*self.robot_height / 2)) - 1.0, -1.0, 1.0)

        return observation


    def get_reward(self, done):
        info = {"t": self.episode_step}
        return self.reward_function.reward_and_info(info, done)


    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.mode == "test":
            pygame.quit()


    def check_collision(self, groups1, groups2):
        if isinstance(groups1, list):
            ids1 = [self.collision_groups[group] for group in groups1]
            ids1 = set().union(*ids1)
        else:
            ids1 = self.collision_groups[groups1]
        
        if isinstance(groups2, list):
            ids2 = [self.collision_groups[group] for group in groups2]
            ids2 = set().union(*ids2)
        else:
            ids2 = self.collision_groups[groups2]

        for coni in range(0, self.data.ncon):
            con = self.data.contact[coni]

            collision = con.geom1 in ids1 and con.geom2 in ids2
            collision_trans = con.geom1 in ids2 and con.geom2 in ids1

            if collision or collision_trans:
                return True

        return False
    

    def check_any_collision(self, groups):
        if isinstance(groups, list):
            ids = [self.collision_groups[group] for group in groups]
            ids = set().union(*ids)
        else:
            ids = self.collision_groups[groups]

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                return True
        
        return False


    def check_any_collision_for_all(self, groups):
        ids = [self.collision_groups[group] for group in groups]
        ids = set().union(*ids)

        any_collision = {idx: False for idx in ids}

        for con_i in range(0, self.data.ncon):
            con = self.data.contact[con_i]
            if con.geom1 in ids:
                any_collision[con.geom1] = True
                ids.remove(con.geom1)
            if con.geom2 in ids:
                any_collision[con.geom2] = True
                ids.remove(con.geom2)
        
        return any_collision
