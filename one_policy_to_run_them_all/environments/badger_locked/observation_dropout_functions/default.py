import numpy as np


class DefaultObservationDropout:
    def __init__(self, env, dynamic_dropout_chance=0.05):
        self.env = env
        self.dynamic_dropout_chance = dynamic_dropout_chance

    def init(self):
        joint_position_names = [joint_name + "_position" for joint_name in self.env.joint_names]
        joint_velocity_names = [joint_name + "_velocity" for joint_name in self.env.joint_names]
        joint_previous_action_names = [joint_name + "_previous_action" for joint_name in self.env.joint_names]

        self.joint_position_ids = [self.env.observation_name_to_id[name] for name in joint_position_names]
        self.joint_velocity_ids = [self.env.observation_name_to_id[name] for name in joint_velocity_names]
        self.joint_previous_action_ids = [self.env.observation_name_to_id[name] for name in joint_previous_action_names]

    def modify_observation(self, obs):
        joint_mask = self.env.np_rng.uniform(0, 1, len(self.joint_position_ids)) < self.dynamic_dropout_chance
        for ids in [self.joint_position_ids, self.joint_velocity_ids, self.joint_previous_action_ids]:
            obs[ids] = np.where(joint_mask, self.env.missing_value, obs[ids])

        return obs
