import gymnasium as gym

from one_policy_to_run_them_all.environments.atlas.environment import Atlas
from one_policy_to_run_them_all.environments.atlas.wrappers import RLXInfo, RecordEpisodeStatistics
from one_policy_to_run_them_all.environments.atlas.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed):
        def thunk():
            env = Atlas(
                seed=seed,
                render=config.environment.render,
                mode=config.environment.mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=config.environment.termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_control_type=config.environment.domain_randomization_control_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                terrain_type=config.environment.terrain_type,
                missing_value=config.environment.missing_value,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    env = vector_environment_class([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    return env
