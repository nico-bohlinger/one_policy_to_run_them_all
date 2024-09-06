from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 1

    config.seed = 1
    config.render = False
    config.mode = "train"
    config.control_type = "rudin2022"
    config.command_type = "random"
    config.command_sampling_type = "step_probability"
    config.initial_state_type = "default"
    config.reward_type = "rudin_own_var"
    config.termination_type = "rudin2022"
    config.domain_randomization_sampling_type = "none"
    config.domain_randomization_action_delay_type = "none"
    config.domain_randomization_mujoco_model_type = "none"
    config.domain_randomization_control_type = "none"
    config.domain_randomization_perturbation_type = "none"
    config.domain_randomization_perturbation_sampling_type = "none"
    config.observation_noise_type = "none"
    config.observation_dropout_type = "none"
    config.terrain_type = "plane"
    config.missing_value = 0.0
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20

    return config
