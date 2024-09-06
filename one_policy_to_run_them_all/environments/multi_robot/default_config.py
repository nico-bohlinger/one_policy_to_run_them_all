from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 1
    config.nr_eval_envs = 0
    config.train_robot_types = (
        "unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb", "badger", "bittle",
        "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie",
        "hexapod"
    )
    config.eval_robot_types = ()

    config.seed = 1
    config.async_skip_percentage = 0.0
    config.cycle_cpu_affinity = False
    config.render = False
    config.multi_render = False
    config.mode = "train"
    config.control_type = "rudin2022"
    config.command_type = "random"
    config.command_sampling_type = "step_probability"
    config.initial_state_type = "random"
    config.reward_type = "rudin_own_var"
    config.termination_type_a1 = "rudin2022"
    config.termination_type_go1 = "rudin2022"
    config.termination_type_go2 = "rudin2022"
    config.termination_type_h1 = "height"
    config.termination_type_g1 = "height"
    config.termination_type_bg = "rudin2022"
    config.termination_type_bglk = "rudin2022"
    config.termination_type_hbg = "rudin2022"
    config.termination_type_hx = "trunk_and_hip_collision"
    config.termination_type_tl = "height"
    config.termination_type_anyb = "rudin2022"
    config.termination_type_anyc = "rudin2022"
    config.termination_type_op3 = "height"
    config.termination_type_bkv0 = "rudin2022"
    config.termination_type_bkvb = "rudin2022"
    config.termination_type_cas = "height"
    config.termination_type_nao5 = "height"
    config.termination_type_bit = "rudin2022"
    config.termination_type_at = "height"
    config.termination_type_snk = "none"
    config.domain_randomization_sampling_type = "step_probability"
    config.domain_randomization_action_delay_type = "default"
    config.domain_randomization_mujoco_model_type = "default"
    config.domain_randomization_seen_robot_type = "default"
    config.domain_randomization_unseen_robot_type = "default"
    config.domain_randomization_perturbation_sampling_type = "step_probability"
    config.domain_randomization_perturbation_type = "default"
    config.observation_noise_type = "default"
    config.observation_dropout_type = "default"
    config.terrain_type = "plane"
    config.missing_value = 0.0
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20

    return config
