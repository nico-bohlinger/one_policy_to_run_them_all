import logging
import gymnasium as gym

from one_policy_to_run_them_all.environments.multi_robot.environment import TrainAndEvalEnvironment, DummyEnvironment
from one_policy_to_run_them_all.environments.multi_robot.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from one_policy_to_run_them_all.environments.multi_robot.multi_single_vectorized_wrapper import MultiSingleVectorEnv
from one_policy_to_run_them_all.environments.multi_robot.wrappers import RLXInfo, MultiRenderWrapper, RecordEpisodeStatistics
from one_policy_to_run_them_all.environments.multi_robot.general_properties import GeneralProperties
from one_policy_to_run_them_all.environments.multi_robot.robot_helper import ROBOTS
from one_policy_to_run_them_all.environments.multi_robot.cpu_gpu_testing import get_global_cpu_ids, get_fastest_cpu_for_gpu_connection

rlx_logger = logging.getLogger("rl_x")


def create_env(config):
    def make_env(
            env_class, seed,
            max_observation_size=-1, max_action_size=-1,
            nr_envs_per_type=1, mode="train", purpose_initial_check=False,
            env_cpu_id=None
        ):
        def thunk():
            for robot in ROBOTS:
                if robot.cls == env_class:
                    short_name = robot.short_name
                    termination_type = getattr(config.environment, f"termination_type_{short_name}")
            env = env_class(
                seed=seed,
                render=config.environment.render and not purpose_initial_check,
                mode=mode,
                control_type=config.environment.control_type,
                command_type=config.environment.command_type,
                command_sampling_type=config.environment.command_sampling_type,
                initial_state_type=config.environment.initial_state_type,
                reward_type=config.environment.reward_type,
                termination_type=termination_type,
                domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
                domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
                domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
                domain_randomization_seen_robot_type=config.environment.domain_randomization_seen_robot_type,
                domain_randomization_unseen_robot_type=config.environment.domain_randomization_unseen_robot_type,
                domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
                domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
                observation_noise_type=config.environment.observation_noise_type,
                observation_dropout_type=config.environment.observation_dropout_type,
                terrain_type=config.environment.terrain_type,
                missing_value=config.environment.missing_value,
                add_goal_arrow=config.environment.add_goal_arrow,
                timestep=config.environment.timestep,
                episode_length_in_seconds=config.environment.episode_length_in_seconds,
                total_nr_envs=config.environment.nr_envs,
                multi_robot_max_observation_size=max_observation_size,
                multi_robot_max_action_size=max_action_size,
                nr_envs_per_type=nr_envs_per_type,
                cpu_id=env_cpu_id
            )
            env = RecordEpisodeStatistics(env, short_name)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk


    def get_env_class_from_robot_type(robot_type):
        for robot in ROBOTS:
            if robot.long_name == robot_type:
                return robot.cls


    if config.environment.render and config.environment.multi_render:
        raise ValueError("Render and multi_render cannot be true at the same time")

    nr_envs_per_type = config.environment.nr_envs // len(config.environment.train_robot_types)
    if nr_envs_per_type * len(config.environment.train_robot_types) != config.environment.nr_envs:
        raise ValueError("Number of train environments must be divisible by number of robot types")
    if config.environment.nr_eval_envs != 0:
        nr_eval_envs_per_type = config.environment.nr_eval_envs / len(config.environment.eval_robot_types)
        if nr_eval_envs_per_type != 1.0:
            raise ValueError("The number of evaluation environments must be 1 per robot type")
        else:
            nr_eval_envs_per_type = 1
    
    global_cpu_ids = None
    if config.environment.cycle_cpu_affinity or config.algorithm.determine_fastest_cpu_for_gpu:
        global_cpu_ids = get_global_cpu_ids()
        rlx_logger.info(f"Global CPU IDs: {global_cpu_ids}")

    fastest_cpu_id = None
    if config.algorithm.determine_fastest_cpu_for_gpu:
        fastest_cpu_id = get_fastest_cpu_for_gpu_connection(global_cpu_ids)

    env_cpu_ids = None
    if config.environment.cycle_cpu_affinity:
        usable_cpu_ids_for_envs = global_cpu_ids.copy()
        if fastest_cpu_id is not None:
            usable_cpu_ids_for_envs.remove(fastest_cpu_id)
        env_cpu_ids = []
        for i in range(config.environment.nr_envs):
            env_cpu_ids.append(usable_cpu_ids_for_envs[i % len(usable_cpu_ids_for_envs)])
    
    observation_sizes = []
    action_sizes = []
    for robot_type in config.environment.train_robot_types + config.environment.eval_robot_types:
        env_class = get_env_class_from_robot_type(robot_type)
        env = make_env(env_class, config.environment.seed, purpose_initial_check=True)()
        observation_sizes.append(env.observation_space.shape[0])
        action_sizes.append(env.action_space.shape[0])
        env.close()
        del env
    max_observation_size = max(observation_sizes)
    max_action_size = max(action_sizes)
    
    env_list = []
    env_id = 0
    for robot_type in config.environment.train_robot_types:
        env_class = get_env_class_from_robot_type(robot_type)
        for i in range(nr_envs_per_type):
            env_cpu_id = None if env_cpu_ids is None else env_cpu_ids[env_id]
            env_list.append(make_env(
                env_class, config.environment.seed + i,
                max_observation_size, max_action_size,
                nr_envs_per_type, mode=config.environment.mode,
                env_cpu_id=env_cpu_id
            ))
            env_id += 1
    if config.environment.nr_envs == 1 or config.environment.multi_render:
        env = MultiSingleVectorEnv(env_list)
        if config.environment.multi_render:
            env = MultiRenderWrapper(env, config.environment.nr_envs)
    else:
        env = AsyncVectorEnvWithSkipping(env_list, config.environment.async_skip_percentage)
    env = RLXInfo(env, fastest_cpu_id)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    eval_env_list = []
    for robot_type in config.environment.eval_robot_types:
        env_class = get_env_class_from_robot_type(robot_type)
        for i in range(nr_eval_envs_per_type):
            eval_env_list.append(make_env(
                env_class, config.environment.seed + i,
                max_observation_size, max_action_size,
                nr_eval_envs_per_type, mode="eval"
            ))
    if config.environment.nr_eval_envs == 0:
        eval_env = DummyEnvironment()
    else:
        if config.environment.nr_eval_envs == 1:
            eval_env = gym.vector.SyncVectorEnv(eval_env_list)
        else:
            eval_env = AsyncVectorEnvWithSkipping(eval_env_list, config.environment.async_skip_percentage)
        eval_env = RLXInfo(eval_env)
        eval_env.general_properties = GeneralProperties

        eval_env.reset(seed=config.environment.seed)

    env = TrainAndEvalEnvironment(env, eval_env)

    return env
