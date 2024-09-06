from copy import deepcopy
import numpy as np
import gymnasium as gym


class MultiSingleVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space, action_space, copy)

        self.active_env_id = 0


    def step_async(self, action):
        self.action = action[0]


    def reset_wait(self, **kwargs):
        observations, infos = super().reset_wait(**kwargs)

        return observations[self.active_env_id].reshape(1, -1), infos


    def step_wait(self):
        env = self.envs[self.active_env_id]
        infos = {}
        observation, reward, terminated, truncated, info = env.step(self.action)
        if terminated or truncated:
            old_observation, old_info = observation, info
            observation, info = env.reset()
            info["final_observation"] = old_observation
            info["final_info"] = old_info
        infos = self._add_info(infos, info, 0)
        observation = np.array([observation])
        reward = np.array([reward])
        terminated = np.array([terminated])
        truncated = np.array([truncated])

        return (
            deepcopy(observation) if self.copy else observation,
            np.copy(reward),
            np.copy(terminated),
            np.copy(truncated),
            infos,
        )
