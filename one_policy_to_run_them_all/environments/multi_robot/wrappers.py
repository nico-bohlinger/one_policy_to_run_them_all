import os
import time
import gymnasium as gym
import pygame

from one_policy_to_run_them_all.environments.multi_robot.viewer import MujocoViewer


class RLXInfo(gym.Wrapper):
    def __init__(self, env, fastest_cpu_id=None):
        super(RLXInfo, self).__init__(env)
        self.fastest_cpu_id = fastest_cpu_id
    

    def get_logging_info_dict(self, info):
        all_keys = list(info.keys())
        keys_to_remove = ["final_observation", "final_info", "t"]

        logging_info = {
            key: info[key][info["_" + key]].tolist()
                for key in all_keys if key not in keys_to_remove and not key.startswith("_") and len(info[key][info["_" + key]]) > 0
        }
        if "final_info" in info:
            for done, final_info in zip(info["_final_info"], info["final_info"]):
                if done:   
                    for key, info_value in final_info.items():
                        if key not in keys_to_remove:
                            logging_info.setdefault(key, []).append(info_value)

        return logging_info
    

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]
    

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]
    

class MultiRenderWrapper(gym.Wrapper):
    def __init__(self, env, nr_envs, file_name="multi_render.txt"):
        super(MultiRenderWrapper, self).__init__(env)

        self.nr_envs = nr_envs
        self.file_name = file_name

        pygame.init()
        pygame.joystick.init()
        self.joystick_present = False
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.joystick_present = True
            self.last_joystick_change = time.time()

        self.env_to_render = 0
        self.env_to_render = self.get_env_to_render()
        model = env.call("model")[self.env_to_render]
        dt = env.call("dt")[self.env_to_render]
        self.viewer = MujocoViewer(model, dt)


    def get_env_to_render(self):
        if self.joystick_present:
            pygame.event.pump()
            current_time = time.time()
            if current_time - self.last_joystick_change < 0.3:
                return self.env_to_render
            if self.joystick.get_button(5):
                self.last_joystick_change = current_time
                return (self.env_to_render + 1) % self.nr_envs
            elif self.joystick.get_button(4):
                self.last_joystick_change = current_time
                return (self.env_to_render - 1) % self.nr_envs
            return self.env_to_render
        elif os.path.isfile(self.file_name):
            with open(self.file_name, "r") as f:
                env_to_render = f.read()
            return min(int(env_to_render), self.nr_envs-1)
        return 0


    def step(self, action):
        data = super().step(action)
        env_to_render = self.get_env_to_render()
        if self.env_to_render != env_to_render:
            self.env.active_env_id = env_to_render
            model = self.env.call("model")[env_to_render]
            self.env_to_render = env_to_render
            self.viewer.load_new_model(model)
        self.viewer.render(self.env.call("data")[env_to_render])
        return data
    

    def close(self):
        self.viewer.close()
        pygame.quit()
        return super().close()


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, robot_type):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.robot_type = robot_type
        self.episode_returns = None
        self.episode_lengths = None


    def reset(self, **kwargs):
        self.episode_return = 0.0
        self.episode_length = 0.0
        return self.env.reset(**kwargs)


    def step(self, action):
        observation, reward, termination, truncation, info = super(RecordEpisodeStatistics, self).step(action)
        done = termination | truncation
        self.episode_return += reward
        self.episode_length += 1
        if done:
            info["episode_return_" + self.robot_type] = self.episode_return
            info["episode_length_" + self.robot_type] = self.episode_length
        return (observation, reward, termination, truncation, info)
