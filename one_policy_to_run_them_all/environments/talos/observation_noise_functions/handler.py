from one_policy_to_run_them_all.environments.talos.observation_noise_functions.default import DefaultObservationNoise
from one_policy_to_run_them_all.environments.talos.observation_noise_functions.none import NoneObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
