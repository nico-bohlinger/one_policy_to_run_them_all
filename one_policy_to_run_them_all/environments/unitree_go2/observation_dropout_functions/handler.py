from one_policy_to_run_them_all.environments.unitree_go2.observation_dropout_functions.default import DefaultObservationDropout
from one_policy_to_run_them_all.environments.unitree_go2.observation_dropout_functions.none import NoneObservationDropout


def get_observation_dropout_function(name, env, **kwargs):
    if name == "default":
        return DefaultObservationDropout(env, **kwargs)
    elif name == "none":
        return NoneObservationDropout(env, **kwargs)
    else:
        raise NotImplementedError
