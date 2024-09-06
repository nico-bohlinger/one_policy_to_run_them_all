from one_policy_to_run_them_all.environments.hexapod.initial_state_functions.default import DefaultInitialState
from one_policy_to_run_them_all.environments.hexapod.initial_state_functions.random import RandomInitialState


def get_initial_state_function(name, env, **kwargs):
    if name == "default":
        return DefaultInitialState(env, **kwargs)
    elif name == "random":
        return RandomInitialState(env, **kwargs)
    else:
        raise NotImplementedError
