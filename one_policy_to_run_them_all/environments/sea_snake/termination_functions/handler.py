from one_policy_to_run_them_all.environments.sea_snake.termination_functions.none import NoneTerminationFunction


def get_termination_function(name, env, **kwargs):
    if name == "none":
        return NoneTerminationFunction(env, **kwargs)
    else:
        raise NotImplementedError
