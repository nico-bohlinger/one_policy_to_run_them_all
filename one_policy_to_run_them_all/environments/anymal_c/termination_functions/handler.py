from one_policy_to_run_them_all.environments.anymal_c.termination_functions.rudin2022 import Rudin2022Termination


def get_termination_function(name, env, **kwargs):
    if name == "rudin2022":
        return Rudin2022Termination(env, **kwargs)
    else:
        raise NotImplementedError
