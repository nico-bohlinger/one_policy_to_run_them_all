from one_policy_to_run_them_all.environments.unitree_g1.control_functions.rudin2022 import Rudin2022Control


def get_control_function(name, env, **kwargs):
    if name == "rudin2022":
        return Rudin2022Control(env, **kwargs)
    else:
        raise NotImplementedError
