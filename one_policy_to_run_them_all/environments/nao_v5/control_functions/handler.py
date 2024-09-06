from one_policy_to_run_them_all.environments.nao_v5.control_functions.rudin2022 import Rudin2022Control


def get_control_function(name, env, **kwargs):
    if name == "rudin2022":
        return Rudin2022Control(env, **kwargs)
    else:
        raise NotImplementedError
