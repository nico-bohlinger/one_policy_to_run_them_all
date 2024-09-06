from one_policy_to_run_them_all.environments.cassie.termination_functions.rudin2022 import Rudin2022Termination
from one_policy_to_run_them_all.environments.cassie.termination_functions.height import HeightTermination


def get_termination_function(name, env, **kwargs):
    if name == "rudin2022":
        return Rudin2022Termination(env, **kwargs)
    elif name == "height":
        return HeightTermination(env, **kwargs)
    else:
        raise NotImplementedError
