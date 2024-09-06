from one_policy_to_run_them_all.environments.robotis_op3.sampling_functions.step_probability import StepProbabilitySampling
from one_policy_to_run_them_all.environments.robotis_op3.sampling_functions.none import NoneSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    else:
        raise NotImplementedError
