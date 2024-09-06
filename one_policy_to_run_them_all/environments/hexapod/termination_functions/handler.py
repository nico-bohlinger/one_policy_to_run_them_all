from one_policy_to_run_them_all.environments.hexapod.termination_functions.trunk_and_hip_collision import TrunkAndHipCollisionTermination


def get_termination_function(name, env, **kwargs):
    if name == "trunk_and_hip_collision":
        return TrunkAndHipCollisionTermination(env, **kwargs)
    else:
        raise NotImplementedError
