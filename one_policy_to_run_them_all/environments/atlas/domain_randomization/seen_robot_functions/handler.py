from one_policy_to_run_them_all.environments.atlas.domain_randomization.seen_robot_functions.default import DefaultDomainSeenRobotFunction
from one_policy_to_run_them_all.environments.atlas.domain_randomization.seen_robot_functions.none import NoneDomainSeenRobotFunction


def get_domain_randomization_seen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainSeenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDomainSeenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
