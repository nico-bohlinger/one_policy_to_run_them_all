from one_policy_to_run_them_all.environments.unitree_go1.domain_randomization.unseen_robot_functions.default import DefaultDomainUnseenRobotFunction
from one_policy_to_run_them_all.environments.unitree_go1.domain_randomization.unseen_robot_functions.none import NoneDomainUnseenRobotFunction


def get_domain_randomization_unseen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainUnseenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDomainUnseenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
