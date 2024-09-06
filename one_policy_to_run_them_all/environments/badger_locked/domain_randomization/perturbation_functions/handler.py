from one_policy_to_run_them_all.environments.badger_locked.domain_randomization.perturbation_functions.default import DefaultDomainPerturbation
from one_policy_to_run_them_all.environments.badger_locked.domain_randomization.perturbation_functions.none import NoneDomainPerturbation


def get_domain_randomization_perturbation_function(name, env, **kwargs):
    if name == "default":
        return DefaultDomainPerturbation(env, **kwargs)
    elif name == "none":
        return NoneDomainPerturbation(env, **kwargs)
    else:
        raise NotImplementedError
