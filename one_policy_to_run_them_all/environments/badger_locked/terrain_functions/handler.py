from one_policy_to_run_them_all.environments.badger_locked.terrain_functions.plane import PlaneTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
