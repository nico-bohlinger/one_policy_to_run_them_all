from one_policy_to_run_them_all.environments.anymal_c.terrain_functions.plane import PlaneTerrainGeneration


def get_terrain_function(name, env, **kwargs):
    if name == "plane":
        return PlaneTerrainGeneration(env, **kwargs)
    else:
        raise NotImplementedError
