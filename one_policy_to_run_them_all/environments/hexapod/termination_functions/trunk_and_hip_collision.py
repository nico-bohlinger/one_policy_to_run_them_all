class TrunkAndHipCollisionTermination:
    def __init__(self, env):
        self.env = env

    def should_terminate(self, obs):
        hip_ids = self.env.collision_groups["hip"]
        trunk_ids = self.env.collision_groups["trunk"]
        ids = hip_ids.union(trunk_ids)

        collision = False
        for con_i in range(0, self.env.data.ncon):
            con = self.env.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                collision = True
                break

        return collision
