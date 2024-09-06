class Rudin2022Termination:
    def __init__(self, env):
        self.env = env

    def should_terminate(self, obs):
        ids = self.env.collision_groups["trunk"]

        trunk_collision = False
        for con_i in range(0, self.env.data.ncon):
            con = self.env.data.contact[con_i]
            if con.geom1 in ids or con.geom2 in ids:
                trunk_collision = True
                break

        return trunk_collision
