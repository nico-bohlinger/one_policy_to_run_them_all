from one_policy_to_run_them_all.environments.unitree_a1.environment import UnitreeA1
from one_policy_to_run_them_all.environments.unitree_go1.environment import UnitreeGo1
from one_policy_to_run_them_all.environments.unitree_go2.environment import UnitreeGo2
from one_policy_to_run_them_all.environments.unitree_h1.environment import UnitreeH1
from one_policy_to_run_them_all.environments.unitree_g1.environment import UnitreeG1
from one_policy_to_run_them_all.environments.badger.environment import Badger
from one_policy_to_run_them_all.environments.badger_locked.environment import BadgerLocked
from one_policy_to_run_them_all.environments.honey_badger.environment import HoneyBadger
from one_policy_to_run_them_all.environments.hexapod.environment import Hexapod
from one_policy_to_run_them_all.environments.talos.environment import Talos
from one_policy_to_run_them_all.environments.anymal_b.environment import AnymalB
from one_policy_to_run_them_all.environments.anymal_c.environment import AnymalC
from one_policy_to_run_them_all.environments.robotis_op3.environment import RobotisOP3
from one_policy_to_run_them_all.environments.barkour_v0.environment import BarkourV0
from one_policy_to_run_them_all.environments.barkour_vb.environment import BarkourVB
from one_policy_to_run_them_all.environments.cassie.environment import Cassie
from one_policy_to_run_them_all.environments.nao_v5.environment import NaoV5
from one_policy_to_run_them_all.environments.bittle.environment import Bittle
from one_policy_to_run_them_all.environments.atlas.environment import Atlas
from one_policy_to_run_them_all.environments.sea_snake.environment import SEASnake


class Robot:
    def __init__(self, cls, long_name, short_name):
        self.cls = cls
        self.long_name = long_name
        self.short_name = short_name

      
ROBOTS = [
    Robot(UnitreeA1, UnitreeA1.LONG_NAME, UnitreeA1.SHORT_NAME),
    Robot(UnitreeGo1, UnitreeGo1.LONG_NAME, UnitreeGo1.SHORT_NAME),
    Robot(UnitreeGo2, UnitreeGo2.LONG_NAME, UnitreeGo2.SHORT_NAME),
    Robot(UnitreeH1, UnitreeH1.LONG_NAME, UnitreeH1.SHORT_NAME),
    Robot(UnitreeG1, UnitreeG1.LONG_NAME, UnitreeG1.SHORT_NAME),
    Robot(Badger, Badger.LONG_NAME, Badger.SHORT_NAME),
    Robot(BadgerLocked, BadgerLocked.LONG_NAME, BadgerLocked.SHORT_NAME),
    Robot(HoneyBadger, HoneyBadger.LONG_NAME, HoneyBadger.SHORT_NAME),
    Robot(Hexapod, Hexapod.LONG_NAME, Hexapod.SHORT_NAME),
    Robot(Talos, Talos.LONG_NAME, Talos.SHORT_NAME),
    Robot(AnymalB, AnymalB.LONG_NAME, AnymalB.SHORT_NAME),
    Robot(AnymalC, AnymalC.LONG_NAME, AnymalC.SHORT_NAME),
    Robot(RobotisOP3, RobotisOP3.LONG_NAME, RobotisOP3.SHORT_NAME),
    Robot(BarkourV0, BarkourV0.LONG_NAME, BarkourV0.SHORT_NAME),
    Robot(BarkourVB, BarkourVB.LONG_NAME, BarkourVB.SHORT_NAME),
    Robot(Cassie, Cassie.LONG_NAME, Cassie.SHORT_NAME),
    Robot(NaoV5, NaoV5.LONG_NAME, NaoV5.SHORT_NAME),
    Robot(Bittle, Bittle.LONG_NAME, Bittle.SHORT_NAME),
    Robot(Atlas, Atlas.LONG_NAME, Atlas.SHORT_NAME),
    Robot(SEASnake, SEASnake.LONG_NAME, SEASnake.SHORT_NAME),
]
