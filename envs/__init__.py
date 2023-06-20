from .skt_franka_grasping_env import FrankaGraspingEnv
from .HSR_env import HSR
from .Anymal_env import Anymal

task_map = {
    "Franka": FrankaGraspingEnv,
    "HSR": HSR,
    "Anymal": Anymal,
}