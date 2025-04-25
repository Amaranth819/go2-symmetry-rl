from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())




from legged_gym.envs.go2_gait.go2_gait_env import Go2GaitEnv
from legged_gym.envs.go2_gait.go2_gait_cfg import Go2GaitCfg, Go2GaitCfgPPO
task_registry.register("go2_gait", Go2GaitEnv, Go2GaitCfg(), Go2GaitCfgPPO())