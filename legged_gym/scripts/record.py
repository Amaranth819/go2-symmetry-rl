import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch



def play(args):
    args.record_video = True
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    env._create_camera(0, p = [0, -1, 0], axis = [0, 0, 1], angle = 90.0, follow = 'FOLLOW_POSITION')

    for idx, gait_name in enumerate(['trot', 'bound', 'halfbound', 'gallop']):
        for fwd_vel_cmd in [-1, 1]:
            obs, _ = env.reset()
            env._reset_foot_periodicity_manually( 
                fwd_vel_cmd = fwd_vel_cmd, 
                lateral_vel_cmd = 0, 
                yaw_vel_cmd = 0, 
                foot_thetas = env.init_foot_thetas[idx], 
                duty_factor = 0.37, 
                gait_period = 0.45, 
                calculate_from_slip_model = True
            )
            for i in range(1*int(env.max_episode_length)):
                actions = policy(obs.detach())
                obs, _, rews, dones, infos = env.step(actions.detach())
                reset_env_inds = dones.nonzero(as_tuple=False).flatten()

                if len(reset_env_inds) > 0:
                    obs, _ = env.reset()
                    env._reset_foot_periodicity_manually( 
                        fwd_vel_cmd = fwd_vel_cmd, 
                        lateral_vel_cmd = 0, 
                        yaw_vel_cmd = 0, 
                        foot_thetas = env.init_foot_thetas[idx], 
                        duty_factor = 0.37, 
                        gait_period = 0.45, 
                        calculate_from_slip_model = True
                    )

            env.save_record_video(name = f'{gait_name}_fwdvelcmd={fwd_vel_cmd}m', ext = 'gif')



if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
