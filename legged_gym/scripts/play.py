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
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
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

    # E_C_frcs = []
    # E_C_spds = []

    total_rews = torch.zeros(obs.size(0), device = obs.device, dtype = torch.float)
    total_steps = torch.zeros_like(total_rews, dtype = torch.int)
    for i in range(1*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # print(env._get_contact_forces([3, 7, 13, 17]))
        # print(torch.sum(torch.abs(env.torques), dim = -1))
        # print(torch.norm(env.foot_velocities, dim = -1))
        # print(torch.norm(env.rigid_body_state[:, env.feet_indices, 7:10], dim = -1))
        # print(env.E_C_spd[:, 0])

        total_rews += rews
        total_steps += 1
        env_ids = dones.nonzero(as_tuple=False).flatten()
        for i in env_ids:
            print(f'rewards = {total_rews[i].item():.3f}, steps = {total_steps[i].item()}')
            total_rews[i] = 0
            total_steps[i] = 0

        # E_C_frcs.append(env.E_C_frc.clone())
        # E_C_spds.append(env.E_C_spd.clone())

    env.save_record_video()

    # E_C_frcs = torch.cat(E_C_frcs).cpu().numpy()
    # start_idx = 0
    # end_idx = 150 # E_C_frcs.shape[0]
    # foot_seq = ['FL', 'FR', 'RL', 'RR']
    # import matplotlib.pyplot as plt
    # plt.figure(figsize = (24, 4))
    # for i, ft in enumerate(foot_seq):
    #     plt.plot(np.arange(start_idx, end_idx), E_C_frcs[start_idx:end_idx, i], label = ft)
    # plt.legend()
    # plt.savefig('E_C_frcs.png')
    # plt.close()



if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
