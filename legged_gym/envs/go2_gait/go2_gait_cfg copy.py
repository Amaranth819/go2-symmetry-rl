from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


class Go2GaitCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 48 # if not base velocities # 54
        episode_length_s = 20
        num_envs = 2048
    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards():
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        # class scales( LeggedRobotCfg.rewards.scales ):
        #     torques = -0.0002
        #     dof_pos_limits = -10.0

        class scales():
            alive_bonus = 1.0
            foot_periodicity = 0.25
            morphological_symmetry = 0.1
            cmd = 0.2
            smoothness = 0.1
            pitching = 0.05
            # collision = 0.1

    
    class foot_periodicity:
        gait_period = 0.45 # 0.45
        duty_factor = 0.37 # 0.37
        # Order: ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'] front left, front right, rear left, rear right
        # init_foot_thetas = [-0.10, 0.6, 0.10, 0.4] # Galloping
        # init_foot_thetas = [0.1, 0.5, -0.1, 0.5] # half-bounding
        # init_foot_thetas = [0.0, 0.0, 0.5, 0.5] # bounding
        init_foot_thetas = [0, 0.5, 0.5, 0] # Trotting
        kappa = 16
        c_swing_frc = -1
        c_swing_spd = 0
        c_stance_frc = 0
        c_stance_spd = -1

        add_noise = False
        noise_scale = 0.01
        noise_level = 10


    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.3, 2.0] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]



# class Go2TrotCfg(Go2GaitCfg):
#     class foot_periodicity(Go2GaitCfg.foot_periodicity):
#         init_foot_thetas = [0, 0.5, 0.5, 0] # Trotting


# class Go2GallopCfg(Go2GaitCfg):
#     class foot_periodicity(Go2GaitCfg.foot_periodicity):
#         init_foot_thetas = [-0.10, 0.6, 0.10, 0.4] # Galloping


# class Go2HalfBoundCfg(Go2GaitCfg):
#     class foot_periodicity(Go2GaitCfg.foot_periodicity):
#         init_foot_thetas = [0.1, 0.5, -0.1, 0.5] # half-bounding


# class Go2BoundCfg(Go2GaitCfg):
#     class foot_periodicity(Go2GaitCfg.foot_periodicity):
#         init_foot_thetas = [0.0, 0.5, 0.0, 0.5] # bounding



# class Go2GaitCfgPPO(LeggedRobotCfgPPO):
#     use_wandb = True
#     class runner(LeggedRobotCfgPPO.runner):
#         experiment_name = "gait_go2"
#         policy_class_name = 'ActorCritic'
#         algorithm_class_name = 'PPO'
#         max_iterations = 500
#         save_interval = 250
#     class policy(LeggedRobotCfgPPO.policy):
#         init_noise_std = 1.0
#         actor_hidden_dims = [256, 256, 256]
#         critic_hidden_dims = [256, 256, 256]
#         activation = 'elu'
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         learning_rate = 0.00025


class Go2GaitCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrent'
        run_name = ''
        experiment_name = 'gait_go2'
        save_interval = 100
        max_iterations = 500
    # class policy(LeggedRobotCfgPPO.policy):
    #     rnn_type = 'lstm'
    #     rnn_hidden_size = 512
    #     rnn_num_layers = 1


# class Go2GaitCfgPPOAug(LeggedRobotCfgPPO):
#     use_wandb = True
#     class runner(LeggedRobotCfgPPO.runner):
#         experiment_name = "gait_go2_aug"
#         policy_class_name = 'ActorCritic'
#         algorithm_class_name = 'PPOAugmented'
#         max_iterations = 20000
#         save_interval = 500
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [256, 256, 256]
#         critic_hidden_dims = [256, 256, 256]
#         activation = 'relu'
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         learning_rate = 0.00025


# class Go2GaitCfgPPOEMLP(LeggedRobotCfgPPO):
#     use_wandb = True
#     class runner(LeggedRobotCfgPPO.runner):
#         experiment_name = "gait_go2_emlp"
#         policy_class_name = 'ActorCriticSymm'
#         algorithm_class_name = 'PPO'
#         max_iterations = 20000
#         save_interval = 500
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [256, 256, 256]
#         critic_hidden_dims = [256, 256, 256]
#         activation = 'relu'
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         learning_rate = 0.00025