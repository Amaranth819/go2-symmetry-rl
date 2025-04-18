from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2_gait.go2_gait_cfg import Go2GaitCfg
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.utils.math import quat_apply_yaw
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, quat_conjugate, quat_mul
from scipy.stats import vonmises_line
import torch
from torch import Tensor
from collections import deque
import numpy as np
import os
import torch
import sys
# from gym.spaces import Box
from legged_gym.envs.go2_gait.von_mise import E_periodic_property
from collections import defaultdict, deque


CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
MAX_VIDEO_LENGTH = 1000
VIDEO_FPS = 30


class Go2GaitEnv(LeggedRobot):
    def __init__(self, cfg : Go2GaitCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # if self.headless:
        #     camera_position = gymapi.Vec3(-0.5 + self.env_origins[self.cam_env_id, 0], 0.5 + self.env_origins[self.cam_env_id, 1], 1 + self.env_origins[self.cam_env_id, 2])
        #     camera_target = gymapi.Vec3(0.5 + self.env_origins[self.cam_env_id, 0], 0. + self.env_origins[self.cam_env_id, 1], 0.5 + self.env_origins[self.cam_env_id, 2])
        #     self.gym.set_camera_location(self.camera_handle, self.envs[self.cam_env_id], camera_position, camera_target)

        self.foot_periodicity_cfg = self.cfg.foot_periodicity


        self.cameras = {}
        self.camera_tensors = []
        self.camera_frames = defaultdict(lambda: deque(maxlen = MAX_VIDEO_LENGTH))

        self.episode_times = torch.zeros_like(self.episode_length_buf, dtype = torch.float)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt # 1
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}



    def _init_foot_periodicity_buffer(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        self.init_foot_thetas = torch.as_tensor(self.cfg.foot_periodicity.init_foot_thetas, dtype = torch.float, device = self.device).unsqueeze(0)
        self.duty_factors = torch.ones_like(self.episode_length_buf) * foot_periodicity_cfg.duty_factor # Duty factor (the ratio of the stance phase) of the current gait
        self.kappa = torch.ones_like(self.duty_factors) * foot_periodicity_cfg.kappa # Kappa
        self.foot_thetas = torch.as_tensor(foot_periodicity_cfg.init_foot_thetas, dtype = torch.float, device = self.device).unsqueeze(0).repeat(self.num_envs, 1) # Clock input shift
        self.gait_period_steps = torch.ones_like(self.episode_length_buf) * foot_periodicity_cfg.gait_period / self.sim_params.dt
        self.gait_periods = torch.ones_like(self.episode_length_buf) * foot_periodicity_cfg.gait_period

        # Foot periodicity information to compute different rewards.
        self.E_C_frc = torch.zeros(size = (self.num_envs, len(self.feet_indices)), dtype = torch.float32, device = self.device)
        self.E_C_spd = torch.zeros(size = (self.num_envs, len(self.feet_indices)), dtype = torch.float32, device = self.device)


    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot_periodicity_buffer()
        self.last_torques = torch.zeros_like(self.torques)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)


    def _get_contact_forces(self, indices):
        return torch.norm(self.contact_forces[:, indices, :], dim = -1)
    

    def _get_foot_phis(self):
        phi = self._get_periodicity_ratio()
        sign_indicator = torch.ones_like(self.commands[..., 0:1])
        sign = torch.where(self.commands[..., 0:1] >= 0, sign_indicator, -sign_indicator)
        return torch.abs(((phi.unsqueeze(-1) + self.foot_thetas) * sign) % sign)
    

    def _get_periodicity_ratio(self):
        # return self.episode_length_buf / self.gait_period_steps
        return self.episode_times / self.gait_periods

    
    def _compute_E_C(self):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        foot_phis = self._get_foot_phis().cpu().numpy()
        duty_factor = self.duty_factors.cpu().numpy()[..., None] # [num_envs, 1]
        kappa = self.kappa.cpu().numpy()[..., None] # [num_envs, 1]

        E_C_frc = E_periodic_property(foot_phis, duty_factor, kappa, foot_periodicity_cfg.c_swing_frc, foot_periodicity_cfg.c_stance_frc)
        E_C_frc = torch.from_numpy(E_C_frc).to(self.device)
        E_C_spd = E_periodic_property(foot_phis, duty_factor, kappa, foot_periodicity_cfg.c_swing_spd, foot_periodicity_cfg.c_stance_spd)
        E_C_spd = torch.from_numpy(E_C_spd).to(self.device)

        return E_C_frc, E_C_spd
    

    def _reset_foot_periodicity(self, env_ids, calculate_from_slip_model = True, add_noise = False):
        foot_periodicity_cfg = self.cfg.foot_periodicity
        self.kappa[env_ids] = foot_periodicity_cfg.kappa
        self.foot_thetas[env_ids, :] = self.init_foot_thetas

        if calculate_from_slip_model:
            # Reset the parameters to the values calculated from the SLIP model.
            self.duty_factors[env_ids] = self._compute_duty_factor_from_cmd_forward_linvel(self.commands[env_ids, 0])
            self.gait_periods[env_ids] = self._compute_period_from_cmd_forward_linvel(self.commands[env_ids, 0])
            self.gait_period_steps[env_ids] = self._compute_period_from_cmd_forward_linvel(self.commands[env_ids, 0]) / self.dt
        else:
            # Reset the parameters to the values in the configuration file.
            self.duty_factors[env_ids] = foot_periodicity_cfg.duty_factor
            self.gait_periods[env_ids] = foot_periodicity_cfg.gait_period
            self.gait_period_steps[env_ids] = foot_periodicity_cfg.gait_period / self.dt

        # Randomization to foot thetas
        if add_noise:
            noise_scale = self.foot_periodicity_cfg.noise_scale
            noise_level = self.foot_periodicity_cfg.noise_level
            self.foot_thetas[env_ids, :] += torch.randint(low = -noise_level, high = noise_level, size = (len(env_ids), len(self.feet_indices)), device = self.device) * noise_scale

        # print(self.foot_thetas)

        self.E_C_frc, self.E_C_spd = self._compute_E_C()


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_torques[env_ids] = 0
        self.episode_times[env_ids] = 0
        self._reset_foot_periodicity(env_ids, calculate_from_slip_model = True, add_noise = self.foot_periodicity_cfg.add_noise)


    '''
        Generate the gait parameters from the command linear velocity. 
    '''
    def _compute_period_from_cmd_forward_linvel(self, cmd_forward_linvel):
        abs_cmd_forward_linvel = torch.abs(cmd_forward_linvel)
        random_scale = torch.rand_like(cmd_forward_linvel) * 2 - 1.0
        return 0.2576 * torch.exp(-0.9829 * abs_cmd_forward_linvel) * (1 + random_scale * abs_cmd_forward_linvel * 0.25)
    

    def _compute_duty_factor_from_cmd_forward_linvel(self, cmd_forward_linvel):
        abs_cmd_forward_linvel = torch.abs(cmd_forward_linvel)
        random_scale = torch.rand_like(cmd_forward_linvel) * 2 - 1.0
        return 0.5588 * torch.exp(-0.6875 * abs_cmd_forward_linvel) * (1 + random_scale * abs_cmd_forward_linvel * 0.25)
    

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

        noise_vec[:3] = noise_scales.gravity * noise_level
        noise_vec[3:6] = 0. # commands
        noise_vec[6:6 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6 + self.num_actions:6 + 2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6 + 2*self.num_actions:6 + 3*self.num_actions] = 0. # previous actions

        noise_vec[-6:-3] = 0. # foot_phis_sin
        noise_vec[-3:] = 0. # phase ratios

        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        return noise_vec
    

    def post_physics_step(self):
        self.episode_times += self.dt
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.E_C_frc, self.E_C_spd = self._compute_E_C()
        super().post_physics_step()
        self.last_torques[:] = self.torques[:]
        self._render_cameras()
    

    def compute_observations(self):
        # num_feet: 4
        foot_phis_sin = torch.sin(2 * torch.pi * self._get_foot_phis())
                              
        # phase ratios: 2
        phase_ratios = torch.stack([
            1.0 - self.duty_factors, # swing phase ratio
            self.duty_factors, # stance phase ratio
        ], dim = -1)

        # Do not use self.obs_buf[:] = ... !!!!!!!!!!
        self.obs_buf = torch.cat([
            # self.base_lin_vel * self.obs_scales.lin_vel,
            # self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            foot_phis_sin,    
            phase_ratios
        ], dim = -1)    

        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def check_termination(self):
        """
            Check if the environments need to be reset.
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.reset_buf |= torch.logical_or(torch.abs(self.base_pos[..., 2]) < 0.15, torch.abs(self.base_pos[..., 2]) > 0.45)
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf


    def _reward_alive_bonus(self):
        return 1.0
    

    def _reward_foot_periodicity(self):
        # ['base', 
        # 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 
        # 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 
        # 'Head_upper', 'Head_lower', 
        # 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 
        # 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
        # foot_frcs = self._get_contact_forces(self.feet_indices)
        foot_frcs = self._get_contact_forces([3, 7, 13, 17]) # Use calf indices rather than foot indices 
        R_foot_frcs = torch.sum((1 - torch.exp(-0.02 * foot_frcs)) * self.E_C_frc, dim = -1)
        foot_spds = torch.norm(self.rigid_body_state[:, self.feet_indices, 7:10], dim = -1)
        R_foot_spds = torch.sum((1 - torch.exp(-4 * foot_spds)) * self.E_C_spd, dim = -1)
        return R_foot_frcs + R_foot_spds
    

    def _reward_cmd(self):
        x_lin_vel_err = torch.abs(self.commands[..., 0] - self.base_lin_vel[..., 0])
        x_lin_vel_cmd_rew = 1 - torch.exp(-2 * x_lin_vel_err)

        y_lin_vel_err = torch.abs(self.commands[..., 1] - self.base_lin_vel[..., 1])
        y_lin_vel_cmd_rew = 1 - torch.exp(-2 * y_lin_vel_err)

        z_ang_vel_err = torch.abs(self.commands[..., 2] - self.base_ang_vel[..., 2])
        z_ang_vel_cmd_rew = 1 - torch.exp(-3 * z_ang_vel_err)

        return -(x_lin_vel_cmd_rew + y_lin_vel_cmd_rew + z_ang_vel_cmd_rew)
    

    def _reward_smoothness(self):
        # torque = torch.sum(torch.abs(self.torques), dim = -1)
        # R_torque = 1 - torch.exp(-0.02 * torque)

        torque_diff = torch.sum(torch.abs(self.torques - self.last_torques), dim = -1)
        R_torque = 1 - torch.exp(-0.1 * torque_diff)

        # action_diff = torch.sum(torch.abs(self.actions - self.last_actions), dim = -1)
        # R_action_diff = 1 - torch.exp(-2 * action_diff)

        return -(R_torque)
    

    def _reward_pitching(self):
        # z_lin_vel = torch.abs(self.base_lin_vel[..., 2]) # penalize lin vel z 
        # R_pitching_vel = 1 - torch.exp(-2 * z_lin_vel)
        # return -R_pitching_vel

        # base_height_diff = torch.clamp(0.25 - self.base_pos[..., 2], min = 0)
        # R_base_height_diff = 1 - torch.exp(-4 * base_height_diff)
        # return -R_base_height_diff

        base_pitch_angle = torch.abs(self.rpy[..., 1])
        R_pitching_angle = 1 - torch.exp(-3 * base_pitch_angle)
        return -(R_pitching_angle)
    

    def _reward_collision(self):
        num_collision = torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        R_collision = 1 - torch.exp(-0.5 * num_collision)
        return -(R_collision)


    def _reward_morphological_symmetry(self):
        '''
            foot theta orders: ['FrontLeft', 'FrontRight', 'RearLeft', 'RearRight']
            ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        '''
        # foot_idx, (hip, thigh, calf)
        temp_dict = {
            'FL' : (0, [0, 1, 2]),
            'FR' : (1, [3, 4, 5]),
            'RL' : (2, [6, 7, 8]),
            'RR' : (3, [9, 10, 11])
        }
        # jpos_same_direction = torch.as_tensor([-1, 1, 1], dtype = torch.float, device = self.device).unsqueeze(0)
        threshold = 0.01

        def morph_sym_error(tag1, tag2):
            tag1_foot_idx, tag1_foot_joint_indices = temp_dict[tag1]
            tag2_foot_idx, tag2_foot_joint_indices = temp_dict[tag2]

            jpos_same_direction = []
            # Hip joint positions are opposite if one is left and another is right
            if tag1[-1] != tag2[-1]:
                jpos_same_direction = [-1, 1, 1]
            else:
                jpos_same_direction = [1, 1, 1]
            jpos_same_direction = torch.as_tensor(jpos_same_direction, dtype = torch.float, device = self.device).unsqueeze(0)

            theta_consistent = torch.abs(self.foot_thetas[..., tag1_foot_idx] - self.foot_thetas[..., tag2_foot_idx]) <= threshold # [B,]
            error = torch.sum(torch.abs(self.dof_pos[..., tag1_foot_joint_indices] - jpos_same_direction * self.dof_pos[..., tag2_foot_joint_indices]) * theta_consistent.unsqueeze(-1), dim = -1)
            return error

        error_sum = torch.zeros_like(self.episode_length_buf, dtype = torch.float)
        # Left-right symmetry
        error_sum += morph_sym_error('FL', 'FR')
        error_sum += morph_sym_error('RL', 'RR')
        # Front-rear symmetry
        error_sum += morph_sym_error('FL', 'RL')
        error_sum += morph_sym_error('FR', 'RR')
        # Diagonal symmetry
        error_sum += morph_sym_error('FL', 'RR')
        error_sum += morph_sym_error('RL', 'FR')

        R_morph = 1 - torch.exp(-5 * error_sum)
        return -R_morph
    



    '''
        Camera
    '''
    def _set_viewer_camera(self, pos, lookat, ref_env_idx : int = -1):
        """
            Set the camera position and direction.
            Input:
                pos: (x, y, z)
                lookat: (x, y, z)
        """
        assert len(pos) == 3 and len(lookat) == 3
        cam_pos, cam_target = gymapi.Vec3(*pos), gymapi.Vec3(*lookat)
        if ref_env_idx >= 0:
            env_handle = self.envs[ref_env_idx]
            # Set the camera to track a certain environment.
            ref_env_base_pos = gymapi.Vec3(*self.root_states[ref_env_idx, :3])
            cam_pos = cam_pos + ref_env_base_pos
            cam_target = cam_target + ref_env_base_pos
        else:
            env_handle = None
        self.gym.viewer_camera_look_at(self.viewer, env_handle, cam_pos, cam_target)


    def _create_camera(self, env_idx, p = [0, -1, 0], axis = [0, 0, 1], angle = 90.0, follow = 'FOLLOW_POSITION'):
        self.graphics_device_id = self.sim_device_id
        # If you want to record any videos, call _create_camera() to create a camera in the scene.
        camera_props = gymapi.CameraProperties()
        camera_props.width, camera_props.height = CAMERA_WIDTH, CAMERA_HEIGHT
        camera_props.enable_tensors = True
        camera_idx = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*p)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*axis), np.radians(angle))
        self.gym.attach_camera_to_body(camera_idx, self.envs[env_idx], 0, local_transform, getattr(gymapi, follow))

        self.cameras[camera_idx] = env_idx
        camera_ts = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_idx], camera_idx, gymapi.IMAGE_COLOR) # IMAGE_COLOR - 4x 8 bit unsigned int - RGBA color
        camera_ts = gymtorch.wrap_tensor(camera_ts)
        self.camera_tensors.append(camera_ts)


    def _render_cameras(self):
        if len(self.cameras) > 0:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for idx in range(len(self.camera_tensors)):
                self.camera_frames[idx].append(self.camera_tensors[idx].cpu().numpy())
            self.gym.end_access_image_tensors(self.sim)


    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync


        if self.viewer or (self.headless == True and len(self.cameras) > 0):
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphicss
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)


    def save_record_video(self, name = 'video', postfix = 'mp4'):
        if len(self.camera_frames) > 0:
            assert postfix in ['gif', 'mp4']
            import imageio
            import cv2
            
            for idx, video_frames in self.camera_frames.items():
                video_path = f'{name}_env{idx}.{postfix}'
                if postfix == 'gif':
                    with imageio.get_writer(video_path, mode = 'I', duration = 1 / VIDEO_FPS) as writer:
                        for frame in video_frames:
                            writer.append_data(frame)
                elif postfix == 'mp4':
                    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT), True) # 
                    for frame in video_frames:
                        video.write(frame[..., :-1])
                    video.release()

                print(f'Save video to {video_path} ({len(video_frames)} frames, {len(video_frames) / VIDEO_FPS} seconds).')