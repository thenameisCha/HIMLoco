# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.envs.base.legged_robot import LeggedRobot
from rsl_rl.rsl_rl.datasets.motion_loader import AMPLoader
from .amp_legged_robot_config import AMPLeggedRobotCfg

class AMPLeggedRobot(LeggedRobot):
    def __init__(self, cfg: AMPLeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.cfg.env.reference_state_initialization:
            self.reference_state_initialization_prob = getattr(
                self.cfg.env, "reference_state_initialization_prob", 1.0)
            motion_dict = getattr(self.cfg.env, "amp_motion_files", {})
            if motion_dict:
                self.amp_loader = AMPLoader(
                    device=self.device,
                    time_between_frames=self.dt,
                    preload_transitions=getattr(self.cfg.env, "amp_preload_transitions", False),
                    num_preload_transitions=getattr(self.cfg.env, "amp_num_preload_transitions", 0),
                    reference_dict=motion_dict
                )
            else:
                self.amp_loader = None

    def step_forced(self, joint_states: torch.Tensor, rootstate: torch.Tensor):
        self.render()
        self.reset()
        self.root_states[:, :3] = rootstate[:3].view(1, -1)
        self.root_states[0, :2] += 1.
        self.root_states[:, 3:7] = quat_from_euler_xyz(rootstate[3], rootstate[4], rootstate[5]).unsqueeze(dim=0)
        self.root_states[:, 7:] = rootstate[6:].view(1, -1)
        env_ids_int32 = torch.arange(self.num_envs, device=self.device).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        lower_body_pos = joint_states[self.action_offset:self.action_offset+self.num_actions].view(-1,1)
        lower_body_vel = torch.zeros_like(lower_body_pos)
        lower_body = torch.cat((lower_body_pos, lower_body_vel), dim=-1)
        joint_angles = torch.tile(lower_body, (self.num_envs, 1))
        self.gym.set_dof_state_tensor(self.sim,
                                               gymtorch.unwrap_tensor(joint_angles))
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        termination_privileged_obs_buf, mirror_termination_privileged_obs_buf = super().compute_termination_observations(env_ids)
        termination_amp_state = self.get_amp_observations()[env_ids]
        return termination_privileged_obs_buf, mirror_termination_privileged_obs_buf, termination_amp_state
    
    def get_amp_observations(self):
        dic = self._get_amp_observations_dict()
        ret = []            
        for _, val in dic.items():
            ret.append(val)
        return torch.hstack(ret)

    def get_amp_data_dict(self):
        # For data collection
        kb_pos = torch.zeros((self.num_envs, 13, 3), device=self.device)
        kb_pos[:, 2] = self.rb_states[:, self.feet_indices[0], :3]
        kb_pos[:, 3] = self.rb_states[:, self.feet_indices[1], :3]
        dic = {
            'dof':self.dof_pos,
            'dof_vel':self.dof_vel,
            'root_trans_offset': self.root_states[:, :3],
            'pose_aa': torch.zeros((self.num_envs, 39, 3), device=self.device),
            'smpl_joints': torch.zeros((self.num_envs, 24, 3), device=self.device),
            'root_rot': self.base_quat,
            'root_vel': self.root_states[:, 7:10],
            'root_angvel': self.root_states[:, 10:13],
            'kb_pos': kb_pos,
        }

    #------------- Callbacks --------------
    def _reset_states(self, env_ids):
        use_amp_init = (
            getattr(self.cfg.env, "reference_state_initialization", False)
            and getattr(self, "amp_loader", None) is not None
            and getattr(self.amp_loader, "num_motions", 0) > 0
        )
        if use_amp_init:
            rand_mask = torch.rand(env_ids.shape[0], device=self.device) < self.reference_state_initialization_prob
            env_ids_amp = env_ids[rand_mask]
            if env_ids_amp.numel() > 0:
                frames = self.amp_loader.get_full_frame_batch(env_ids_amp.shape[0])
                self._reset_dofs_amp(env_ids, frames, rand_mask)
                self._reset_root_states_amp(env_ids, frames, rand_mask)
        else:
            self._reset_dofs(env_ids) 
            self._reset_root_states(env_ids) 

    def _reset_dofs_amp(self, env_ids, frames, amp_mask):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        env_ids_amp = env_ids[amp_mask]
        self.dof_pos[env_ids] = self.default_dof_pos_cfg[:]
        self.dof_pos[env_ids,  self.action_offset:self.action_offset+self.num_actions] = self.default_dof_pos_cfg[:,  self.action_offset:self.action_offset+self.num_actions].repeat(len(env_ids), 1) * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_actions), device=self.device)
        # self.dof_pos[env_ids, :self.num_actions] = self.default_dof_pos_cfg[:, :self.num_actions].repeat(len(env_ids), 1) + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_actions), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.dof_pos[env_ids_amp,  self.action_offset:self.action_offset+self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = AMPLoader.get_joint_pose_batch(frames)
        # self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames, amp_mask):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
            self.root_states[env_ids, :2] += torch_rand_float(-2., 2., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2] = self.base_init_state[2] + self._get_heights()[env_ids].amax(dim=-1)
            self.custom_env_origins[env_ids] = self.root_states[env_ids, :3]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.custom_env_origins[env_ids] = self.root_states[env_ids, :3]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        # self.root_states[env_ids, 2] = AMPLoader.get_root_pos_batch(frames)

        random_yaw_angle = (2*torch.rand((env_ids.shape[0],1), device=self.device)-1)*torch.pi
        quat = quat_from_euler_xyz(torch.zeros_like(random_yaw_angle.squeeze()), torch.zeros_like(random_yaw_angle.squeeze()), random_yaw_angle.squeeze())
        self.root_states[env_ids, 3:7] = quat
        projected_gravity = AMPLoader.get_root_rot_batch(frames) # roll pitch
        if projected_gravity is not None:
            gx, gy, gz = projected_gravity.unbind(-1)
            roll  = torch.atan2(-gy, -gz)                           # rotation about body-X
            pitch = torch.atan2(gx, torch.sqrt(gy * gy + gz * gz))                 
            quat_amp = quat_from_euler_xyz(roll, pitch, random_yaw_angle[amp_mask].squeeze())
            self.root_states[env_ids[amp_mask], 3:7] = quat_amp
        # self.root_states[env_ids, 7:10] = AMPLoader.get_linear_vel_batch(frames)
        # self.root_states[env_ids, 10:13] = AMPLoader.get_angular_vel_batch(frames)
        self.last_root_pos[env_ids] = self.root_states[env_ids, :3]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    #----------------------------------------
    def _get_amp_observations_dict(self):
        base_height = self._get_base_heights()  
        # projected_gravity = self.projected_gravity
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        foot_pos = self._feet_positions_in_base_frame()
        dic = {
            'q_pos':self.dof_pos[:, self.action_offset:self.action_offset+self.num_actions],
            'q_vel':self.dof_vel[:, self.action_offset:self.action_offset+self.num_actions],
            # 'root_height': base_height.unsqueeze(dim=-1),
            'projected_gravity': self.projected_gravity,
            # 'root_linvel': base_lin_vel[:, 0].unsqueeze(dim=-1),
            # 'root_angvel': base_ang_vel,
            'feet_pos': foot_pos,
        }
        return dic

    def _feet_positions_in_base_frame(self):

        feet_indices = self.feet_indices
        feet_states = self.rb_states[:, feet_indices, :]
        assert feet_states.shape == (self.num_envs, 2, 13), f"feet state shape is {feet_states.shape}"
        Lfoot_positions_local = quat_rotate_inverse(self.base_quat ,feet_states[:, 0, :3] - self.root_states[:, :3]) 
        Rfoot_positions_local = quat_rotate_inverse(self.base_quat ,feet_states[:, 1, :3] - self.root_states[:, :3]) 

        return torch.concat((Lfoot_positions_local, Rfoot_positions_local), dim=-1)
    
    #------------ reward functions----------------
