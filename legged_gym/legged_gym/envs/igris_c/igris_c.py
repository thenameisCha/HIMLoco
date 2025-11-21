from legged_gym import LEGGED_GYM_ROOT_DIR, envs

import torch
import torch.nn.functional as F
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.igris_c.igris_c_config import IGRISCCfg
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym.torch_utils import *

class IGRISC(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.debug_viz = False
        self.waist_constraints = {
            'r_a_init_': torch.tensor([
                [0.0, 0.0905, -0.04],
                [0.0, -0.0905, -0.04]
            ], dtype=torch.float32, device=self.device),
            'r_b_init_': torch.tensor([
                [-0.05167, 0.09050, -0.04587],
                [-0.05167, -0.09050, -0.04587]
            ], dtype=torch.float32, device=self.device),
            'r_c_init_': torch.tensor([
                [-0.05, 0.0940, 0.014],
                [-0.05, -0.0940, 0.014]
            ], dtype=torch.float32, device=self.device),
            'r_c_offset_local_': torch.tensor([
                [-0.05, 0.094, 0.014],
                [-0.05, -0.094, 0.014]
            ], dtype=torch.float32, device=self.device),
            
            'base_to_p1_offset': torch.tensor([0.0, 0.0, -0.04], dtype=torch.float32, device=self.device),
            'base_to_p1_axis': torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=self.device),
            'p1_to_p2_offset': torch.tensor([0.0, 0.0, 0.04], dtype=torch.float32, device=self.device),
            'p1_to_p2_axis': torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            'motor_angles_min_': torch.tensor([-0.05, -0.05], dtype=torch.float32, device=self.device),
            'motor_angles_max_': torch.tensor([0.68, 0.68], dtype=torch.float32, device=self.device),
            'is_elbow_up_': True
        }
        self.L_ankle_constraints = {
            'r_a_init_': torch.tensor([
                [0.0, 0.03775, 0.26],
                [0.0, -0.03775, 0.152]
            ], dtype=torch.float32, device=self.device),
            'r_b_init_': torch.tensor([
                [-0.03750, 0.03750, 0.25989],
                [-0.03750, -0.03750, 0.15181]
            ], dtype=torch.float32, device=self.device),
            'r_c_init_': torch.tensor([
                [-0.03400, 0.03100, 0.0],
                [-0.03400, -0.03100, 0.0]
            ], dtype=torch.float32, device=self.device),
            'r_c_offset_local_': torch.tensor([
                [-0.034, 0.031, 0.0],
                [-0.034, -0.031, 0.0]
            ], dtype=torch.float32, device=self.device),
            
            'base_to_p1_offset': torch.tensor([0.0, 0.0, -0.0], dtype=torch.float32, device=self.device),
            'base_to_p1_axis': torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            'p1_to_p2_offset': torch.tensor([0.0, 0.0, -0.0], dtype=torch.float32, device=self.device),
            'p1_to_p2_axis': torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            'motor_angles_min_': torch.tensor([-36.1 *torch.pi/180, -35.4 *torch.pi/180], dtype=torch.float32, device=self.device),
            'motor_angles_max_': torch.tensor([34.9 *torch.pi/180, 30 *torch.pi/180], dtype=torch.float32, device=self.device),
            'is_elbow_up_': False
        }
        self.R_ankle_constraints = {
            'r_a_init_': torch.tensor([
                [0.0, -0.03775, 0.26],
                [0.0, 0.03775, 0.152]
            ], dtype=torch.float32, device=self.device),
            'r_b_init_': torch.tensor([
                [-0.03750, -0.03750, 0.25989],
                [-0.03750, 0.03750, 0.15181]
            ], dtype=torch.float32, device=self.device),
            'r_c_init_': torch.tensor([
                [-0.03400, -0.03100, 0.0],
                [-0.03400, 0.03100, 0.0]
            ], dtype=torch.float32, device=self.device),
            'r_c_offset_local_': torch.tensor([
                [-0.034, -0.031, 0.0],
                [-0.034, 0.031, 0.0]
            ], dtype=torch.float32, device=self.device),

            'base_to_p1_offset': torch.tensor([0.0, 0.0, -0.0], dtype=torch.float32, device=self.device),
            'base_to_p1_axis': torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device),
            'p1_to_p2_offset': torch.tensor([0.0, 0.0, -0.0], dtype=torch.float32, device=self.device),
            'p1_to_p2_axis': torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
            'motor_angles_min_': torch.tensor([-36.1 *torch.pi/180, -35.4 *torch.pi/180], dtype=torch.float32, device=self.device),
            'motor_angles_max_': torch.tensor([34.9 *torch.pi/180, 30 *torch.pi/180], dtype=torch.float32, device=self.device),
            'is_elbow_up_': False
        }
    
    def _mirror_observations(self, obs):
        num_waist = self.cfg.env.num_waist
        num_legs = self.cfg.env.num_lower_actions
        num_arms = self.num_actions - num_waist - num_legs
        mirror_obs = obs.clone()
        start=0
        # commands
        num_section = 3
        mirror_obs[:, 1] *= -1
        mirror_obs[:, 2] *= -1
        start += num_section
        # ang vel
        num_section=3
        mirror_obs[:, 3] *= -1
        mirror_obs[:, 5] *= -1
        start += num_section
        # projected gravity
        num_section = 3
        mirror_obs[:, 7] *= -1
        start += num_section
        # dof pos
        num_section = self.num_actions
        mirror_obs[:, start] *= -1
        if num_waist==3:
            mirror_obs[:, start+1] *= -1
        mirror_obs[:, start+num_waist:start+num_waist + num_legs//2] = obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs]
        mirror_obs[:, start+num_waist+1:start+num_waist+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs//2-1] *= -1
        mirror_obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs] = obs[:, start+num_waist:start+num_waist+num_legs//2]
        mirror_obs[:, start+num_waist+num_legs//2+1:start+num_waist+num_legs//2+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs-1] *= -1
            # arms
        if num_arms > 0:
            mirror_obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms] = obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2]
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+2] *= -1
            mirror_obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2] = obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms]
            mirror_obs[:, start+num_waist+num_legs+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+2] *= -1
        start += num_section
        # dof vel
        num_section = self.num_actions
        mirror_obs[:, start] *= -1
        if num_waist==3:
            mirror_obs[:, start+1] *= -1
        mirror_obs[:, start+num_waist:start+num_waist + num_legs//2] = obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs]
        mirror_obs[:, start+num_waist+1:start+num_waist+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs//2-1] *= -1
        mirror_obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs] = obs[:, start+num_waist:start+num_waist+num_legs//2]
        mirror_obs[:, start+num_waist+num_legs//2+1:start+num_waist+num_legs//2+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs-1] *= -1
            # arms
        if num_arms > 0:
            mirror_obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms] = obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2]
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+2] *= -1
            mirror_obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2] = obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms]
            mirror_obs[:, start+num_waist+num_legs+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+2] *= -1
        start += num_section
        # actions
        num_section = self.num_actions
        mirror_obs[:, start] *= -1
        if num_waist==3:
            mirror_obs[:, start+1] *= -1
        mirror_obs[:, start+num_waist:start+num_waist + num_legs//2] = obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs]
        mirror_obs[:, start+num_waist+1:start+num_waist+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs//2-1] *= -1
        mirror_obs[:, start+num_waist+num_legs//2:start+num_waist+num_legs] = obs[:, start+num_waist:start+num_waist+num_legs//2]
        mirror_obs[:, start+num_waist+num_legs//2+1:start+num_waist+num_legs//2+1+2] *= -1
        mirror_obs[:, start+num_waist+num_legs-1] *= -1
            # arms
        if num_arms > 0:
            mirror_obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms] = obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2]
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+num_arms//2+2] *= -1
            mirror_obs[:, start+num_waist+num_legs:start+num_waist+num_legs+num_arms//2] = obs[:, start+num_waist+num_legs+num_arms//2:start+num_waist+num_legs+num_arms]
            mirror_obs[:, start+num_waist+num_legs+1] *= -1
            mirror_obs[:, start+num_waist+num_legs+2] *= -1
        start += num_section
        # base lin vel
        num_section = 3
        mirror_obs[:, start+1] *= -1
        start += num_section
        # disturbance
        num_section = 3
        mirror_obs[:, start+1] *= -1
        start += num_section
        # height scan
        if self.cfg.terrain.measure_heights:
            num_section = self.measured_heights.shape[-1]
            mirror_obs[:, start:start+num_section] = obs[:, start:start+num_section][:, self.height_perm]
            start += num_section
        assert start == self.num_one_step_privileged_obs, f'_mirror_observations size mismatch, obs shape : {obs.shape}, mirror obs length : {start}'
        return mirror_obs
          
    def _compute_joint_limits(self):
        limits = self.hard_dof_pos_limits[0]  # [dof, 2]
        B = self.dof_pos.shape[0]

        def compute_side(roll_id, pitch_id, r_at_pmin, r_at_pmax):
            rmin, rmax = limits[roll_id]
            pmin, pmax = limits[pitch_id]

            pitch_raw = self.dof_pos[:, pitch_id]
            # Out-of-range mask on raw pitch (your choice: zero out in this case)
            oob = (pitch_raw < pmin) | (pitch_raw > pmax)

            # Clamp used for ramps to avoid extrapolation
            pitch = torch.clamp(pitch_raw, min=pmin, max=pmax)

            # Masks (cover boundaries)
            z1 = pitch <= p1
            z2 = (pitch > p1) & (pitch < p2)
            z3 = pitch >= p2

            # Safe denominators
            eps = 1e-8
            d1 = max(p1 - pmin, eps)   # pmin -> p1
            d3 = max(pmax - p2, eps)   # p2   -> pmax

            # Scales in [0,1]
            s1 = (pitch - pmin) / d1         # 0 at pmin -> 1 at p1
            s3 = (pmax - pitch) / d3         # 1 at p2   -> 0 at pmax

            # Zone values
            # zone1: ramp 0 -> [rmin,rmax]
            lo_z1 = s1 * (rmin - (-r_at_pmin)) + (-r_at_pmin) 
            hi_z1 = s1 * (rmax - r_at_pmin) + r_at_pmin 
            # zone2: flat at independent limits
            lo_z2 = pitch.new_full((B,), rmin)
            hi_z2 = pitch.new_full((B,), rmax)
            # zone3: ramp [rmin,rmax] -> 0
            lo_z3 = s3 * (rmin - (-r_at_pmax)) + (-r_at_pmax)
            hi_z3 = s3 * (rmax - r_at_pmax) + r_at_pmax

            lo = torch.where(z1, lo_z1, torch.where(z2, lo_z2, lo_z3))
            hi = torch.where(z1, hi_z1, torch.where(z2, hi_z2, hi_z3))

            # Clamp to independent bounds (keeps numerical noise in range)
            lo = torch.clamp(lo, min=rmin, max=rmax)
            hi = torch.clamp(hi, min=rmin, max=rmax)

            # Apply OOB policy: collapse to 0 if pitch is outside its valid range
            zero = pitch.new_zeros((B,))
            lo = torch.where(oob, -r_at_pmin, lo)
            hi = torch.where(oob, r_at_pmin, hi)
            m = (hi + lo) / 2
            r = (hi - lo)

            # Write back
            self.dof_pos_limits[:, roll_id, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[:, roll_id, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        ###WAIST###
        if self.cfg.env.num_waist >= 2:
            num_waist = self.cfg.env.num_waist
            w_roll = self.action_offset+num_waist-2
            w_pitch = self.action_offset+num_waist-1
            p1, p2 = -0.68, 0.05
            r_at_pmin, r_at_pmax = 0.2, 0.2
            if not (p1 < p2):
                raise ValueError("Require p1 < p2")
            self.dof_pos_limits[:, w_pitch, 0] = limits[w_pitch][0]
            self.dof_pos_limits[:, w_pitch, 1] = limits[w_pitch][1]
            compute_side(w_roll, w_pitch, r_at_pmin, r_at_pmax)

        ###ANKLE###
        # Indices
        L_roll = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions // 2 - 1
        R_roll = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions - 1
        L_pitch = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions // 2 - 2
        R_pitch = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions - 2

        # Piecewise thresholds (must satisfy p1 < p2)
        p1, p2 = -0.37, 0.2
        r_at_pmax, r_at_pmin = 0.05, 0.05
        if not (p1 < p2):
            raise ValueError("Require p1 < p2")

        # Ensure pitch plane stays at independent limits (optional, for clarity)
        self.dof_pos_limits[:, L_pitch, 0] = limits[L_pitch][0]
        self.dof_pos_limits[:, L_pitch, 1] = limits[L_pitch][1]
        self.dof_pos_limits[:, R_pitch, 0] = limits[R_pitch][0]
        self.dof_pos_limits[:, R_pitch, 1] = limits[R_pitch][1]
        compute_side(L_roll, L_pitch, r_at_pmin, r_at_pmax)
        compute_side(R_roll, R_pitch, r_at_pmin, r_at_pmax)

    def _update_standstill_flags(self):
        self.standstill_flag[:] = 0
        zero_envs = (torch.norm(self.commands[:, :3], dim=-1) < 0.1)\
            & (torch.norm(self.rb_states[:, self.feet_indices[0], :2] - self.rb_states[:, self.feet_indices[1], :2], dim=-1) < 0.3)
        self.standstill_flag[zero_envs] = 1

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_standstill_flags()
        self._update_commands()
        self._compute_centroidal_dynamics()

    def _init_buffers(self):
        self.standstill_flag = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return super()._init_buffers()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set y commands of high vel envs to zero
        self.commands[env_ids, 1:3] *= (torch.norm(self.commands[env_ids, 0:1], dim=1) < 0.6).unsqueeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, :2] *= (torch.abs(self.commands[env_ids, :2]) > 0.2)
        self.commands[env_ids, 2] *= (self.commands[env_ids, 2].abs() > 0.2)

    def _update_commands(self):
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1])
            self.commands[:, 2] *= (torch.norm(self.commands[:, 0:1], dim=1) < 0.6)
            self.commands[:, 2] *= (self.commands[:, 2].abs() > 0.2)
 
    def _update_fourbar_linkage(self):
        self.motor_pos[:] = self.dof_pos[:]
        self.motor_vel[:] = self.dof_vel[:]

        dof_pos_clipped = torch.clip(self.dof_pos, self.dof_pos_limits[..., 0], self.dof_pos_limits[..., 1])
        waist_dof_pos = dof_pos_clipped[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist][:, [1,0]]  # pitch, roll
        L_ankle_dof_pos = dof_pos_clipped[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2]
        R_ankle_dof_pos = dof_pos_clipped[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions]
        waist_dof_vel = self.dof_vel[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist][:, [1,0]] # pitch, roll
        L_ankle_dof_vel = self.dof_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2]
        R_ankle_dof_vel = self.dof_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions]

        self.waist_jac, self.motor_pos[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist] = self._fourbar_logic(waist_dof_pos, self.waist_constraints)
        self.L_ankle_jac, self.motor_pos[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2] = self._fourbar_logic(L_ankle_dof_pos, self.L_ankle_constraints)
        self.R_ankle_jac, self.motor_pos[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = self._fourbar_logic(R_ankle_dof_pos, self.R_ankle_constraints)
        self.motor_vel[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist] = torch.bmm(self.waist_jac, waist_dof_vel.unsqueeze(-1)).squeeze(-1)
        self.motor_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2] = torch.bmm(self.L_ankle_jac, L_ankle_dof_vel.unsqueeze(-1)).squeeze(-1)
        self.motor_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = torch.bmm(self.R_ankle_jac, R_ankle_dof_vel.unsqueeze(-1)).squeeze(-1)

    def _fourbar_logic(self, dof_pos_clipped, constraints):
        # IK computation, dof_pos -> motor_pos
        dof_pos = dof_pos_clipped.clone()
        B = dof_pos.shape[0]
        r_c_ = torch.zeros((B, 2, 3), device=self.device) # [B,2,3]        
        r_a_init_ = constraints['r_a_init_'].unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_b_init_ = constraints['r_b_init_'].unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_c_init_ = constraints['r_c_init_'].unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_c_offset_local_ = constraints['r_c_offset_local_'].unsqueeze(dim=0).repeat(B,1,1)   # [B,2,3]
        base_to_p1_offset = constraints['base_to_p1_offset'].unsqueeze(dim=0).repeat(B,1)  # [B,3]
        base_to_p1_axis = constraints['base_to_p1_axis'].unsqueeze(dim=0).repeat(B,1)    # [B,3]
        p1_to_p2_offset = constraints['p1_to_p2_offset'].unsqueeze(dim=0).repeat(B,1)    # [B,3]
        p1_to_p2_axis = constraints['p1_to_p2_axis'].unsqueeze(dim=0).repeat(B,1)      # [B,3]
        is_elbow_up = constraints['is_elbow_up_'] # bool
        motor_limit_l_ = constraints.get('motor_angles_min_', torch.tensor([-1.57, -1.57], device=self.device))
        motor_limit_h_ = constraints.get('motor_angles_max_', torch.tensor([1.57, 1.57], device=self.device))
        motor_angles_temp = torch.zeros((B,2), device=self.device)

        l_bar_ = torch.norm(r_b_init_ - r_a_init_, dim=-1) # [B,2]
        l_rod_ = torch.norm(r_c_init_ - r_b_init_, dim=-1)  # [B,2]
        r_b_offset_local_ = r_b_init_ - r_a_init_  # [B,2,3]
        b_vec_ = r_b_init_ - r_a_init_  # [B,2,3]

        rot_base_to_p1 = axis_angle_to_quat(base_to_p1_axis, dof_pos[:, 0])  # [B,4]
        rot_p1_to_p2 = axis_angle_to_quat(p1_to_p2_axis, dof_pos[:, 1])  # [B,4]
        base_to_p1_offset_expanded = base_to_p1_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        p1_to_p2_offset_expanded = p1_to_p2_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        base_to_p1_axis_expanded = base_to_p1_axis.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        p1_to_p2_axis_expanded = p1_to_p2_axis.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        r_c_ = base_to_p1_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        rot_base_to_p1_expanded = rot_base_to_p1.unsqueeze(1).repeat(1,2,1)  # [B,2,4]
        rot_p1_to_p2_expanded = rot_p1_to_p2.unsqueeze(1).repeat(1,2,1)  # [B,2,4]
        r_c_ = base_to_p1_offset_expanded + quat_apply(rot_base_to_p1_expanded, p1_to_p2_offset_expanded) + \
            quat_apply(quat_mul(rot_base_to_p1_expanded, rot_p1_to_p2_expanded), r_c_offset_local_)  # [B,2,3]
        
        a_vec_ = r_c_ - r_a_init_
        d = -(l_rod_.square()-l_bar_.square()-a_vec_.square().sum(dim=-1)) / 2 # [B,2]
        e = d - a_vec_[..., 1] * b_vec_[..., 1] # [B,2]
        A = (a_vec_.square()[..., 0] + a_vec_.square()[..., 2]) * (b_vec_.square()[..., 0] + b_vec_.square()[..., 2])  # [B,2]
        B_ = (a_vec_[..., 0] * b_vec_[..., 2] - a_vec_[..., 2] * b_vec_[..., 0]) * e  # [B,2]
        C = e.square() - (a_vec_.square()[..., 0] * b_vec_.square()[..., 0] + a_vec_.square()[..., 2] * b_vec_.square()[..., 2] + 2 * a_vec_[..., 0] * a_vec_[..., 2] * b_vec_[..., 0] * b_vec_[..., 2])  # [B,2]
        value_pos_sign = torch.clamp(
            (B_ + torch.sqrt(torch.clamp(B_ * B_ - A * C, min=0.0))) / A, -1.0, 1.0)
        value_neg_sign = torch.clamp(
            (B_ - torch.sqrt(torch.clamp(B_ * B_ - A * C, min=0.0))) / A, -1.0, 1.0)
        
        motor_angle_pos = torch.asin(value_pos_sign)  # [B,2]
        motor_angle_neg = torch.asin(value_neg_sign)  # [B,2]   
        motor_angle_candidates = torch.zeros((6,B,2), device=self.device)  # [6,B,2], initialized to large negative value meaning invalid

        env_1 = (motor_angle_pos >= motor_limit_l_) &\
            (motor_angle_pos <= motor_limit_h_) # [B,2]
        env_2 = (motor_angle_neg >= motor_limit_l_) &\
            (motor_angle_neg <= motor_limit_h_ )
        env_3 = (torch.pi - motor_angle_pos >= motor_limit_l_) &\
            (torch.pi - motor_angle_pos <= motor_limit_h_)
        env_4 = (torch.pi - motor_angle_neg >= motor_limit_l_) &\
            (torch.pi - motor_angle_neg <= motor_limit_h_)
        env_5 = (-torch.pi - motor_angle_pos >= motor_limit_l_ )&\
            (-torch.pi - motor_angle_pos <= motor_limit_h_)
        env_6 = (-torch.pi - motor_angle_neg >= motor_limit_l_) &\
            (-torch.pi - motor_angle_neg <= motor_limit_h_)
        env_masks = [env_1, env_2, env_3, env_4, env_5, env_6] # list of [B,2] masks
        motor_angle_candidates[0][env_1] = motor_angle_pos[env_1] 
        motor_angle_candidates[1][env_2] = motor_angle_neg[env_2]
        motor_angle_candidates[2][env_3] = torch.pi - motor_angle_pos[env_3]
        motor_angle_candidates[3][env_4] = torch.pi - motor_angle_neg[env_4]
        motor_angle_candidates[4][env_5] = -torch.pi - motor_angle_pos[env_5]
        motor_angle_candidates[5][env_6] = -torch.pi - motor_angle_neg[env_6]

        r_a_init_expanded = r_a_init_ # [B,2,3]
        r_b_offset_local_expanded = r_b_offset_local_  # [B,2,3]
        y_axis = torch.tensor([0.,1.,0.], device=self.device).unsqueeze(0)
        for j in range(6):
            env_mask = env_masks[j] # [B,2]
            q_motor_flattened = motor_angle_candidates[j, env_mask] # [2N]
            r_a_init_flattened = r_a_init_expanded[env_mask] # [2N,3]
            r_b_offset_local_flattened = r_b_offset_local_expanded[env_mask] # [2N,3]
            r_c_flattened = r_c_[env_mask]  # [2N,3]
            r_b_ = r_a_init_flattened + quat_apply(axis_angle_to_quat(y_axis, q_motor_flattened), r_b_offset_local_flattened)  # [2N,3])

            bar = r_b_ - r_a_init_flattened  # [2N,3]
            rod = r_c_flattened - r_b_  # [2N,3]
            l_rod_condition = torch.abs(rod.norm(dim=-1) - l_rod_[env_mask]) < 1e-4  # [2N]
            elbow_direction = torch.cross(bar, rod, dim=1)[:, 1] > 0 # [2N]
            elbow_direction_condition = (elbow_direction == is_elbow_up) # [2N]

            valid_condition = l_rod_condition & elbow_direction_condition  # [2N]
            motor_angles_temp[env_mask] = q_motor_flattened * valid_condition.float() + motor_angles_temp[env_mask] * (~valid_condition).float()

        motor_angles_temp = torch.clip(motor_angles_temp, motor_limit_l_, motor_limit_h_)
        
        # Jacobian computation, \dot{m} = J\dot{q}
        r_b_ = r_a_init_ + quat_apply(axis_angle_to_quat(y_axis, motor_angles_temp), r_b_offset_local_) # [B,2,3]
        r_bar_ = r_b_ - r_a_init_ # [B,2,3]
        r_rod_ = r_c_ - r_b_ # [B,2,3]
        J_x = torch.zeros((B,2,6), dtype=torch.float32, device=self.device)
        J_x[..., :3] = r_rod_
        J_x[..., 3:] = torch.cross(r_c_, r_rod_)
        J_theta = torch.zeros(B, 2, 2, device=self.device, dtype=torch.float32)
        cross_bar_rod = torch.cross(r_bar_, r_rod_, dim=-1)   # (B,2,3)
        J_theta[:, 0, 0] = cross_bar_rod[:, 0, 1]
        J_theta[:, 1, 1] = cross_bar_rod[:, 1, 1]

        d0 = J_theta[:, 0, 0].clone()
        d1 = J_theta[:, 1, 1].clone()
        eps = 1e-8
        mask0 = torch.abs(d0) < eps
        mask1 = torch.abs(d1) < eps

        d0[mask0] = torch.where(d0[mask0] >= 0, torch.full_like(d0[mask0], eps),
                                                torch.full_like(d0[mask0], -eps))
        d1[mask1] = torch.where(d1[mask1] >= 0, torch.full_like(d1[mask1], eps),
                                                torch.full_like(d1[mask1], -eps))
        J_=J_x.clone()
        J_[:,0] /= d0.unsqueeze(dim=1)
        J_[:,1] /= d1.unsqueeze(dim=1)

        axis1_base = F.normalize(constraints['base_to_p1_axis'], dim=-1)   # (3,)
        axis2_local = F.normalize(constraints['p1_to_p2_axis'], dim=-1)    # (3,)
        axis1_base = axis1_base.unsqueeze(0).expand(B, 3)                  # (B,3)
        axis2_local = axis2_local.unsqueeze(0).expand(B, 3)                # (B,3)

        # rotations we already computed earlier in IK
        rot_base_to_p1 = axis_angle_to_quat(
            constraints['base_to_p1_axis'].unsqueeze(0).expand(B, 3),
            dof_pos[:, 0]
        )  # (B,4)

        # joint 2 axis in base frame
        axis2_base = quat_apply(rot_base_to_p1, axis2_local)               # (B,3)

        # joint origins and p2 origin in base
        base_to_p1_offset = constraints['base_to_p1_offset'].unsqueeze(0).expand(B, 3)
        p1_to_p2_offset   = constraints['p1_to_p2_offset'].unsqueeze(0).expand(B, 3)

        o1 = base_to_p1_offset                                             # (B,3)
        p2_origin = base_to_p1_offset + quat_apply(rot_base_to_p1, p1_to_p2_offset)  # (B,3)
        o2 = p2_origin                                                     # joint 2 axis through p2 origin

        # angular part
        Jw1 = axis1_base                                                  # (B,3)
        Jw2 = axis2_base                                                  # (B,3)

        # linear part: Jv1 = ω1 × (p2 - o1), Jv2 = ω2 × (p2 - o2) = 0
        Jv1 = torch.cross(Jw1, (p2_origin - o1), dim=-1)                  # (B,3)
        Jv2 = torch.zeros_like(Jv1)                                       # (B,3)

        # Build jac_joint with [linear; angular] order, already swapped like in C++
        jac_joint = torch.zeros(B, 6, 2, device=self.device, dtype=torch.float32)
        jac_joint[:, 0:3, 0] = Jv1
        jac_joint[:, 3:6, 0] = Jw1
        jac_joint[:, 0:3, 1] = Jv2
        jac_joint[:, 3:6, 1] = Jw2

        return torch.bmm(J_, jac_joint), motor_angles_temp

    def _process_torques(self, torques):
        control_type = self.cfg.control.control_type
        if control_type=="P":
            ### Jacobian Mapping from motor Kp/Kd to joint Kp/Kd (J^T@K@J) ###
            # scaled_pgain, scaled_dgain = self.p_gains*self.Kp_factors, self.d_gains*self.Kd_factors
            # waist_pgain, waist_dgain = scaled_pgain[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist], scaled_dgain[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist]
            # waist_pgain, waist_dgain = waist_pgain[:, [1,0]], waist_dgain[:, [1,0]]
            # L_ankle_pgain, L_ankle_dgain = scaled_pgain[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2], scaled_dgain[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2]
            # R_ankle_pgain, R_ankle_dgain = scaled_pgain[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions], scaled_dgain[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions]

            # waist_pgains, waist_dgains = waist_pgain.diag_embed(), waist_dgain.diag_embed()
            # L_ankle_pgains, L_ankle_dgains = L_ankle_pgain.diag_embed(), L_ankle_dgain.diag_embed()
            # R_ankle_pgains, R_ankle_dgains = R_ankle_pgain.diag_embed(), R_ankle_dgain.diag_embed()

            # waist_pgains, waist_dgains = torch.bmm(torch.bmm(self.waist_jac.transpose(1,2), waist_pgains), self.waist_jac), torch.bmm(torch.bmm(self.waist_jac.transpose(1,2), waist_dgains), self.waist_jac)
            # L_ankle_pgains, L_ankle_dgains = torch.bmm(torch.bmm(self.L_ankle_jac.transpose(1,2), L_ankle_pgains), self.L_ankle_jac), torch.bmm(torch.bmm(self.L_ankle_jac.transpose(1,2), L_ankle_dgains), self.L_ankle_jac)
            # R_ankle_pgains, R_ankle_dgains = torch.bmm(torch.bmm(self.R_ankle_jac.transpose(1,2), R_ankle_pgains), self.R_ankle_jac), torch.bmm(torch.bmm(self.R_ankle_jac.transpose(1,2), R_ankle_dgains), self.R_ankle_jac)
            
            # waist_dof_error = (self.joint_pos_target - self.dof_pos)[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist][:, [1,0]].unsqueeze(dim=-1)
            # waist_dof_vel = self.dof_vel[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist][:, [1,0]].unsqueeze(dim=-1)
            # L_ankle_dof_error = (self.joint_pos_target - self.dof_pos)[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2].unsqueeze(dim=-1)
            # L_ankle_dof_vel = self.dof_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2].unsqueeze(dim=-1)
            # R_ankle_dof_error = (self.joint_pos_target - self.dof_pos)[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions].unsqueeze(dim=-1)
            # R_ankle_dof_vel = self.dof_vel[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions].unsqueeze(dim=-1)
            
            # torques[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist] = (waist_pgains@waist_dof_error-waist_dgains@waist_dof_vel).squeeze(dim=-1)[:, [1,0]]
            # torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2] = (L_ankle_pgains@L_ankle_dof_error-L_ankle_dgains@L_ankle_dof_vel).squeeze(dim=-1)
            # torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = (R_ankle_pgains@R_ankle_dof_error-R_ankle_dgains@R_ankle_dof_vel).squeeze(dim=-1)

            ### Recompute PD torques in motor space, and Jacobian transpose ###
            motor_torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.motor_pos) - self.d_gains * self.Kd_factors * self.motor_vel
            
            waist_motor_torque = motor_torques[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist]
            L_ankle_motor_torque = motor_torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2]
            R_ankle_motor_torque = motor_torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions]

            torques[:, self.cfg.env.num_waist-2:self.cfg.env.num_waist] = torch.bmm(self.waist_jac.transpose(1,2), waist_motor_torque.unsqueeze(-1)).squeeze(-1)[:, [1,0]]
            torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions//2] = torch.bmm(self.L_ankle_jac.transpose(1,2), L_ankle_motor_torque.unsqueeze(-1)).squeeze(-1)
            torques[:, self.cfg.env.num_waist+self.cfg.env.num_lower_actions-2:self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = torch.bmm(self.R_ankle_jac.transpose(1,2), R_ankle_motor_torque.unsqueeze(-1)).squeeze(-1)

    def _saturate_target_dof_pos(self, target_pos):
        '''
        Implements equation (9) from https://arxiv.org/pdf/2509.06342
        target_pos is the target joint position computed from the policy actions.
        When the robot has four-bar linkages, target_pos is in motor space.
        '''
        num_waist = self.cfg.env.num_waist
        num_legs = self.cfg.env.num_lower_actions
        hard_motor_limits = self.dof_pos_limits.clone() # Motor level limits
        hard_motor_limits[:, num_waist-2:num_waist, 0] = self.waist_constraints['motor_angles_min_']
        hard_motor_limits[:, num_waist-2:num_waist, 1] = self.waist_constraints['motor_angles_max_'] 
        hard_motor_limits[:, num_waist+num_legs//2-2:num_waist+num_legs//2, 0] = self.L_ankle_constraints['motor_angles_min_']
        hard_motor_limits[:, num_waist+num_legs//2-2:num_waist+num_legs//2, 1] = self.L_ankle_constraints['motor_angles_max_'] 
        hard_motor_limits[:, num_waist+num_legs-2:num_waist+num_legs, 0] = self.R_ankle_constraints['motor_angles_min_']
        hard_motor_limits[:, num_waist+num_legs-2:num_waist+num_legs, 1] = self.R_ankle_constraints['motor_angles_max_'] 

        soft_motor_limits = hard_motor_limits.clone()
        soft_limit_band = 0.95
        m = (hard_motor_limits[:, :, 0] + hard_motor_limits[:, :, 1]) / 2
        r = (hard_motor_limits[:, :, 1] - hard_motor_limits[:, :, 0])
        soft_motor_limits[:, :, 0] = m - 0.5 * r * soft_limit_band
        soft_motor_limits[:, :, 1] = m + 0.5 * r * soft_limit_band

        upper_soft_violation = (self.motor_pos > soft_motor_limits[:, :, 1]) & (target_pos > hard_motor_limits[:, :, 1])
        lower_soft_violation = (self.motor_pos < soft_motor_limits[:, :, 0]) & (target_pos < hard_motor_limits[:, :, 0])

        target_pos[upper_soft_violation] -= ((self.motor_pos - soft_motor_limits[:, :, 1])/(hard_motor_limits[:, :, 1] - soft_motor_limits[:, :, 1])* (target_pos - hard_motor_limits[:, :, 1]))[upper_soft_violation]
        target_pos[lower_soft_violation] += ((soft_motor_limits[:, :, 0] - self.motor_pos)/(soft_motor_limits[:, :, 0] - hard_motor_limits[:, :, 0]) * (hard_motor_limits[:, :, 0] - target_pos))[lower_soft_violation]

    #------------ reward functions----------------

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = self.dof_pos < self.dof_pos_limits[..., 0] # lower limit
        out_of_limits |= self.dof_pos > self.dof_pos_limits[..., 1]
        return torch.any(out_of_limits, dim=1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * self.standstill_flag
    
    def _reward_stand_still_vel(self):
        return torch.sum(torch.abs(self.dof_vel), dim=1) * self.standstill_flag

    def _reward_stand_still_contact(self):
        l_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 250.).float()
        r_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 250.).float()
        ret = (l_contact + r_contact) * self.standstill_flag
        return ret
    
    def _reward_contact_power(self):
        ret = torch.sum(torch.sum(self.contact_forces[:, self.feet_indices] * self.rb_states[:, self.feet_indices, 7:10], dim=-1).abs(), dim=-1)
        ret[self.standstill_flag] = 0.
        return ret

    def _reward_slow_touchdown(self): # https://arxiv.org/pdf/2509.06342
        ret = (((torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=-1) > 1.)&(torch.norm(self.last_contact_forces[:, self.feet_indices, :3], dim=-1) < 1.))*\
            torch.amax(torch.cat((torch.norm(self.rb_states[:, self.feet_indices, 7:10], dim=-1, keepdim=True), torch.norm(self.last_feet_states[..., 7:10], dim=-1, keepdim=True)), dim=-1), dim=-1)).sum(dim=-1)
        ret[self.standstill_flag] = 0.
        return ret
    
    def _reward_dof_pos(self):
        i = [2, 5, 6, 8, 11, 12]
        j = [0]
        k = [1, 3, 4, 7, 9, 10, 13]
        # k_woyaw = [1,3,7,9,13]
        ret_i = -0.05*torch.sum((self.dof_pos[:, i] - self.default_dof_pos[:, i]).abs(), dim=-1)
        ret_j = -0.1*torch.sum((self.dof_pos[:, j] - self.default_dof_pos[:, j]).abs(), dim=-1)
        ret_k = -0.2*torch.sum((self.dof_pos[:, k] - self.default_dof_pos[:, k]).abs(), dim=-1)
        # ret_k[self.commands[:, 2].abs() > 0.4] = -0.2*torch.sum((self.dof_pos[:, k_woyaw] - self.default_dof_pos[:, k_woyaw]).abs(), dim=-1)[self.commands[:, 2].abs() > 0.4]
        return ret_i + ret_j + ret_k

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        in_contact = self.feet_contact_time > 0.
        in_mode_time = torch.where(in_contact, self.feet_contact_time, self.feet_air_time)
        single_stance = in_contact.int().sum(dim=-1) == 1
        rew_airTime = torch.amin(torch.where(single_stance.unsqueeze(dim=-1), in_mode_time, 0.), dim=1).clip(max=0.6)
        # rew_airTime = torch.sum(torch.clip(self.feet_air_time+self.feet_contact_time, max=0.4), dim=-1) * condition # reward only on first contact with the ground
        # rew_airTime = torch.sum(torch.clip(self.feet_air_time, max=0.4), dim=-1) * condition # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command

        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= torch.logical_and(contact, self.last_contacts) 
        return rew_airTime

    def _reward_feet_contact_forces(self):
        ret = torch.clip(self.contact_forces[:, self.feet_indices, 2].sum(dim=-1)-650., max=400)
        return ret*(self.contact_forces[:, self.feet_indices, 2].sum(dim=-1) > 650)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
            torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_feet_sliding(self):
        # Penalize feet hitting vertical surfaces
        return torch.sum(torch.norm(self.feet_vel[..., :2], dim=-1) * (self.contact_forces[:, self.feet_indices, 2] > 1.), dim=1)
        
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        double_contact = torch.sum(1.*contacts, dim=1)==0
        return 1.*double_contact

    def _reward_collision(self):
        contact_forces_masked = self.contact_forces.clone()
        contact_forces_masked[:, self.feet_indices, :] = 0.
        return torch.sum(1.*(torch.norm(contact_forces_masked, dim=-1) > 1.), dim=-1)
    
class IGRISCWB( IGRISC ):
    def _reward_dof_pos(self):
        i = [3, 6, 7, 9, 12, 13]
        j = [0, 1]
        k = [2, 4, 5, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        ret_i = -0.05*torch.sum((self.dof_pos[:, i] - self.default_dof_pos[:, i]).abs(), dim=-1)
        ret_j = -0.1*torch.sum((self.dof_pos[:, j] - self.default_dof_pos[:, j]).abs(), dim=-1)
        ret_k = -0.2*torch.sum((self.dof_pos[:, k] - self.default_dof_pos[:, k]).abs(), dim=-1)
        return ret_i + ret_j + ret_k
    

################HELPER#########################
def axis_angle_to_quat(axis_vecs: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Args:
        axis_vecs: (..., 3) tensor of rotation axes (need not be unit length).
        angles:    (...) tensor of rotation angles in radians; broadcast-compatible
                   with axis_vecs[..., 0].
    Returns:
        (..., 4) tensor of quaternions in [x, y, z, w] order.
    """
    if axis_vecs.shape[-1] != 3:
        raise ValueError(f"axis_vecs must end with dim 3, got {axis_vecs.shape}")

    # Move angles onto the same device/dtype
    angles = angles.to(device=axis_vecs.device, dtype=axis_vecs.dtype)

    # Determine broadcasted batch shape
    batch_shape = torch.broadcast_shapes(axis_vecs.shape[:-1], angles.shape)

    # Broadcast tensors
    axis = axis_vecs.expand(*batch_shape, 3)
    ang = angles.expand(*batch_shape)

    axis = F.normalize(axis, dim=-1)
    half = 0.5 * ang
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)

    quat = torch.empty(*batch_shape, 4, dtype=axis.dtype, device=axis.device)
    quat[..., :3] = axis * sin_half.unsqueeze(-1)
    quat[..., 3] = cos_half
    return quat