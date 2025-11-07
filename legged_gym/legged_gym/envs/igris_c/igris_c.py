from legged_gym import LEGGED_GYM_ROOT_DIR, envs

import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.igris_c.igris_c_config import IGRISCCfg
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym.torch_utils import *

class IGRISC(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def get_current_obs(self):
        current_obs = torch.cat((   
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 1. - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        mirror_current_obs = self._mirror_observations(current_obs)
        return current_obs, mirror_current_obs
    
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

            def compute_side(roll_id, pitch_id):
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
                lo = torch.where(oob, zero, lo)
                hi = torch.where(oob, zero, hi)
                m = (hi + lo) / 2
                r = (hi - lo)

                # Write back
                self.dof_pos_limits[:, roll_id, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[:, roll_id, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

            compute_side(w_roll, w_pitch)

        ###ANKLE###
        # Indices
        L_roll = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions // 2 - 1
        R_roll = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions - 1
        L_pitch = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions // 2 - 2
        R_pitch = self.action_offset + self.cfg.env.num_waist + self.cfg.env.num_lower_actions - 2

        # Piecewise thresholds (must satisfy p1 < p2)
        p1, p2 = -0.37, 0.23
        if not (p1 < p2):
            raise ValueError("Require p1 < p2")

        # Ensure pitch plane stays at independent limits (optional, for clarity)
        self.dof_pos_limits[:, L_pitch, 0] = limits[L_pitch][0]
        self.dof_pos_limits[:, L_pitch, 1] = limits[L_pitch][1]
        self.dof_pos_limits[:, R_pitch, 0] = limits[R_pitch][0]
        self.dof_pos_limits[:, R_pitch, 1] = limits[R_pitch][1]

        def compute_side(roll_id, pitch_id):
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
            lo_z1 = s1 * rmin
            hi_z1 = s1 * rmax
            # zone2: flat at independent limits
            lo_z2 = pitch.new_full((B,), rmin)
            hi_z2 = pitch.new_full((B,), rmax)
            # zone3: ramp [rmin,rmax] -> 0
            lo_z3 = s3 * rmin
            hi_z3 = s3 * rmax

            lo = torch.where(z1, lo_z1, torch.where(z2, lo_z2, lo_z3))
            hi = torch.where(z1, hi_z1, torch.where(z2, hi_z2, hi_z3))

            # Clamp to independent bounds (keeps numerical noise in range)
            lo = torch.clamp(lo, min=rmin, max=rmax)
            hi = torch.clamp(hi, min=rmin, max=rmax)

            # Apply OOB policy: collapse to 0 if pitch is outside its valid range
            zero = pitch.new_zeros((B,))
            lo = torch.where(oob, zero, lo)
            hi = torch.where(oob, zero, hi)
            m = (hi + lo) / 2
            r = (hi - lo)

            # Write back
            self.dof_pos_limits[:, roll_id, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[:, roll_id, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        compute_side(L_roll, L_pitch)
        compute_side(R_roll, R_pitch)

    def _update_standstill_flags(self, env_ids):
        if env_ids.numel():
            self.standstill_flag[env_ids] = 0
            zero_envs = env_ids[(torch.norm(self.commands[env_ids, :3], dim=-1) < 0.1)\
                & (torch.norm(self.rb_states[env_ids, self.feet_indices[0], :2] - self.rb_states[env_ids, self.feet_indices[1], :2], dim=-1) < 0.3)]
            self.standstill_flag[zero_envs] = 1

    def _init_buffers(self):
        self.target_raibert_footholds = torch.zeros((self.num_envs, 3), device=self.device)
        self.standstill_flag = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return super()._init_buffers()

    #------------ reward functions----------------

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = self.dof_pos < self.dof_pos_limits[..., 0] # lower limit
        out_of_limits |= self.dof_pos > self.dof_pos_limits[..., 1]
        return torch.any(out_of_limits, dim=1)
    
    def _compute_standstill_reward(self):
        l_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 200.).float()
        r_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 200.).float()
        ret = (
            (l_contact * r_contact)*\
        (
            torch.exp(-(self.dof_pos - self.default_dof_pos).square().amax(dim=-1)/0.1)+\
            torch.exp(-self.dof_vel.square().amax(dim=-1)/0.1)
        )
        )[self.standstill_flag]
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
        k_woyaw = [1,3,7,9,13]
        ret_i = -0.03*torch.sum((self.dof_pos[:, i] - self.default_dof_pos[:, i]).abs(), dim=-1)
        ret_j = -0.12*torch.sum((self.dof_pos[:, j] - self.default_dof_pos[:, j]).abs(), dim=-1)
        ret_k = -0.22*torch.sum((self.dof_pos[:, k] - self.default_dof_pos[:, k]).abs(), dim=-1)
        ret_k[self.commands[:, 2].abs() > 0.4] = -0.25*torch.sum((self.dof_pos[:, k_woyaw] - self.default_dof_pos[:, k_woyaw]).abs(), dim=-1)[self.commands[:, 2].abs() > 0.4]
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