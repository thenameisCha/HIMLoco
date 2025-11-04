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
        cos_phase =  torch.cos(2 * torch.pi * self.phase ).unsqueeze(1)
        sin_phase = torch.sin(2 * torch.pi * self.phase ).unsqueeze(1)
        current_obs = torch.cat((   
                                    cos_phase, sin_phase,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs[:, 2:] += (2 * torch.rand_like(current_obs[:, 2:]) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

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
        num_section = 5
        mirror_obs[:, 0] = obs[:, 1]
        mirror_obs[:, 1] = obs[:, 0]
        mirror_obs[:, 3] *= -1
        mirror_obs[:, 4] *= -1
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

                # Write back
                self.dof_pos_limits[:, roll_id, 0] = lo
                self.dof_pos_limits[:, roll_id, 1] = hi

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

            # Write back
            self.dof_pos_limits[:, roll_id, 0] = lo
            self.dof_pos_limits[:, roll_id, 1] = hi

        compute_side(L_roll, L_pitch)
        compute_side(R_roll, R_pitch)
  
    def _post_physics_step_callback(self):
        period = 1.
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.phase_indicator = (self.phase >= 0.5).long() # phase indicator=1 means left is swing
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        phase_change_envs = torch.arange(self.num_envs, device=self.device)[(self.phase < 1.e-3) | ((self.phase - 0.5).abs() < 1.e-3)]
        self._update_raibert_footholds(phase_change_envs)
        self._update_standstill_flags(phase_change_envs)
        return super()._post_physics_step_callback()

    def _update_raibert_footholds(self, env_ids):
        """With current command velocities, compute target footholds based on Raibert's heuristic.
        Footholds are in stance frame coordinates.
        """
        T      = 1.
        vx     = self.commands[env_ids, 0]
        vy     = self.commands[env_ids, 1]
        omega  = self.commands[env_ids, 2]
        phase_ind = self.phase_indicator[env_ids]
        dpsi   = (
            T * omega
        )

        # Nominal forward placement (no yaw coupling yet)
        x_nom = T * vx

        # Your original lateral placement (nominal y without yaw coupling)
        y_nom = (
            (phase_ind * (2 * T * vy + self.cfg.commands.default_feet_width)
            - (1 - phase_ind) * self.cfg.commands.default_feet_width) * (vy >= 0.)
            +
            ((1 - phase_ind) * (2 * T * vy - self.cfg.commands.default_feet_width)
            + phase_ind * self.cfg.commands.default_feet_width) * (vy < 0.)
        )

        # First-order rotation of the nominal target by Δψ = ωT (Raibert small-angle coupling)
        c, s = torch.cos(dpsi), torch.sin(dpsi)
        x =  x_nom * c - y_nom * s
        y =  x_nom * s + y_nom * c
        self.target_raibert_footholds[env_ids, 0] = x
        self.target_raibert_footholds[env_ids, 1] = y

        # self.target_raibert_footholds[env_ids, 0] = x_nom
        # self.target_raibert_footholds[env_ids, 1] = y_nom

        # Optional: keep your safety check or clamp to enforce minimum width
        # assert not (y.abs() < self.cfg.commands.default_feet_width).any(), \
        #     f'Foot width target under self.cfg.commands.default_feet_width! At ' \
        #     f'{torch.nonzero(y.abs() < self.cfg.commands.default_feet_width, as_tuple=True)[0]}, ' \
        #     f'Values {torch.nonzero(y.abs() < self.cfg.commands.default_feet_width)[1]}'

        # Foot yaw target (unchanged): desired heading change this step
        self.target_raibert_footholds[env_ids, 2] = dpsi

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

    def _reward_swing_push(self):
        # Pushes the swing foot towards the target foothold, without specifying a dense trajectory
        # https://arxiv.org/pdf/2408.02662
        l_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 1.).float()
        r_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 1.).float()
        phase_sine = torch.sin(2*torch.pi*self.phase)
        env_list = torch.arange(self.num_envs, device=self.device)

        stance_idx = self.feet_indices[self.phase_indicator]
        swing_idx = self.feet_indices[1-self.phase_indicator]
        stance_state_global_frame = self.rb_states[env_list, stance_idx]
        swing_state_global_frame = self.rb_states[env_list, swing_idx]

        xy0 = torch.cat((self.target_raibert_footholds[:, :2], torch.zeros((self.num_envs, 1), device=self.device)), dim=-1)
        target_swing_pos_global_frame = stance_state_global_frame[:, :3] + quat_apply_yaw(stance_state_global_frame[:, 3:7], xy0)
        swing_pos_global_frame = swing_state_global_frame[:, :3]

        ret = (l_contact - r_contact)*\
            (phase_sine/torch.sqrt(phase_sine.square()+0.04))*\
                torch.exp(-torch.norm(swing_pos_global_frame[:, :2] - target_swing_pos_global_frame[:, :2], dim=-1)/0.25)

        ret[self.standstill_flag] = self._compute_standstill_reward()
        return ret
    
    def _reward_swing_ori(self):
        # Pushes the swing foot towards the target foothold, without specifying a dense trajectory
        # https://arxiv.org/pdf/2408.02662
        l_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 1.).float()
        r_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 1.).float()
        phase_sine = torch.sin(2*torch.pi*self.phase)
        env_list = torch.arange(self.num_envs, device=self.device)

        stance_idx = self.feet_indices[self.phase_indicator]
        swing_idx = self.feet_indices[1-self.phase_indicator]
        stance_state_global_frame = self.rb_states[env_list, stance_idx]
        swing_state_global_frame = self.rb_states[env_list, swing_idx]

        _, _, stance_y = get_euler_xyz(stance_state_global_frame[:, 3:7])
        _, _, swing_y = get_euler_xyz(swing_state_global_frame[:, 3:7])
        target_swing_yaw_global_frame = stance_y + self.target_raibert_footholds[:, 2]

        yaw_diff = wrap_to_pi(target_swing_yaw_global_frame - swing_y)
        ret = (l_contact - r_contact)*\
            (phase_sine/torch.sqrt(phase_sine.square()+0.04))*\
                torch.exp(-yaw_diff.square()/0.25)

        ret[self.standstill_flag] = self._compute_standstill_reward()
        return ret
    
    def _compute_standstill_reward(self):
        l_contact = (self.contact_forces[:, self.feet_indices[0], 2] > 100.).float()
        r_contact = (self.contact_forces[:, self.feet_indices[1], 2] > 100.).float()
        ret = (
            (l_contact * r_contact)*\
        (
            torch.exp(-(self.dof_pos - self.default_dof_pos).square().amax(dim=-1)/0.1)+\
            torch.exp(-self.dof_vel.square().amax(dim=-1)/0.1)
        )
        )[self.standstill_flag]
        return ret
    
    def _reward_feet_sliding(self):
        return torch.sum(self.rb_states[:, self.feet_indices, 7:9].norm(dim=-1)*(self.contact_forces[:, self.feet_indices, 2] > 1.).float(), dim=-1)