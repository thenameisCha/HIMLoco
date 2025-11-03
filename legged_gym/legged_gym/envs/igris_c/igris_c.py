from legged_gym import LEGGED_GYM_ROOT_DIR, envs

import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.igris_c.igris_c_config import IGRISCAMPCfg
from legged_gym.utils.math import quat_apply_yaw, convert_euler_to_pi_range, wrap_to_pi


class IGRISCAMP(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def get_current_obs(self):
        current_obs = torch.cat((   
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
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
        num_section=3
        # ang vel
        mirror_obs[:, 0] *= -1
        mirror_obs[:, 2] *= -1
        start += num_section
        # projected gravity
        num_section = 3
        mirror_obs[:, 4] *= -1
        start += num_section
        # commands
        num_section = 3
        mirror_obs[:, 7] *= -1
        mirror_obs[:, 8] *= -1
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


    #------------ reward functions----------------

    def _reward_upper_regularization(self):
        num_arms = self.num_actions-self.cfg.env.num_waist-self.cfg.env.num_lower_actions
        WY = 0
        WR = self.action_offset if self.cfg.env.num_waist==2 else WY+1
        WP = WR+1
        LHP = self.action_offset+self.cfg.env.num_waist
        LHR = LHP+1
        LHY = LHR+1
        LKP = LHY+1
        LAP = LKP+1
        LAR = LAP+1
        RHP = LHP+self.cfg.env.num_lower_actions//2
        RHR = LHR+self.cfg.env.num_lower_actions//2
        RHY = LHY+self.cfg.env.num_lower_actions//2
        RKP = LKP+self.cfg.env.num_lower_actions//2
        RAP = LAP+self.cfg.env.num_lower_actions//2
        RAR = LAR+self.cfg.env.num_lower_actions//2
        LSP = self.action_offset+self.cfg.env.num_waist+self.cfg.env.num_lower_actions
        LSR = LSP+1
        LSY = LSR+1
        LEP = LSY+1
        RSP = LSP+num_arms//2
        RSR = LSR+num_arms//2
        RSY = LSY+num_arms//2
        REP = LEP+num_arms//2
        joint_idx = torch.tensor([
            WR,
            WP,
            LHR, LHY,
            LAR,
            RHR, RHY,
            RAR,
        ], device=self.device)
        joint_idx_upper = torch.tensor([
            # WY,
            LSP, LSR, LSY, LEP,
            RSP, RSR, RSY, REP
        ], device=self.device)
        joint_sigma_upper = torch.tensor([
            # .1,
            .3, .1, .1, .3,
            .3, .1, .1, .3
        ], device=self.device).unsqueeze(dim=0)
        return torch.exp(-torch.mean(torch.square(self.dof_pos[:, joint_idx_upper] - self.default_dof_pos[:, joint_idx_upper]) / joint_sigma_upper, dim=1))
       

### HELPERS ###    
def cubic(start_pos, start_vel, end_pos, end_vel, end_time, current_time: torch.Tensor):
    '''
    A function that outputs f(current_time), where f is constrained to be a cubic function where
    f(t=0) = start_pos
    f(t=end_time) = end_pos
    f'(t=0) = start_vel
    f'(t=end_time) = end_vel
    '''

    # Coefficients of the cubic polynomial
    over_ids = (current_time > end_time + 1e-5).nonzero(as_tuple=False).flatten()
    assert (current_time <= end_time + 1e-3).all(), f"Current time in cubic excceds end time!!\nend time : {end_time[over_ids]}\ncurrent_time : {current_time[over_ids]}"
    a = (2 * (start_pos - end_pos) + start_vel * end_time + end_vel * end_time) / (end_time ** 3)
    b = (3 * (end_pos - start_pos) - 2 * start_vel * end_time - end_vel * end_time) / (end_time ** 2)
    c = start_vel
    d = start_pos

    # Compute position at current_time
    pos = a * (current_time ** 3) + b * (current_time ** 2) + c * current_time + d
    vel = 3*a*(current_time ** 2) + 2*b*current_time + c

    return pos, vel

def heel_toe_swing_position(
    start_xyz,                    # (N,3) tensor or tuple/list of 3 tensors (N,)
    end_xyz,                      # (N,3) tensor or tuple/list of 3 tensors (N,)
    s: torch.Tensor,              # (N,) normalized phase in [0,1]
    step_time: torch.Tensor,  # step period
    *,
    apex_clearance: float = 0.05, # [m] fixed apex above ground
    apex_phase: float = 0.45,     # 0.45–0.55 typical
    burst_gain: float = 1.,      # initial horizontal velocity scale (bursty lift-off)
    retract_phase: float = 0.80,  # begin late-swing retraction
    retract_max: float = 0.03,    # [m] cap retraction magnitude (2–5 cm typical)
    adaptive_retract: bool = True,# scale retraction with step length
    retract_mode: str = "auto",   # "auto" (along travel dir), "x_only", or "none"
):
    """
    Returns:
    pos: (N,3) desired (x,y,z) at phase s
    vel: (N,3) desired velocity; physical m/s if step_time is given, else d/ds

    Notes:
    - x,y: quintic with x'(0)=burst, x''(0)=0, x'(1)=0, x''(1)=0, plus min-jerk retraction.
    - z  : piecewise min-jerk with fixed apex at `apex_clearance` and zero vertical vel at apex & touchdown.
    """
    over_ids = (s > 1.0).nonzero(as_tuple=False).flatten()
    assert (s <= 1.0+ 1e-3).all(), f"Current time in cubic excceds end time!!\ncurrent_time : {s[over_ids]}"

    # ---------- helpers ----------
    def _as_xyz(p):
        if isinstance(p, (tuple, list)):
            return p[0], p[1], p[2]
        return p[..., 0], p[..., 1], p[..., 2]

    def _minjerk(u: torch.Tensor) -> torch.Tensor:
        # 10u^3 - 15u^4 + 6u^5 ; C2, zero vel/acc at endpoints
        return 10*u**3 - 15*u**4 + 6*u**5

    def _dminjerk(u: torch.Tensor) -> torch.Tensor:
        # derivative wrt u: 30u^2 - 60u^3 + 30u^4 = 30*u^2*(1-u)**2
        return 30*u**2 - 60*u**3 + 30*u**4

    def _quintic_axis(x0, x1, s, v0):
        # x(0)=x0, x(1)=x1, x'(0)=v0, x'(1)=0, x''(0)=x''(1)=0
        dx = x1 - x0
        s2, s3, s4, s5 = s*s, s**3, s**4, s**5
        x = (x0
            + v0 * s
            + (10.0*dx - 6.0*v0) * s3
            + (-15.0*dx + 8.0*v0) * s4
            + (6.0*dx - 3.0*v0) * s5)
        # derivative wrt s
        dx_ds = (v0
                + 3.0*(10.0*dx - 6.0*v0) * s2
                + 4.0*(-15.0*dx + 8.0*v0) * s3
                + 5.0*(6.0*dx - 3.0*v0) * s4)
        return x, dx_ds

    # ---------- unpack & clamp ----------
    sx, sy, sz = _as_xyz(start_xyz)
    ex, ey, ez = _as_xyz(end_xyz)
    s = s.clamp(0.0, 1.0)
    eps = torch.finfo(s.dtype).eps

    # ---------- horizontal (x,y): quintic + retraction ----------
    dx, dy = ex - sx, ey - sy
    v0x, v0y = burst_gain * dx, burst_gain * dy


    if retract_mode == "none":
        xq, dxq_ds = _quintic_axis(sx, ex, s, v0x)
        yq, dyq_ds = _quintic_axis(sy, ey, s, v0y)
        xr, yr = xq, yq
        vxr_ds, vyr_ds = dxq_ds, dyq_ds
    else:
        # Retraction blend from retract_phase→1 via min-jerk
        u = ((s - retract_phase) / (1.0 - retract_phase + eps)).clamp(0.0, 1.0)
        r_blend = _minjerk(u)
        du_ds = torch.where((s >= retract_phase) & (s <= 1.0),
                            torch.full_like(s, 1.0 / (1.0 - retract_phase + eps)),
                            torch.zeros_like(s))
        drblend_ds = _dminjerk(u) * du_ds  # zero outside [retract_phase,1]

        if retract_mode == "x_only":
            L = torch.abs(dx)
            dir_x = torch.sign(dx)
            dir_y = torch.zeros_like(dy)
        else:
            # "auto": along planar travel direction
            L = torch.sqrt(dx*dx + dy*dy)
            dir_x = torch.where(L > 1e-9, dx / (L + eps), torch.zeros_like(dx))
            dir_y = torch.where(L > 1e-9, dy / (L + eps), torch.zeros_like(dy))

        if adaptive_retract:
            r_mag = torch.clamp(0.25 * L, max=retract_max)  # ≤25% of step, capped
        else:
            r_mag = torch.full_like(L, retract_max)

        # positions with retraction

        xq, dxq_ds = _quintic_axis(sx, ex+r_mag*dir_x, s, v0x)
        yq, dyq_ds = _quintic_axis(sy, ey+r_mag*dir_y, s, v0y)
        xr = xq - r_mag * r_blend * dir_x
        yr = yq - r_mag * r_blend * dir_y

        # derivatives wrt s
        vxr_ds = dxq_ds - r_mag * drblend_ds * dir_x
        vyr_ds = dyq_ds - r_mag * drblend_ds * dir_y



    # ---------- vertical (z): piecewise min-jerk with fixed apex ----------
    sa = apex_phase
    za = (sz + ez) / 2 + apex_clearance
    u1 = (s / (sa + eps)).clamp(0.0, 1.0)
    u2 = ((s - sa) / (1.0 - sa + eps)).clamp(0.0, 1.0)

    z_up  = sz + (za - sz) * _minjerk(u1)
    z_dn  = za + (ez - za) * _minjerk(u2)
    dzup_ds = (za - sz) * _dminjerk(u1) * (1.0 / (sa + eps))
    dzdn_ds = (ez - za) * _dminjerk(u2) * (1.0 / (1.0 - sa + eps))

    zr     = torch.where(s <= sa, z_up, z_dn)
    vzr_ds = torch.where(s <= sa, dzup_ds, dzdn_ds)

    # ---------- pack ----------
    vxr_ds, vyr_ds, vzr_ds = vxr_ds/step_time, vyr_ds/step_time, vzr_ds/step_time
    return xr, yr, zr, vxr_ds, vyr_ds, vzr_ds