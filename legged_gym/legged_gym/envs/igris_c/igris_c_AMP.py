from legged_gym import LEGGED_GYM_ROOT_DIR, envs

import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.igris_c.igris_c import IGRISC
from legged_gym.envs.igris_c.igris_c_AMP_config import IGRISCAMPCfg
from rsl_rl.datasets.motion_loader import AMPLoader
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from rsl_rl.modules.normalizer import Normalizer_obs

class IGRISCAMP(IGRISC):
    def __init__(self, cfg: IGRISCAMPCfg, sim_params, physics_engine, sim_device, headless):
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
        self.amp_normalizer = Normalizer_obs(self.amp_loader.observation_dim)
            
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        extras = super().compute_termination_observations(env_ids)
        termination_amp_state = self.get_amp_observations()[env_ids]
        extras['termination_amp_obs'] = termination_amp_state
        return extras
    
    def get_amp_observations(self):
        dic = self._get_amp_observations_dict()
        ret = []            
        for _, val in dic.items():
            ret.append(val)
        ret = torch.hstack(ret)
        return self.amp_normalizer(ret)
    
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
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.dof_pos[env_ids_amp,  self.action_offset:self.action_offset+self.cfg.env.num_waist+self.cfg.env.num_lower_actions] = AMPLoader.get_joint_pose_batch(frames)
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
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.3, 0.3, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        projected_gravity = AMPLoader.get_root_rot_batch(frames) # roll pitch
        if projected_gravity is not None:
            gx, gy, gz = projected_gravity.unbind(-1)
            roll  = torch.atan2(-gy, -gz)                           # rotation about body-X
            pitch = torch.atan2(gx, torch.sqrt(gy * gy + gz * gz))                 
            quat_amp = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
            self.root_states[env_ids[amp_mask], 3:7] = quat_amp
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
