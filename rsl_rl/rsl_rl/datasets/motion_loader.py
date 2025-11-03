import os
import glob
import logging
from lxml import etree

import torch
import numpy as np
import joblib
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util
from isaacgym.torch_utils import *
from scipy.signal import savgol_filter
from torch import optim

ROOT_HEIGHT_OFFSET = 0.3 #[m], retarget motion clip floats
WY = 0
WR = WY+1
WP = WR+1
LHP = WP+1
LHR = LHP+1
LHY = LHR+1
LKP = LHY+1
LAP = LKP+1
LAR = LAP+1
RHP = LHP+6
RHR = LHR+6
RHY = LHY+6
RKP = LKP+6
RAP = LAP+6
RAR = LAR+6
LSP = RAR+1
LSR = LSP+1
LSY = LSR+1
LEP = LSY+1
LWY = LEP+1
LWR = LWY+1
LWP = LWR+1
RSP = LSP+7
RSR = LSR+7
RSY = LSY+7
REP = LEP+7
RWY = LWY+7
RWR = LWR+7
RWP = LWP+7
JOINT_IDX = [ # Select the joints of interest
    WR, WP,
    LHP, LHR, LHY, LKP, LAP, LAR,
    RHP, RHR, RHY, RKP, RAP, RAR,
    # LSP, LSR, LSY, LEP,
    # RSP, RSR, RSY, REP
]
NUM_ACTIONS = len(JOINT_IDX) # Number of joints to imitate

class AMPLoader:
    def __init__(
            self,
            device,
            time_between_frames,
            preload_transitions=False,
            num_preload_transitions=1000000,
            reference_dict={}
            ):
        """Expert dataset provides AMP observations from human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.observation_keys = None
        self.joint_observation_keys = None
        self.observation_dim = 0
        self.joint_observation_dim = 0
        for i, (motion_path, info) in enumerate(reference_dict.items()): # 
            '''
            reference_dict : {
            motion_name: [data_dir, 
                        {
                        hz: mocap_hz, 
                        start_time: data_start_time [s],    
                        end_time: data_end_time [s],
                        weight: motion_weight
                        }]
                        }
            The data contains unwanted parts at the beginning and end.
            start_time and end_time indicate which part is the meaningful motion data.
            Weight decided how often the motion will be sampled by the AMPLoader
            '''                                                     
            motion_file = motion_path
            hz = info['hz']
            motion_start_index = int(hz * info['start_time'])
            motion_end_index = int(hz * info['end_time'])
            self.motion_start_tick = int(motion_start_index)
            self.motion_end_tick = int(motion_end_index)
            self.hz = float(hz)
            motion_weight = info['weight']
            
            print(motion_file)
            self.trajectory_names.append(os.path.splitext(os.path.basename(motion_file))[0])
            motion_data = self._load_motion_file(motion_file)
            processed_data_full = self.process_data(motion_data)
            processed_data_joints = {}
            for key, val in processed_data_full.items():
                if 'q_' in key and val is not None:
                    processed_data_full[key] = val
            self.trajectories.append( # Only joint space
                processed_data_joints
            )
            self.trajectories_full.append(
                processed_data_full
            )
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(float(motion_weight))
            frame_duration = 1/self.hz
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (self.motion_end_tick - self.motion_start_tick - 1) * frame_duration
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(self.motion_end_tick - self.motion_start_tick)

            self.observation_dim = 0
            for _, val in processed_data_full.items():
                if val is not None:
                    self.observation_dim += val.shape[-1]
            # self.observation_dim = processed_data_full.shape[-1]
            print(f"Loaded {traj_len}s. motion from {motion_file}.")
            print(f"Size of Reference Observation : {self.observation_dim}")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions

        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times) # error
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            
            preloaded_s, preloaded_s_next = [], []
            for key, _ in self.preloaded_s.items():
                preloaded_s.append(self.preloaded_s[key])
                preloaded_s_next.append(self.preloaded_s_next[key])
            self.preloaded_s_tensor, self.preloaded_s_next_tensor = torch.hstack(preloaded_s), torch.hstack(preloaded_s_next)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1
    
    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        ret = {}
        blend = p * n - idx_low
        for key, val in self.trajectories[traj_idx].items():
            frame_start = val[idx_low]
            frame_end = val[idx_high]
            ret[key] = self.slerp(frame_start, frame_end, blend)
        return ret

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        # all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float, device=self.device)
        # all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float, device=self.device)
        ret = {}
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float, requires_grad=False).unsqueeze(-1)
        it = 0
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            for key, val in trajectory.items():
                traj_mask = traj_idxs == traj_idx
            
                frame_start = val[idx_low[traj_mask]]
                frame_end = val[idx_high[traj_mask]]
                slerp = self.slerp(frame_start, frame_end, blend[traj_mask])
                if it == 0:
                    ret[key] = slerp
                else:
                    ret[key] = torch.vstack((ret[key], slerp))
            it += 1
        return ret

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        ret = {}
        blend = p * n - idx_low
        for key, val in self.trajectories_full[traj_idx].items():
            frame_start = val[idx_low]
            frame_end = val[idx_high]
            ret[key] = self.slerp(frame_start, frame_end, blend)
        return ret

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        full_dim = self.observation_dim
        ret = {}
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float, requires_grad=False).unsqueeze(-1)
        it = 0
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            for key, val in trajectory.items():
                traj_mask = traj_idxs == traj_idx
            
                frame_start = val[idx_low[traj_mask]]
                frame_end = val[idx_high[traj_mask]]
                slerp = self.slerp(frame_start, frame_end, blend[traj_mask])
                if it == 0:
                    ret[key] = slerp
                else:
                    ret[key] = torch.vstack((ret[key], slerp))
            it += 1
        return ret

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                len(self.preloaded_s), size=num_frames)
            ret = {}
            for key, val in self.preloaded_s.items():
                ret[key] = val[idxs]
            return ret
            # return [self.preloaded_s[int(i)] for i in idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)


    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s_tensor.shape[0], size=mini_batch_size)
                s = self.preloaded_s_tensor[idxs, :]
                s_next = self.preloaded_s_next_tensor[idxs, :]

            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    frame, next_frame = self.get_full_frame_at_time(traj_idx, frame_time), self.get_full_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames)
                    s_, s_next_ = [], []
                    for key, _ in frame.items():
                        s_ += frame[key]
                        s_next_ += next_frame[key]
                    s.append(torch.hstack(s_)), s_next.append(torch.hstack(s_next_))
                s, s_next = torch.vstack(s), torch.vstack(s_next)
            yield s.to(torch.float32), s_next.to(torch.float32)

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def _load_motion_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pkl':
            return joblib.load(path)
        motion = np.load(path, allow_pickle=True)
        if isinstance(motion, np.ndarray) and motion.ndim == 0 and hasattr(motion, 'item'):
            return motion.item()
        return np.asarray(motion)

    def process_data(self, traj): # Process the raw mocap data
        if isinstance(traj, dict):
            root_pos = np.asarray(traj['root_trans_offset'])
            pose_aa = np.asarray(traj.get('pose_aa', np.zeros((root_pos.shape[0], 24, 3))))
            q_pos = np.asarray(traj['dof'])
            root_quat = np.asarray(traj['root_rot'])
            root_linvel = np.asarray(traj.get('root_vel', np.zeros_like(root_pos)))
            root_angvel = np.asarray(traj.get('root_angvel', np.zeros_like(root_pos)))
            q_vel = np.asarray(traj.get('dof_vel', np.zeros_like(q_pos)))
            smpl_joints = np.asarray(traj.get('smpl_joints', np.zeros((root_pos.shape[0], 39, 3))))
            kb_pos = np.asarray(traj.get('kb_pos', np.zeros((root_pos.shape[0], 13, 3))))
            L_wrist_pos, R_wrist_pos, L_ankle_pos, R_ankle_pos = [kb_pos[:, i, :] for i in range(4)]
        else:
            start = 0
            num_section = 3
            root_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 39*3
            pose_aa = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 31
            q_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 4 # WXYZ quaternion
            root_quat = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            root_linvel = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            root_angvel = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 31
            q_vel = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 24*3
            smpl_joints = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            L_wrist_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            R_wrist_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            L_ankle_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]
            start += num_section
            num_section = 3
            R_ankle_pos = traj[self.motion_start_tick:self.motion_end_tick, start:start+num_section]

        root_pos_torch = to_torch(root_pos, dtype=torch.float)
        root_quat_torch = to_torch(root_quat, dtype=torch.float)
        # root_quat_torch_clone = root_quat_torch.clone()
        # root_quat_torch[:, 1:] = root_quat_torch_clone[:, :3]
        # root_quat_torch[:, 0] = root_quat_torch_clone[:, 3]
        root_vel_torch = to_torch(root_linvel, dtype=torch.float)
        root_angvel_torch = to_torch(root_angvel, dtype=torch.float)
        q_pos_torch = to_torch(q_pos, dtype=torch.float)
        q_vel_torch = to_torch(q_vel, dtype=torch.float)
        L_wrist_pos_torch = to_torch(L_wrist_pos, dtype=torch.float)
        R_wrist_pos_torch = to_torch(R_wrist_pos, dtype=torch.float)
        L_foot_pos_torch = to_torch(L_ankle_pos, dtype=torch.float)
        R_foot_pos_torch = to_torch(R_ankle_pos, dtype=torch.float)
        # root_quat_torch_wxyz = root_quat_torch.clone()
        # root_quat_torch[:, :3] = root_quat_torch_wxyz[:, 1:4]
        # root_quat_torch[:, 3] = root_quat_torch_wxyz[:, 0] # Change quat to xyzw

        # Create AMP observation
        base_pos = root_pos_torch[:, :3]
        # base_pos[:, 2] -= ROOT_HEIGHT_OFFSET
        base_height = root_pos_torch[:, 2]
        assert (~torch.isfinite(base_height)).sum() == 0, "Found non finite element"
        base_rpy = convert_euler_to_pi_range(*get_euler_xyz(root_quat_torch)) # tuple of tensors
        assert (~torch.isfinite(root_quat_torch)).sum() == 0, "Found non finite element 0"
        # base_lin_vel = root_vel_torch
        base_lin_vel = quat_rotate_inverse(root_quat_torch, root_vel_torch) 
        assert (~torch.isfinite(base_lin_vel)).sum() == 0, "Found non finite element 1"
        # base_ang_vel = root_angvel_torch
        base_ang_vel = quat_rotate_inverse(root_quat_torch, root_angvel_torch)
        assert (~torch.isfinite(base_ang_vel)).sum() == 0, "Found non finite element 2"
        L_wrist_pos_base_torch = quat_rotate_inverse(root_quat_torch, L_wrist_pos_torch - root_pos_torch)
        assert (~torch.isfinite(L_wrist_pos_base_torch)).sum() == 0, "Found non finite element 3"
        R_wrist_pos_base_torch = quat_rotate_inverse(root_quat_torch, R_wrist_pos_torch - root_pos_torch) 
        assert (~torch.isfinite(R_wrist_pos_base_torch)).sum() == 0, "Found non finite element 4"
        L_foot_pos_base_torch = quat_rotate_inverse(root_quat_torch, L_foot_pos_torch - root_pos_torch)
        assert (~torch.isfinite(L_foot_pos_base_torch)).sum() == 0, "Found non finite element 3"
        R_foot_pos_base_torch = quat_rotate_inverse(root_quat_torch, R_foot_pos_torch - root_pos_torch) 
        assert (~torch.isfinite(R_foot_pos_base_torch)).sum() == 0, "Found non finite element 4"

        q_pos_subset = q_pos_torch[:, JOINT_IDX]
        q_vel_subset = q_vel_torch[:, JOINT_IDX]
        gravity_vec = root_quat_torch.new_tensor([0.0, 0.0, -1.0]).expand(root_quat_torch.shape[0], -1)
        projected_gravity = quat_rotate_inverse(root_quat_torch, gravity_vec)
        ret = {
            'q_pos':q_pos_subset,
            'q_vel':q_vel_subset,
            # 'root_height': base_height.unsqueeze(dim=-1),
            'projected_gravity': projected_gravity,
            # 'root_linvel': base_lin_vel[:, 0].unsqueeze(dim=-1),
            # 'root_angvel': base_ang_vel,
            # 'wrist_pos': torch.cat((L_wrist_pos_base_torch, R_wrist_pos_base_torch), dim=-1),
            'feet_pos': torch.cat((L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1),

        }
        # ret = {
        #     'q_pos':q_pos_subset,
        #     'q_vel':q_vel_subset,
        #     'root_pos': base_pos,
        #     'projected_gravity': projected_gravity,
        #     'root_rpy': torch.cat((base_rpy[0].unsqueeze(dim=-1),base_rpy[1].unsqueeze(dim=-1),(base_rpy[2].unsqueeze(dim=-1))), dim=-1),
        #     'root_linvel': base_lin_vel,
        #     'root_angvel': base_ang_vel,
        #     'feet_pos': torch.cat((L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1),
        # }
        return ret

    # def get_root_pos(pose):
    #     return pose[2*NUM_ACTIONS]

    # def get_root_pos_batch(poses):
    #     return poses[:, 2*NUM_ACTIONS]

    def get_root_rot(pose):
        return pose['projected_gravity']

    def get_root_rot_batch(poses):
        return poses['projected_gravity'] if 'projected_gravity' in poses else None

    # def get_linear_vel(pose):
    #     return pose[2*NUM_ACTIONS+3:2*NUM_ACTIONS+6]
    
    # def get_linear_vel_batch(poses):
    #     return poses[:, 2*NUM_ACTIONS+3:2*NUM_ACTIONS+6]

    # def get_angular_vel(pose):
    #     return pose[2*NUM_ACTIONS+6:2*NUM_ACTIONS+9]  

    # def get_angular_vel_batch(poses):
    #     return poses[:, 2*NUM_ACTIONS+6:2*NUM_ACTIONS+9]
    
    def get_joint_pose(pose):
        return pose['q_pos']

    def get_joint_pose_batch(poses):
        return poses['q_pos']
  
    # def get_joint_vel(pose):
    #     return pose[NUM_ACTIONS:2*NUM_ACTIONS]

    # def get_joint_vel_batch(poses):
    #     return poses[:, NUM_ACTIONS:2*NUM_ACTIONS]
          
    def get_wrist_pos(pose):
        return pose['wrist_pos']

    def get_wrist_pos_batch(poses):
        return poses['wrist_pos']
          
    def get_foot_pos(pose):
        return pose['feet_pos']

    def get_foot_pos_batch(poses):
        return poses['feet_pos']


##################HELPER FUNCTIONS###################

def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles (ZYX) to a rotation matrix. """
    B = euler_angles.shape[0]
    c = torch.cos(euler_angles)
    s = torch.sin(euler_angles)

    # Preallocate rotation matrix
    R = torch.zeros((B, 3, 3), dtype=torch.float16, device=euler_angles.device)

    # Fill in the entries
    R[:, 0, 0] = c[:, 1] * c[:, 0]
    R[:, 0, 1] = c[:, 1] * s[:, 0]
    R[:, 0, 2] = -s[:, 1]
    R[:, 1, 0] = s[:, 2] * s[:, 1] * c[:, 0] - c[:, 2] * s[:, 0]
    R[:, 1, 1] = s[:, 2] * s[:, 1] * s[:, 0] + c[:, 2] * c[:, 0]
    R[:, 1, 2] = s[:, 2] * c[:, 1]
    R[:, 2, 0] = c[:, 2] * s[:, 1] * c[:, 0] + s[:, 2] * s[:, 0]
    R[:, 2, 1] = c[:, 2] * s[:, 1] * s[:, 0] - s[:, 2] * c[:, 0]
    R[:, 2, 2] = c[:, 2] * c[:, 1]

    return R

def transform_vector(euler_angles, vectors):
    """ Transform vectors from frame A to B using a batch of Euler angles.
    
    Args:
    - euler_angles (numpy.array): Array of shape [B, 3] containing Euler angles.
    - vectors (numpy.array): Array of shape [B, 3] containing vectors in frame A.
    
    Returns:
    - transformed_vectors (numpy.array): Array of shape [B, 3] containing vectors in frame B.
    """
    # Convert inputs to torch tensors
    euler_angles = torch.tensor(euler_angles, dtype=torch.float16)
    vectors = torch.tensor(vectors, dtype=torch.float16)

    # Get rotation matrices from Euler angles
    R = euler_to_rotation_matrix(euler_angles)

    # Transform vectors
    transformed_vectors = torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)

    # Convert back to numpy arrays
    return transformed_vectors.numpy()

# def quat_from_angle_axis(angle, axis: np.ndarray):
#     theta = (angle / 2).expand_dims(-1)
#     xyz = normalize(axis) * np.sin(theta)
#     w = np.cos(theta)
#     return quat_unit(np.concatenate((xyz, w), axis=-1))

def normalize(x: np.ndarray, eps: float = 1e-9):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True).clip(eps, None)

def quat_unit(a):
    return normalize(a)

def denoise(data, window_length=5, polyorder=3):
    print("DENOISING!!")
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def convert_euler_to_pi_range(roll_, pitch_, yaw_):
    roll = (roll_ + np.pi) % (2 * np.pi) - np.pi
    pitch = (pitch_ + np.pi) % (2 * np.pi) - np.pi
    yaw = (yaw_ + np.pi) % (2 * np.pi) - np.pi
    return roll, pitch, yaw
