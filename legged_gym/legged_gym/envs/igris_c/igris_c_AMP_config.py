import glob
from legged_gym.envs.igris_c.igris_c_config import IGRISCCfg, IGRISCCfgPPO

class IGRISCAMPCfg( IGRISCCfg ):
    class env( IGRISCCfg.env ):
        reference_state_initialization = True
        reference_state_initialization_prob = .8
        amp_motion_files = {
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/walk_slow_woohyun_phc 1.pkl": {
                "hz": 30,
                "start_time": 95./50.,
                "end_time": 140./50.,
                "weight": 1.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/walk_slow2.pkl": {
                "hz": 30,
                "start_time": 125./30.,
                "end_time": 163./30.,
                "weight": 1.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/Female1Walking_c3d_0-B3_-_walk1_poses.pkl": {
                "hz": 30,
                "start_time": 29./30.,
                "end_time": 65./30.,
                "weight": 1.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/Female1Walking_c3d_0-B1_-_stand_to_walk_poses.pkl": {
                "hz": 30,
                "start_time": 1./30.,
                "end_time": 35./30.,
                "weight": 3.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/turn_1_woohyun_phc 1.pkl": {
                "hz": 30,
                "start_time": 125./30.,
                "end_time": 170./30.,
                "weight": 1.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/turn_1_woohyun_phc 1.pkl": {
                "hz": 30,
                "start_time": 30./30.,
                "end_time": 75./30.,
                "weight": 1.0
            },
            # "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/side_step.pkl": {
            #     "hz": 30,
            #     "start_time": 150./30.,
            #     "end_time": 180./30.,
            #     "weight": 1.0
            # },
            # "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/side_step.pkl": {
            #     "hz": 30,
            #     "start_time": 57./30.,
            #     "end_time": 99./30.,
            #     "weight": 1.0
            # },
        }
        amp_preload_transitions = True
        amp_num_preload_transitions = 2000000

    class rewards( IGRISCCfg.rewards ):
        base_height_target = 0.95
        soft_dof_pos_limit = .95 # percentage of urdf limits, values above this limit are penalized
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        class scales:
            termination = 0.0
            joint_power = -2.e-4
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.
            base_height = 0.5
            orientation = 1.
            dof_pos = 0.2
            penalize_contact_power = -1.e-2
            slow_touchdown = -0.2
            torque = -5.e-6
            swing_push = 4.
            swing_ori = 2.

    class commands(IGRISCCfg.commands):
        num_commands = 4
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0., 1.0] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]    # min max [rad/s]
            heading = [-3.14, 3.14]

class IGRISCAMPCfgPPO(IGRISCCfgPPO):
    runner_class_name = 'HIMOnPolicyRunner_AMP'
    class algorithm(IGRISCCfgPPO.algorithm):
        amp_replay_buffer_size = 100000
        disc_coef = 1.
        disc_grad_pen = 1.
    
    class runner(IGRISCCfgPPO.runner):
        experiment_name = 'igris_c_AMP' # should be the same as 'env' in env.py and env_config.py 
        algorithm_class_name = 'HIMPPO_AMP'
        amp_reward_coef = 3.0 * (IGRISCAMPCfg.sim.dt * IGRISCAMPCfg.control.decimation)
        amp_motion_files = IGRISCAMPCfg.env.amp_motion_files
        amp_num_preload_transitions = IGRISCAMPCfg.env.amp_num_preload_transitions
        amp_task_reward_lerp = .3
        amp_discr_hidden_dims = [128,]

        LOG_WANDB = True
        env_name = 'igris_c_AMP'
        file_name = 'igris_c_AMP'
        config_name = 'igris_c_AMP'
        wandb_name = 'igris_c'
