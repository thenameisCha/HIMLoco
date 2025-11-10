import glob
from legged_gym.envs.igris_c.igris_c_config import IGRISCCfg, IGRISCCfgPPO, IGRISCWBCfg, IGRISCWBCfgPPO

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
            # "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/Female1Walking_c3d_0-B1_-_stand_to_walk_poses.pkl": {
            #     "hz": 30,
            #     "start_time": 1./30.,
            #     "end_time": 35./30.,
            #     "weight": 1.0
            # },
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
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/side_step.pkl": {
                "hz": 30,
                "start_time": 150./30.,
                "end_time": 180./30.,
                "weight": 1.0
            },
            "/home/robros/isaac_ws/HIMLoco/rsl_rl/rsl_rl/datasets/mocap_motions/igris_motions/side_step.pkl": {
                "hz": 30,
                "start_time": 57./30.,
                "end_time": 99./30.,
                "weight": 1.0
            },
        }
        amp_preload_transitions = True
        amp_num_preload_transitions = 2000000

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
        amp_task_reward_lerp = .5
        amp_discr_hidden_dims = [128,]
        LOG_WANDB = True
        env_name = 'igris_c'
        file_name = 'igris_c_AMP'
        config_name = 'igris_c_AMP'
        file2_name = 'igris_c'
        config2_name = 'igris_c'
        wandb_name = 'igris_c'

class IGRISCWBAMPCfg( IGRISCWBCfg ):
    class env( IGRISCWBCfg.env ):
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

class IGRISCWBAMPCfgPPO( IGRISCWBCfgPPO ):
    runner_class_name = 'HIMOnPolicyRunner_AMP'
    class algorithm(IGRISCWBCfgPPO.algorithm):
        amp_replay_buffer_size = 100000
        disc_coef = 1.
        disc_grad_pen = 1.
    
    class runner(IGRISCWBCfgPPO.runner):
        experiment_name = 'igris_c_AMP' # should be the same as 'env' in env.py and env_config.py 
        algorithm_class_name = 'HIMPPO_AMP'
        amp_reward_coef = 3.0 * (IGRISCAMPCfg.sim.dt * IGRISCAMPCfg.control.decimation)
        amp_motion_files = IGRISCAMPCfg.env.amp_motion_files
        amp_num_preload_transitions = IGRISCAMPCfg.env.amp_num_preload_transitions
        amp_task_reward_lerp = .5
        amp_discr_hidden_dims = [128,]
        LOG_WANDB = True
        env_name = 'igris_c'
        file_name = 'igris_c_AMP'
        config_name = 'igris_c_AMP'
        file2_name = 'igris_c'
        config2_name = 'igris_c'
        wandb_name = 'igris_c'

class IGRISCAMPIMCfg( IGRISCAMPCfg ):
    pass
class IGRISCAMPIMCfgPPO( IGRISCAMPCfgPPO ):
    runner_class_name = 'PIMOnPolicyRunner_AMP'
    class runner(IGRISCAMPCfgPPO.runner):
        algorithm_class_name = 'PIMPPO_AMP'
        policy_class_name = 'PIMActorCritic'