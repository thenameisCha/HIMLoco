import glob
from legged_gym.envs.igris_c.igris_c_config import IGRISCCfg, IGRISCCfgPPO

class IGRISCAMPCfg( IGRISCCfg ):
    class env( IGRISCCfg.env ):
        reference_state_initialization = True
        reference_state_initialization_prob = .8
        amp_motion_files = {
            # "path/to/pkl": {
            #     "hz": motion_hz,
            #     "start_time": clip_motion_start [s],
            #     "end_time": clip_motion_end [s],
            #     "weight": motion_weight
            # },
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
        amp_task_reward_lerp = .3
        amp_discr_hidden_dims = [128,]

        LOG_WANDB = True
        env_name = 'igris_c_AMP'
        file_name = 'igris_c_AMP'
        config_name = 'igris_c_AMP'
        wandb_name = 'igris_c_AMP'
