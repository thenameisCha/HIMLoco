import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class IGRISCCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_one_step_observations = 51
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 51 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 14
        num_waist = 2 # How many waist joints are in the action
        num_lower_actions = 12
        action_offset = 0 # Where does the action start in the joint list

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": -0.1,
            "Joint_Hip_Roll_Left": 0.0,
            "Joint_Hip_Yaw_Left": 0.,
            "Joint_Knee_Pitch_Left": 0.3,
            "Joint_Ankle_Pitch_Left": -0.2,
            "Joint_Ankle_Roll_Left": 0.0,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": -0.1,
            "Joint_Hip_Roll_Right": 0.0,
            "Joint_Hip_Yaw_Right": 0.0,
            "Joint_Knee_Pitch_Right": 0.3,
            "Joint_Ankle_Pitch_Right": -0.2,
            "Joint_Ankle_Roll_Right": 0.0,
            # Waist + Neck (5)
            # "Joint_Waist_Yaw": 0.0,
            "Joint_Waist_Roll": 0.0,
            "Joint_Waist_Pitch": 0.0,
            # "Joint_Neck_Yaw": 0.0,
            # "Joint_Neck_Pitch": 0.0,
            # # Left arm (7)
            # "Joint_Shoulder_Pitch_Left": 0.13,
            # "Joint_Shoulder_Roll_Left": 0.13,
            # "Joint_Shoulder_Yaw_Left": 0.,
            # "Joint_Elbow_Pitch_Left": -0.3,
            # "Joint_Wrist_Yaw_Left": 0.0,
            # "Joint_Wrist_Roll_Left": 0.0,
            # "Joint_Wrist_Pitch_Left": 0.0,
            # # Right arm (7)
            # "Joint_Shoulder_Pitch_Right": 0.13,
            # "Joint_Shoulder_Roll_Right": -0.13,
            # "Joint_Shoulder_Yaw_Right": -0.,
            # "Joint_Elbow_Pitch_Right": -0.3,
            # "Joint_Wrist_Yaw_Right": 0.0,
            # "Joint_Wrist_Roll_Right": 0.0,
            # "Joint_Wrist_Pitch_Right": 0.0,
        }


    class control(LeggedRobotCfg.control):
        stiffness = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": 50,
            "Joint_Hip_Roll_Left": 50,
            "Joint_Hip_Yaw_Left": 12.0,
            "Joint_Knee_Pitch_Left": 30,
            "Joint_Ankle_Pitch_Left": 7,
            "Joint_Ankle_Roll_Left": 5,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 50,
            "Joint_Hip_Roll_Right": 50,
            "Joint_Hip_Yaw_Right": 12.0,
            "Joint_Knee_Pitch_Right": 30,
            "Joint_Ankle_Pitch_Right": 7,
            "Joint_Ankle_Roll_Right": 5,
            # Waist + Neck (5)
            # "Joint_Waist_Yaw": 100.0,
            "Joint_Waist_Roll": 20.0,
            "Joint_Waist_Pitch": 20.0,
            # "Joint_Neck_Yaw": 30.0,
            # "Joint_Neck_Pitch": 30.0,
            # # Left arm (7)
            # "Joint_Shoulder_Pitch_Left": 50.0,
            # "Joint_Shoulder_Roll_Left": 50.,
            # "Joint_Shoulder_Yaw_Left": 30.0,
            # "Joint_Elbow_Pitch_Left": 25.0,
            # "Joint_Wrist_Yaw_Left": 10,
            # "Joint_Wrist_Roll_Left": 10,
            # "Joint_Wrist_Pitch_Left": 10,
            # # Right arm (7)
            # "Joint_Shoulder_Pitch_Right": 50.0,
            # "Joint_Shoulder_Roll_Right": 50.,
            # "Joint_Shoulder_Yaw_Right": 30.0,
            # "Joint_Elbow_Pitch_Right": 25.0,
            # "Joint_Wrist_Yaw_Right": 10.0,
            # "Joint_Wrist_Roll_Right": 10.0,
            # "Joint_Wrist_Pitch_Right": 10.0,
        }
        damping = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": 2.5,
            "Joint_Hip_Roll_Left": 2.5,
            "Joint_Hip_Yaw_Left": 0.7,
            "Joint_Knee_Pitch_Left": 2.0,
            "Joint_Ankle_Pitch_Left": 0.8,
            "Joint_Ankle_Roll_Left": 0.7,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 2.5,
            "Joint_Hip_Roll_Right": 2.5,
            "Joint_Hip_Yaw_Right": 0.7,
            "Joint_Knee_Pitch_Right": 2.0,
            "Joint_Ankle_Pitch_Right": 0.8,
            "Joint_Ankle_Roll_Right": 0.7,
            # # Waist + Neck (5)
            # "Joint_Waist_Yaw": 2.0,
            "Joint_Waist_Roll": 0.6,
            "Joint_Waist_Pitch": 0.6,
            # "Joint_Neck_Yaw": 1.0,
            # "Joint_Neck_Pitch": 1.0,
            # # Left arm (7)
            # "Joint_Shoulder_Pitch_Left": 2.,
            # "Joint_Shoulder_Roll_Left": 2.,
            # "Joint_Shoulder_Yaw_Left": 1.5,
            # "Joint_Elbow_Pitch_Left": 1.3,
            # "Joint_Wrist_Yaw_Left": .5,
            # "Joint_Wrist_Roll_Left": .5,
            # "Joint_Wrist_Pitch_Left": .5,
            # # Right arm (7)
            # "Joint_Shoulder_Pitch_Right": 2.,
            # "Joint_Shoulder_Roll_Right": 2.,
            # "Joint_Shoulder_Yaw_Right": 1.5,
            # "Joint_Elbow_Pitch_Right": 1.3,
            # "Joint_Wrist_Yaw_Right": .5,
            # "Joint_Wrist_Roll_Right": .5,
            # "Joint_Wrist_Pitch_Right": .5,
        }


        # action scale: target angle = actionScale * action + defaultAngle

        action_scale = [
            3., 3.,
            3., 3., 5., 5., 10., 10.,
            3., 3., 5., 5., 10., 10.,
            # 1., 1., 1., 2.,
            # 1., 1., 1., 2.,
        ]
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/igris/xml/igris_c_v2_waist.xml' 
        name = "igris"
        foot_name = "Ankle_Roll"
        terminate_after_contacts_on = ['Hip', 'base', 'Hand', 'Wrist', 'Knee']
        armature = [
                    # 0.0307, 
                    0.0614, # 0.0307 
                    0.0614,  # 0.0307
                    0.0521, 0.0786, 0.0307, 0.0521, 0.0598, 0.0598,  # 0.0299
                    0.0521, 0.0786, 0.0307, 0.0521, 0.0598, 0.0598,  # 0.0299
                    # 0.0307, 
                    # 0.0307, 
                    # 0.0307, 
                    # 0.0307,  # Left arm (shoulder+elbow)
                    # 0.01433, 0.01433, 0.01433,  # Left wrist
                    # 0.0307, 
                    # 0.0307, 
                    # 0.0307, 
                    # 0.0307,  # Right arm (shoulder+elbow)
                    # 0.01433, 0.01433, 0.01433,  # Right wrist
                    # 0.2488, 0.2488  # Neck joints
            ]
        
        damping = [
                # 1.e-6, 
                1.e-6, 
                1.e-6,
                1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6,
                1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6,
                # 1.e-6, 
                # 1.e-6, 
                # 1.e-6, 
                # 1.e-6, 
                # 0.1, 0.1, 0.1
                # 1.e-6, 
                # 1.e-6, 
                # 1.e-6, 
                # 1.e-6, 
                # 0.1, 0.1, 0.1
            ] # 0.1
        
        frictionloss = [
            # 0.54,
            0.54,
            0.54,
            
            2.4,
            0.812,
            0.54,
            2.4,
            1.81,
            1.81,

            2.4,
            0.812,
            0.54,
            2.4,
            1.81,
            1.81,

            # 0.54,
            # 0.54,
            # 0.54,
            # 0.54,
            # 0.264,
            # 0.264,
            # 0.264,

            # 0.54,
            # 0.54,
            # 0.54,
            # 0.54,
            # 0.264,
            # 0.264,
            # 0.264,

            # 0.356,
            # 0.356
        ]

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.95
        soft_dof_pos_limit = .95 # percentage of urdf limits, values above this limit are penalized
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            action_rate = -0.01
            smoothness = -0.01
            joint_power = -2.e-5
            contact_power = -1.e-2
            slow_touchdown = -0.2
            no_fly = 0.25
            stand_still = -0.5
            # upper_regularization = 2.
            # centroidal_momentum = 1.


    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        heading_command = True # if true: compute ang vel command from heading error
        default_ankle_height = 0.072 # height from sole to ankle
        default_feet_width = 0.22
        default_apex_clearance = 0.06
        class ranges:
            lin_vel_x = [0.5, 1.0] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]    # min max [rad/s]
            heading = [-3.14, 3.14]

class IGRISCCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # LCP loss
        LCP_cfg = {
            'use_LCP': True,
            'smooth_coef': 1.e-3,
            'mask': [
            ]
        }
        # symmetry loss
        symmetry_cfg = {
            'enforce_symmetry' : True,
            'symmetry_coef' : 50,
            'num_waist' : 2,
            'num_legs' : 12,
            'num_arms' : 8,
        }
    
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'igris_c' # should be the same as 'env' in env.py and env_config.py 
        algorithm_class_name = 'HIMPPO'
        policy_class_name = 'HIMActorCritic'

        min_normalized_std = .1
        LOG_WANDB = True
        env_name = 'igris_c'
        file_name = 'igris_c'
        config_name = 'igris_c'
        wandb_name = 'igris_c'
