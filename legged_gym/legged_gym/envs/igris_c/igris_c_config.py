import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class IGRISCCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_one_step_observations = 51
        num_observations = num_one_step_observations * 10
        num_one_step_privileged_obs = 51 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 14
        num_waist = 2 # How many waist joints are in the action
        num_lower_actions = 12
        action_offset = 0 # Where does the action start in the joint list

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": -0.2,
            "Joint_Hip_Roll_Left": 0.0,
            "Joint_Hip_Yaw_Left": 0.,
            "Joint_Knee_Pitch_Left": 0.3,
            "Joint_Ankle_Pitch_Left": -0.15,
            "Joint_Ankle_Roll_Left": 0.0,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": -0.2,
            "Joint_Hip_Roll_Right": 0.0,
            "Joint_Hip_Yaw_Right": 0.0,
            "Joint_Knee_Pitch_Right": 0.3,
            "Joint_Ankle_Pitch_Right": -0.15,
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
            "Joint_Hip_Pitch_Left": 150,
            "Joint_Hip_Roll_Left": 150,
            "Joint_Hip_Yaw_Left": 100.0,
            "Joint_Knee_Pitch_Left": 150,
            "Joint_Ankle_Pitch_Left": 80.181,
            "Joint_Ankle_Roll_Left": 65.6563,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 100,
            "Joint_Hip_Roll_Right": 100,
            "Joint_Hip_Yaw_Right": 50.0,
            "Joint_Knee_Pitch_Right": 100,
            "Joint_Ankle_Pitch_Right": 80.181,
            "Joint_Ankle_Roll_Right": 65.6563,
            # Waist + Neck (5)
            # "Joint_Waist_Yaw": 100.0,
            "Joint_Waist_Roll": 458.206,
            "Joint_Waist_Pitch": 137.664,
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
            "Joint_Hip_Pitch_Left": 2.,
            "Joint_Hip_Roll_Left": 3.,
            "Joint_Hip_Yaw_Left": 1.5,
            "Joint_Knee_Pitch_Left": 1.,
            "Joint_Ankle_Pitch_Left": 3.5284,
            "Joint_Ankle_Roll_Left": 2.93288,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 2.,
            "Joint_Hip_Roll_Right": 3.,
            "Joint_Hip_Yaw_Right": 1.5,
            "Joint_Knee_Pitch_Right": 1.,
            "Joint_Ankle_Pitch_Right": 3.5284,
            "Joint_Ankle_Roll_Right": 2.93288,
            # # Waist + Neck (5)
            # "Joint_Waist_Yaw": 2.0,
            "Joint_Waist_Roll": 11.7824,
            "Joint_Waist_Pitch": 3.53993,
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
            1., 1.,
            1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1.,
            # 1., 1., 1., 2.,
            # 1., 1., 1., 2.,
        ]
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/igris/xml/igris_c_v2_waist.xml' 
        name = "igris"
        foot_name = "Ankle_Roll"
        terminate_after_contacts_on = ['Hip', 'base', 'Hand', 'Wrist']
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
                1.e-8, 
                1.e-8,
                1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8,
                1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8,
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
            # 0.,
            0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            # 0., 0., 0., 0., 
            # 0., 0., 0.,
            # 0., 0., 0., 0., 
            # 0., 0., 0.,
            # 0., 0.

        ]

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.95
        soft_dof_pos_limit = .95 # percentage of urdf limits, values above this limit are penalized
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        clearance_height_target = -0.8
        class scales:
            termination = -200.
            tracking_lin_vel = 5.0
            tracking_ang_vel = 2.5
            lin_vel_z = -1.
            joint_power = -1.e-3
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            action_rate = -0.01
            orientation = -2.
            dof_pos_limits = -5.
            dof_pos = 1.
            feet_air_time = 8.
            feet_contact_forces = -5.e-3
            stumble = -2.0
            feet_sliding = -0.25
            no_fly = -1.
            collision = -1.
            foot_clearance = -8.
            slow_touchdown = -0.1
            contact_power = -5.e-3
            # stand_still = -.5
            stand_still_vel = -.3
            stand_still_contact = 2.


    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]    # min max [rad/s]
            heading = [-3.14, 3.14]

class IGRISCCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.00
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
            'type' : 'augmentation',
            'symmetry_coef' : 10,
            'num_waist' : 2,
            'num_legs' : 12,
            'num_arms' : 0,
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

class IGRISCWBCfg( IGRISCCfg ):
    class env(IGRISCCfg.env):
        num_one_step_observations = 78
        num_observations = num_one_step_observations * 10
        num_one_step_privileged_obs = 78 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 23
        num_waist = 3 # How many waist joints are in the action
        num_lower_actions = 12
        action_offset = 0 # Where does the action start in the joint list

    class init_state(IGRISCCfg.init_state):
        default_joint_angles = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": -0.45,
            "Joint_Hip_Roll_Left": 0.0,
            "Joint_Hip_Yaw_Left": 0.,
            "Joint_Knee_Pitch_Left": 0.75,
            "Joint_Ankle_Pitch_Left": -0.38,
            "Joint_Ankle_Roll_Left": 0.0,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": -0.45,
            "Joint_Hip_Roll_Right": 0.0,
            "Joint_Hip_Yaw_Right": 0.0,
            "Joint_Knee_Pitch_Right": 0.75,
            "Joint_Ankle_Pitch_Right": -0.38,
            "Joint_Ankle_Roll_Right": 0.0,
            # Waist + Neck (5)
            "Joint_Waist_Yaw": 0.0,
            "Joint_Waist_Roll": 0.0,
            "Joint_Waist_Pitch": 0.0,
            # "Joint_Neck_Yaw": 0.0,
            # "Joint_Neck_Pitch": 0.0,
            # # Left arm (7)
            "Joint_Shoulder_Pitch_Left": 0.13,
            "Joint_Shoulder_Roll_Left": 0.13,
            "Joint_Shoulder_Yaw_Left": 0.,
            "Joint_Elbow_Pitch_Left": -0.3,
            # "Joint_Wrist_Yaw_Left": 0.0,
            # "Joint_Wrist_Roll_Left": 0.0,
            # "Joint_Wrist_Pitch_Left": 0.0,
            # # Right arm (7)
            "Joint_Shoulder_Pitch_Right": 0.13,
            "Joint_Shoulder_Roll_Right": -0.13,
            "Joint_Shoulder_Yaw_Right": -0.,
            "Joint_Elbow_Pitch_Right": -0.3,
            # "Joint_Wrist_Yaw_Right": 0.0,
            # "Joint_Wrist_Roll_Right": 0.0,
            # "Joint_Wrist_Pitch_Right": 0.0,
        }


    class control(IGRISCCfg.control):
        stiffness = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": 100,
            "Joint_Hip_Roll_Left": 100,
            "Joint_Hip_Yaw_Left": 50.0,
            "Joint_Knee_Pitch_Left": 100,
            "Joint_Ankle_Pitch_Left": 50,
            "Joint_Ankle_Roll_Left": 50,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 100,
            "Joint_Hip_Roll_Right": 100,
            "Joint_Hip_Yaw_Right": 50.0,
            "Joint_Knee_Pitch_Right": 100,
            "Joint_Ankle_Pitch_Right": 50,
            "Joint_Ankle_Roll_Right": 50,
            # Waist + Neck (5)
            "Joint_Waist_Yaw": 50.0,
            "Joint_Waist_Roll": 50.0,
            "Joint_Waist_Pitch": 50.0,
            # "Joint_Neck_Yaw": 30.0,
            # "Joint_Neck_Pitch": 30.0,
            # # Left arm (7)
            "Joint_Shoulder_Pitch_Left": 50.0,
            "Joint_Shoulder_Roll_Left": 50.,
            "Joint_Shoulder_Yaw_Left": 30.0,
            "Joint_Elbow_Pitch_Left": 25.0,
            # "Joint_Wrist_Yaw_Left": 10,
            # "Joint_Wrist_Roll_Left": 10,
            # "Joint_Wrist_Pitch_Left": 10,
            # # Right arm (7)
            "Joint_Shoulder_Pitch_Right": 50.0,
            "Joint_Shoulder_Roll_Right": 50.,
            "Joint_Shoulder_Yaw_Right": 30.0,
            "Joint_Elbow_Pitch_Right": 25.0,
            # "Joint_Wrist_Yaw_Right": 10.0,
            # "Joint_Wrist_Roll_Right": 10.0,
            # "Joint_Wrist_Pitch_Right": 10.0,
        }
        damping = {
            # Left leg (6)
            "Joint_Hip_Pitch_Left": 4.0,
            "Joint_Hip_Roll_Left": 4.0,
            "Joint_Hip_Yaw_Left": 2.0,
            "Joint_Knee_Pitch_Left": 4.0,
            "Joint_Ankle_Pitch_Left": 2.5,
            "Joint_Ankle_Roll_Left": 2.5,
            # Right leg (6)
            "Joint_Hip_Pitch_Right": 4.0,
            "Joint_Hip_Roll_Right": 4.0,
            "Joint_Hip_Yaw_Right": 2.0,
            "Joint_Knee_Pitch_Right": 4.0,
            "Joint_Ankle_Pitch_Right": 2.5,
            "Joint_Ankle_Roll_Right": 2.5,
            # # Waist + Neck (5)
            "Joint_Waist_Yaw": 2.0,
            "Joint_Waist_Roll": 2.0,
            "Joint_Waist_Pitch": 2.0,
            # "Joint_Neck_Yaw": 1.0,
            # "Joint_Neck_Pitch": 1.0,
            # # Left arm (7)
            "Joint_Shoulder_Pitch_Left": 2.,
            "Joint_Shoulder_Roll_Left": 2.,
            "Joint_Shoulder_Yaw_Left": 1.5,
            "Joint_Elbow_Pitch_Left": 1.3,
            # "Joint_Wrist_Yaw_Left": .5,
            # "Joint_Wrist_Roll_Left": .5,
            # "Joint_Wrist_Pitch_Left": .5,
            # # Right arm (7)
            "Joint_Shoulder_Pitch_Right": 2.,
            "Joint_Shoulder_Roll_Right": 2.,
            "Joint_Shoulder_Yaw_Right": 1.5,
            "Joint_Elbow_Pitch_Right": 1.3,
            # "Joint_Wrist_Yaw_Right": .5,
            # "Joint_Wrist_Roll_Right": .5,
            # "Joint_Wrist_Pitch_Right": .5,
        }


        # action scale: target angle = actionScale * action + defaultAngle

        action_scale = [
            1.,
            1., 1.,
            1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
        ]

    class rewards( IGRISCCfg.rewards ):
        class scales( IGRISCCfg.rewards.scales ):
            centroidal_momentum = 1.

    class asset(IGRISCCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/igris/xml/igris_c_v2_wholebody.xml' 
        armature = [
                    0.0307, 
                    0.0614, # 0.0307 
                    0.0614,  # 0.0307
                    0.0521, 0.0786, 0.0307, 0.0521, 0.0598, 0.0598,  # 0.0299
                    0.0521, 0.0786, 0.0307, 0.0521, 0.0598, 0.0598,  # 0.0299
                    0.0307, 
                    0.0307, 
                    0.0307, 
                    0.0307,  # Left arm (shoulder+elbow)
                    # 0.01433, 0.01433, 0.01433,  # Left wrist
                    0.0307, 
                    0.0307, 
                    0.0307, 
                    0.0307,  # Right arm (shoulder+elbow)
                    # 0.01433, 0.01433, 0.01433,  # Right wrist
                    # 0.2488, 0.2488  # Neck joints
            ]
        
        damping = [
                1.e-6, 
                1.e-6, 
                1.e-6,
                1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6,
                1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6,
                1.e-6, 
                1.e-6, 
                1.e-6, 
                1.e-6, 
                # 0.1, 0.1, 0.1
                1.e-6, 
                1.e-6, 
                1.e-6, 
                1.e-6, 
                # 0.1, 0.1, 0.1
            ] # 0.1
        
        frictionloss = [
            0.,
            0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 
            # 0., 0., 0.,
            0., 0., 0., 0., 
            # 0., 0., 0.,
            # 0., 0.

        ]

class IGRISCWBCfgPPO( IGRISCCfgPPO ):
    class algorithm(IGRISCCfgPPO.algorithm):
        # symmetry loss
        symmetry_cfg = {
            'enforce_symmetry' : True,
            'type' : 'augmentation', # augmentation
            'symmetry_coef' : 10,
            'num_waist' : 3,
            'num_legs' : 12,
            'num_arms' : 8,
        }

class IGRISCPIMCfg( IGRISCCfg ):
    pass
class IGRISCPIMCfgPPO( IGRISCCfgPPO ):
    runner_class_name = 'PIMOnPolicyRunner'
    class runner(IGRISCCfgPPO.runner):
        algorithm_class_name = 'PIMPPO'
        policy_class_name = 'PIMActorCritic'