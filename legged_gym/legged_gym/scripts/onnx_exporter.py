from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, get_load_path, task_registry
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from rsl_rl.modules import ActorCritic, HIMActorCritic, Normalizer_obs

import numpy as np
import torch

def export_ONNX(args):

    device = 'cuda:0'

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    args.headless = True
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    # train_cfg = class_to_dict(train_cfg)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"Loading model from: {resume_path}")
    actor_critic_class = eval(train_cfg.runner.policy_class_name) 
    num_obs = env_cfg.env.num_observations
    num_critic_obs = env_cfg.env.num_privileged_obs
    num_actions = env_cfg.env.num_actions
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    actor_critic = actor_critic_class( num_actor_obs=num_obs,
                                        num_critic_obs=num_critic_obs,
                                        num_one_step_obs=env_cfg.env.num_one_step_observations,
                                        num_actions=num_actions,
                                        **policy_cfg).to(device)
    normalizer_obs = Normalizer_obs(num_critic_obs)
    normalizer_obs.eval()
    loaded_dict = torch.load(resume_path)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    normalizer_obs.load_state_dict(loaded_dict['obs_normalizer_state_dict'])

    if 'HIMActorCritic' in train_cfg.runner.policy_class_name:
        obs = torch.zeros(1, num_obs, device=device)
        critic_obs = torch.zeros(1, num_critic_obs, device=device)
        torch.onnx.export(
            actor_critic,
            (obs,),
            "/home/robros/isaac_ws/HIMLoco/legged_gym/logs/onnx/actor.onnx",
            input_names=["obs",],
            output_names=["action",],
            # output_names=["action", "hn"],
            opset_version=11,
            do_constant_folding=True
        )
        torch.onnx.export(
            normalizer_obs,
            (critic_obs),
            "/home/robros/isaac_ws/HIMLoco/legged_gym/logs/onnx/normalizer.onnx",
            input_names=["obs"],
            output_names=["normalized_obs"],
            opset_version=11,
            do_constant_folding=True
        )
    elif train_cfg.runner.policy_class_name == 'ActorCritic':
        obs = torch.zeros(1, num_obs, device=device)
        critic_obs = torch.zeros(1, num_critic_obs, device=device)
        torch.onnx.export(
            actor_critic,
            (obs,),
            "/home/robros/isaac_ws/HIMLoco/legged_gym/logs/onnx/actor.onnx",
            input_names=["obs",],
            output_names=["action",],
            # output_names=["action", "hn"],
            opset_version=11,
            do_constant_folding=True
        )
        torch.onnx.export(
            normalizer_obs,
            (critic_obs),
            "/home/robros/isaac_ws/HIMLoco/legged_gym/logs/onnx/normalizer.onnx",
            input_names=["obs"],
            output_names=["normalized_obs"],
            opset_version=11,
            do_constant_folding=True
        )
    else:
        print("NOT A RECOGNIZED POLICY TYPE!!")
        return
    print("ONNX EXPORT COMPLETE.")

if __name__ == '__main__':
    args = get_args()
    export_ONNX(args)