import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms.ppo import PPO

class WMR_PPO( PPO ):
    def __init__(self, 
                 actor_critic, 
                 num_learning_epochs=1, 
                 num_mini_batches=1, 
                 clip_param=0.2, 
                 gamma=0.998, 
                 lam=0.95, 
                 value_loss_coef=1, 
                 entropy_coef=0, 
                 learning_rate=0.001, 
                 max_grad_norm=1, 
                 use_clipped_value_loss=True, 
                 schedule="fixed", 
                 desired_kl=0.01, 
                 device='cpu', 
                 min_std=None, 
                 LCP_cfg = { 'use_LCP': False }, 
                 symmetry_cfg = { 'enforce_symmetry': False }):
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device, min_std, LCP_cfg, symmetry_cfg)
