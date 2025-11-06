# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 min_std=None,
                 LCP_cfg: dict = {'use_LCP': False},
                 symmetry_cfg: dict = {'enforce_symmetry': False,},
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.use_LCP = LCP_cfg['use_LCP']
        self.smooth_coef = LCP_cfg['smooth_coef']
        self.LCP_cfg = LCP_cfg
        self.enforce_symmetry = symmetry_cfg['enforce_symmetry']
        self.symmetry_cfg = symmetry_cfg

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.mirror_transition = RolloutStorage.mirror_Transition()
        self.min_std = min_std

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def save_mirror_state(self, mirror_obs, mirror_critic_obs):
        self.mirror_transition.mirror_observations = mirror_obs
        self.mirror_transition.mirror_critic_observations = mirror_critic_obs

    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def process_mirror_step(self, mirror_next_critic_obs):
        self.mirror_transition.mirror_next_critic_observations = mirror_next_critic_obs
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_mirror_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, \
            mirror_obs_batch, mirror_critic_obs_batch, mirror_next_critic_obs_batch in generator:
                
                og_batch_size = obs_batch.shape[0]
                loss = 0.

                if self.enforce_symmetry:
                    target_mirrored_action_mean = self.mirror_actions(self.actor_critic.act_inference(obs_batch).detach())
                    mirrored_action_mean = self.actor_critic.act_inference(mirror_obs_batch)
                    mirror_loss = (mirrored_action_mean - target_mirrored_action_mean).pow(2).mean()
                    mean_mirror_loss += mirror_loss.item()

                    if self.symmetry_cfg['type']=='augmentation':
                        obs_batch = torch.cat((obs_batch, mirror_obs_batch), dim=0)
                        critic_obs_batch = torch.cat((critic_obs_batch, mirror_critic_obs_batch), dim=0)
                        actions_batch = torch.cat((actions_batch, self.mirror_actions(actions_batch)), dim=0)
                        next_critic_obs_batch = torch.cat((next_critic_obs_batch, mirror_next_critic_obs_batch), dim=0)
                        target_values_batch = torch.cat((target_values_batch, target_values_batch), dim=0)
                        advantages_batch = torch.cat((advantages_batch, advantages_batch), dim=0)
                        returns_batch = torch.cat((returns_batch, returns_batch), dim=0)
                        old_actions_log_prob_batch = torch.cat((old_actions_log_prob_batch, old_actions_log_prob_batch), dim=0)
                    elif self.symmetry_cfg['type']=='loss':
                        loss += mirror_loss
                    else:
                        t = self.symmetry_cfg['type']
                        print(f'Invalid symmetry type {t}')
                        raise NotImplementedError

                if self.use_LCP:
                    mask_idx = self.LCP_cfg['mask'] if 'mask' in self.LCP_cfg else []
                    lcp_obs_batch = obs_batch.clone() # for LCP loss
                    lcp_obs_batch.requires_grad_()
                    mask = torch.zeros_like(lcp_obs_batch)
                    mask[..., mask_idx] = 1
                    eff_lcp_obs_batch = lcp_obs_batch*(1-mask) + lcp_obs_batch.detach()*mask
                    # lcp_hidden_batch.requires_grad_()
                    self.actor_critic.act(eff_lcp_obs_batch)
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    gradient_penalty_loss = self._calc_grad_penalty(obs_batch=lcp_obs_batch, actions_log_prob_batch=actions_log_prob_batch)
                    mean_smooth_loss += gradient_penalty_loss.item()
                else:
                    gradient_penalty_loss = 0.
                    # GRAD PEN FOR SMOOTH MOTION from "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies"
                    # gradient_penalty_loss = self._calc_grad_penalty_2(
                    #     obs_batch    = lcp_obs_batch,
                    #     hidden_batch = lcp_hidden_batch,
                    #     log_prob     = actions_log_prob_batch
                    # )

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean[:og_batch_size]
                sigma_batch = self.actor_critic.action_std[:og_batch_size]
                entropy_batch = self.actor_critic.entropy[:og_batch_size]

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss += surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)


                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "symmetry": mean_mirror_loss,
        }
        return loss_dict

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch = None):
        # Compute gradients separately for obs_batch and history_batch
        grad_log_prob_obs = torch.autograd.grad(
            actions_log_prob_batch.sum(),
            obs_batch,
            create_graph=True,
        )[0]

        # Ensure gradients are not None
        assert grad_log_prob_obs is not None, "Gradient for obs_batch is None."

        # Calculate the gradient penalty loss
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob_obs), dim=-1).mean()            

        return gradient_penalty_loss
        
    def mirror_actions(self, actions: torch.Tensor):
        target_mirrored_action_mean = actions.detach().clone()
        with torch.no_grad():
            num_waist = self.symmetry_cfg['num_waist']
            num_legs = self.symmetry_cfg['num_legs']
            num_arms = self.symmetry_cfg['num_arms']

            start = 0
            num_section = num_waist
            end = start + num_section
            if num_waist:
                target_mirrored_action_mean[..., :end-1] *= -1
            # Lower body
            start = end
            num_section = num_legs
            end = start + num_section
            target_mirrored_action_mean[..., start:start+num_section//2] = actions[..., start+num_section//2:end]
            target_mirrored_action_mean[..., start+num_section//2:end] = actions[..., start:start+num_section//2]
            target_mirrored_action_mean[..., start+1:start+1+2] *= -1
            target_mirrored_action_mean[..., start+num_section//2-1] *= -1
            target_mirrored_action_mean[..., start+num_section//2+1:start+num_section//2+1+2] *= -1
            target_mirrored_action_mean[..., end-1] *= -1
            # Upper body
            if num_arms > 0:
                start = end
                num_section = num_arms
                end = start + num_section
                if num_arms:
                    target_mirrored_action_mean[..., start:start+num_section//2] = actions[..., start+num_section//2:end]
                    target_mirrored_action_mean[..., start+num_section//2:end] = actions[..., start:start+num_section//2]
                    target_mirrored_action_mean[..., start+1] *= -1
                    target_mirrored_action_mean[..., start+num_section//2+1] *= -1
                    if num_arms == 8: # shoulder yaw in action space
                        target_mirrored_action_mean[..., start+2] *= -1
                        target_mirrored_action_mean[..., start+num_section//2+2] *= -1

            assert end == actions.shape[-1], f'change params in mirror actions! action dim : {actions.shape[-1]}, mirror index end : {end}'

        return target_mirrored_action_mean
