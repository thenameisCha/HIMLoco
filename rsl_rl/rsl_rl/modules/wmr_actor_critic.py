import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic_recurrent import ActorCriticRecurrent, get_activation
from rsl_rl.utils import unpad_trajectories

class WMRActorCritic(ActorCriticRecurrent):
    def __init__(self, 
                 num_actor_obs, 
                 num_critic_obs, 
                 num_actions, 
                 actor_hidden_dims=..., 
                 critic_hidden_dims=..., 
                 activation='elu', 
                 rnn_type='lstm', 
                 rnn_hidden_size=256, 
                 rnn_num_layers=1, 
                 init_noise_std=1,
                 decoder_hidden_dims=[256,],
                 **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, rnn_type, rnn_hidden_size, rnn_num_layers, init_noise_std, **kwargs)
        self.decoderC = WMRDecoderC(num_input=rnn_hidden_size, num_output=num_critic_obs-2, hidden_dims=decoder_hidden_dims, activation=activation)
        self.decoderD = WMRDecoderD(num_input=rnn_hidden_size, num_output=2, hidden_dims=decoder_hidden_dims, activation=activation)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        reconstruction = torch.cat((self.decoderC(input_a.squeeze(0)), self.decoderD(input_a.squeeze(0))), dim=-1).detach()
        return super().act(reconstruction)

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        reconstruction = torch.cat((self.decoderC(input_a.squeeze(0)), self.decoderD(input_a.squeeze(0))), dim=-1).detach()
        return super().act_inference(reconstruction)
    
    def reconstruct(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return self.decoderC(input_a.squeeze(0)), self.decoderD(input_a.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
    
class WMRDecoderC(nn.Module):
    def __init__(
            self, 
            num_input, 
            num_output, 
            hidden_dims=[256,], 
            activation='elu',
            **kwargs):
        super(WMRDecoderC, self).__init__()
        activation = get_activation(activation)
        layers = []
        layers.append(activation)
        layers.append(nn.Linear(num_input, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], num_output))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
    
class WMRDecoderD(WMRDecoderC):
    def __init__(
            self, 
            num_input, 
            num_output, 
            hidden_dims=[256,], 
            activation='elu',
            **kwargs):
        super(WMRDecoderC, self).__init__()
        activation = get_activation(activation)
        layers = []
        layers.append(activation)
        layers.append(nn.Linear(num_input, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], num_output))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
