import torch
import torch.nn as nn

class Normalizer_obs(torch.nn.Module):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0, device='cuda:0'):
        super(Normalizer_obs, self).__init__()
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.register_buffer("running_mean", torch.zeros(input_dim, device=device))
        self.register_buffer("running_var", torch.ones(input_dim, device=device))
        self.register_buffer("count", torch.tensor(epsilon, device=device))
        self.input_dim = input_dim

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

        self.running_mean[:] = new_mean
        self.running_var[:] = m2 / tot_count
        self.count = tot_count

    def update_normalizer(self, inputs: torch.Tensor):
        batch_mean = inputs.mean(dim=0)
        batch_var = inputs.var(dim=0)
        batch_count = inputs.size(dim=0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def forward(self, inputs: torch.Tensor):
        original_shape = inputs.shape
        inputs = inputs.view((-1, self.input_dim))
        if self.training:
            self.update_normalizer(inputs)
        mean_adjusted = self.running_mean
        var_adjusted = self.running_var + self.epsilon
        normalized = (inputs - mean_adjusted) / var_adjusted.sqrt()
        return torch.clamp(normalized, -self.clip_obs, self.clip_obs).view(original_shape)
