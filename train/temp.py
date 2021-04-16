import torch
import numpy as np

states = torch.rand((32, 3, 64, 64))
returns = torch.rand((32, 1))
actions = torch.randperm(32).unsqueeze(1)
values = torch.rand((32, 1))
logprobs = torch.rand(32, 4)

print(states.shape, returns.shape, actions.shape, values.shape, logprobs.shape)

batch_size = states.shape[0]
coef = torch.FloatTensor(np.random.beta(0.2, 0.2, size=batch_size))
seq_indices = torch.arange(batch_size)
rand_indices = torch.randperm(batch_size)
indices = torch.where(coef > 0.5, seq_indices, rand_indices)
other_indices = torch.where(coef > 0.5, rand_indices, seq_indices)
coef = torch.where(coef > 0.5, coef, 1 - coef).unsqueeze(1)

print(indices.shape, other_indices.shape, coef.shape)

mix_states = coef.unsqueeze(2).unsqueeze(3) * states[indices, :, :, :] + (1 - coef).unsqueeze(2).unsqueeze(3) * states[other_indices, :, :, :]
mix_returns = coef * returns[indices, :] + (1 - coef) * returns[other_indices, :]
mix_actions = actions[indices, :]
mix_values = coef * values[indices, :] + (1 - coef) * values[other_indices, :]
mix_logprobs = coef * logprobs[indices, :] + (1 - coef) * logprobs[other_indices, :]

print(mix_states.shape, mix_returns.shape, mix_actions.shape, mix_values.shape, mix_logprobs.shape)
