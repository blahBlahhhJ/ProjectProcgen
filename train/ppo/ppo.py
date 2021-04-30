"""
    All the networks for Proximal Policy Optimization
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ImpalaBlock

CONV_DIMS = [16, 32, 32]
FC_DIMS = [256]

class ImpalaPPO(nn.Module):
    """
        The model for PPO.
        input
          |
        impalablock
          |
        impalablock
          |
        impalablock
          |
        flatten
          |
        dense ___
          |      |
        dense   dense
          |      |
        actor   critic
    """
    def __init__(self, data_config, args):
        super().__init__()
        C, H, W = data_config['input_dims']
        num_classes = data_config['num_classes']

        self.args = args
        conv_dims = self.args.conv_dims
        fc_dims = self.args.fc_dims

        self.block1 = ImpalaBlock(C, conv_dims[0])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        self.block2 = ImpalaBlock(conv_dims[0], conv_dims[1])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        self.block3 = ImpalaBlock(conv_dims[1], conv_dims[2])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        fc_input_dim = conv_dims[2] * H * W
        if self.args.flare:
            fc_input_dim *= 2 * (self.args.stack - 1)
            self.ln = nn.LayerNorm(fc_dims[0])
        
        self.fc1 = nn.Linear(fc_input_dim, fc_dims[0])
        self.actor = nn.Linear(fc_dims[0], num_classes)
        self.critic = nn.Linear(fc_dims[0], 1)

        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)
        nn.init.orthogonal_(self.critic.weight.data, gain=1)
        nn.init.constant_(self.critic.bias.data, 0)

    def forward(self, x, coef=None, rand_indices=None):
        if self.args.flare:
            B, SC, H, W = x.shape # (batch_size, stack_size * num_channels, height, width)
            S = self.args.stack
            C = SC // S

            x = x.reshape(B * S, C, H, W) # (batch_size * stack_size, ...image_dim)
            x = self.encode(x).reshape(B, S, -1)  # (batch_size, stack_size, latent_size)
            diff = x[:, 1:] - x[:, :-1].detach() # (batch_size, stack_size - 1, latent_size)
            x = torch.cat([x[:, 1:], diff], axis=1).reshape(B, -1) # (batch_size, 2 * (stack_size - 1) * latent_size)
        else:
            x = self.encode(x, coef, rand_indices)

        a, c = self.decode(x)
        return a, c

    def encode(self, x, coef=None, rand_indices=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if coef is not None:
          x = coef * x + (1 - coef) * x[rand_indices, :, :, :]
        x = torch.flatten(x, 1)
        return x

    def decode(self, x):
        x = self.fc1(x)
        if self.args.flare:
            x = self.ln(x)
        else:
            x = F.relu(x)
        
        a = F.log_softmax(self.actor(x), dim=-1)
        c = self.critic(x)
        return a, c

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dims", type=int, nargs='+', default=CONV_DIMS)
        parser.add_argument("--fc_dims", type=int, nargs='+', default=FC_DIMS)
        return parser
