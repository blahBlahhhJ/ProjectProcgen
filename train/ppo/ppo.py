"""
    All the networks for Proximity Policy Optimization
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ImpalaBlock

CONV_DIMS = [16, 32, 32]
FC_DIMS = [256]

class ImpalaPPO(nn.Module):
    def __init__(self, data_config, args):
        super().__init__()
        C, H, W = data_config['input_dims']
        num_classes = data_config['num_classes']

        self.args = vars(args) if args is not None else {}
        conv_dims = self.args.get('conv_dims', CONV_DIMS)
        fc_dims = self.args.get('fc_dims', FC_DIMS)

        self.block1 = ImpalaBlock(C, conv_dims[0])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        self.block2 = ImpalaBlock(conv_dims[0], conv_dims[1])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        self.block3 = ImpalaBlock(conv_dims[1], conv_dims[2])
        H = (H - 2 - 1) // 2
        W = (W - 2 - 1) // 2
        self.fc1 = nn.Linear(conv_dims[2] * H * W, fc_dims[0])
        self.actor = nn.Linear(fc_dims[0], num_classes)
        self.critic = nn.Linear(fc_dims[0], 1)

        nn.init.orthogonal_(self.actor.weight.data)
        nn.init.constant_(self.actor.bias.data, 0)
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        a = F.softmax(self.actor(x), dim=-1)
        c = self.critic(x)
        return a, c

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dims", type=int, nargs='+', default=CONV_DIMS)
        parser.add_argument("--fc_dims", type=int, nargs='+', default=FC_DIMS)
        return parser
        