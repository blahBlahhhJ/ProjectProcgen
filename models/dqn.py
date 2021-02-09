import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_DIMS = [32, 64, 64]
CONV_KERNELS = [7, 5, 3]
CONV_STRIDES = [4, 2, 1]
FC_DIMS = [512]

class DQN(nn.Module):
    """
        Simple Deep Q Network for predicting the Q value for each actions.
    """
    def __init__(self, data_config, args):
        super().__init__()
        C, H, W = data_config['input_dims']
        num_classes = data_config['num_classes']

        self.args = vars(args) if args is not None else {}
        conv_dims = self.args.get('conv_dims', CONV_DIMS)
        conv_kernels = self.args.get('conv_kernels', CONV_KERNELS)
        conv_strides = self.args.get('conv_strides', CONV_STRIDES)
        fc_dims = self.args.get('fc_dims', FC_DIMS)

        self.conv1 = nn.Conv2d(
            C, conv_dims[0], 
            kernel_size=conv_kernels[0],
            stride=conv_strides[0]
        )
        # conv math
        H = (H - conv_kernels[0] + conv_strides[0]) // conv_strides[0]
        W = (W - conv_kernels[0] + conv_strides[0]) // conv_strides[0]
        self.conv2 = nn.Conv2d(
            conv_dims[0], conv_dims[1], 
            kernel_size=conv_kernels[1],
            stride=conv_strides[1]
        )
        # conv math
        H = (H - conv_kernels[1] + conv_strides[1]) // conv_strides[1]
        W = (W - conv_kernels[1] + conv_strides[1]) // conv_strides[1]
        self.conv3 = nn.Conv2d(
            conv_dims[1], conv_dims[2], 
            kernel_size=conv_kernels[2],
            stride=conv_strides[2]
        )
        # conv math
        H = (H - conv_kernels[2] + conv_strides[2]) // conv_strides[2]
        W = (W - conv_kernels[2] + conv_strides[2]) // conv_strides[2]
        self.fc1 = nn.Linear(conv_dims[2] * H * W, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, image_height, image_width) tensor
        Returns
        -------
            torch.Tensor: (batch_size, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dims", type=list, default=CONV_DIMS)
        parser.add_argument("--conv_kernels", type=list, default=CONV_KERNELS)
        parser.add_argument("--conv_strides", type=list, default=CONV_STRIDES)
        parser.add_argument("--fc_dims", type=list, default=FC_DIMS)
        return parser


# simple test for the network
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    DQN.add_to_argparse(parser)
    data_config = {'input_dims': (3, 128, 128), 'num_classes': 4}
    args = parser.parse_args()
    nnet = DQN(data_config, args)
    x = torch.randn(5, 3, 128, 128)
    y = nnet(x)
    print(nnet)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    assert y.shape == (5, 4)

