"""
    All the networks for Deep Q Learning.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.common.model_utils import ImpalaBlock

CONV_DIMS = [16, 32, 32]
CONV_KERNELS = [7, 5, 3]
CONV_STRIDES = [4, 2, 1]
FC_DIMS = [256]

class NatureDQN(nn.Module):
    """
        Simple Deep Q Network for predicting the Q value for each actions.
        from Mnih in the famous Nature paper.
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
        Args
        ------
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
        parser.add_argument("--conv_dims", type=int, nargs='+', default=CONV_DIMS)
        parser.add_argument("--conv_kernels", type=int, nargs='+', default=CONV_KERNELS)
        parser.add_argument("--conv_strides", type=int, nargs='+', default=CONV_STRIDES)
        parser.add_argument("--fc_dims", type=int, nargs='+', default=FC_DIMS)
        return parser




class ImpalaDQN(nn.Module):
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
        self.fc2 = nn.Linear(fc_dims[0], num_classes)

        nn.init.orthogonal_(self.fc2.weight.data)
        nn.init.constant_(self.fc2.bias.data, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dims", type=int, nargs='+', default=CONV_DIMS)
        parser.add_argument("--fc_dims", type=int, nargs='+', default=FC_DIMS)
        return parser


# simple test for the network
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    NatureDQN.add_to_argparse(parser)
    data_config = {'input_dims': (3, 64, 64), 'num_classes': 15}
    args = parser.parse_args()
    nnet = ImpalaDQN(data_config, args)
    x = torch.randn(5, 3, 64, 64)
    y = nnet(x)
    print(nnet)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    assert y.shape == (5, 15)
