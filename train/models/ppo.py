"""
    All the networks for Proximity Policy Optimization
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO(nn.Module):
    def __init__(self, data_config, args):
        super().__init__()
        