import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt

from dqn.q_agent import QAgent
from ppo.ppo_agent import PPOAgent

np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """
        Set up an argument parser using Python's ArgumentParser for easier experimenting.
        Returns
        -------
            argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=True)

    agent_group = parser.add_argument_group("Agent Args")
    PPOAgent.add_to_argparse(agent_group)

    return parser

def main():
    """
        Run experiemnt.
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_config = {'input_dims': (3, 64, 64), 'num_classes': 15}
    
    agent = PPOAgent(data_config, args)
    # agent.evaluate(render=True)
    # states, returns, actions, values, neglogpacs, = agent.gather_trajectory()
    # print(states.shape, returns.shape, actions.shape, values.shape, neglogpacs.shape)
    # agent.train_step(states, returns, actions, values, neglogpacs)
    agent.train()

    


# the main script
if __name__ == '__main__':
    main()
