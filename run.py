import argparse
import numpy as np
import torch

from models.dqn import DQN
from agents.q_agent import QAgent

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

    model_group = parser.add_argument_group("Model Args")
    DQN.add_to_argparse(model_group)

    agent_group = parser.add_argument_group("Agent Args")
    QAgent.add_to_argparse(agent_group)

    return parser

def main():
    """
        Run experiemnt.
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_config = {'input_dims': (3, 64, 64), 'num_classes': 15}
    
    model = DQN(data_config, args)
    agent = QAgent(model, args)
    agent.evaluate(render=True)


# the main script
if __name__ == '__main__':
    main()
