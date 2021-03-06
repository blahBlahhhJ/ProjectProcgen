import argparse
import numpy as np
import torch
from procgen import ProcgenEnv

from dqn.q_agent import QAgent
from ppo.ppo_agent import PPOAgent

from ppo.ppo import ImpalaPPO

from utils.vec_envs import VecExtractDictObs, VecNormalize, VecMonitor, VecFrameStack

from matplotlib import pyplot as plt

# gym config
ENV_NAME = 'fruitbot'
NUM_ENVS = 64
NUM_LEVELS = 50
START_LEVEL = 500

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
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--stack', type=int, default=1)
    parser.add_argument('--flare', action='store_true')
    parser.add_argument('--mixreg', action='store_true')

    env_group = parser.add_argument_group("Env Args")
    env_group.add_argument('--env_name', type=str, default=ENV_NAME)
    env_group.add_argument('--num_envs', type=int, default=NUM_ENVS)
    env_group.add_argument('--num_levels', type=int, default=NUM_LEVELS)
    env_group.add_argument('--start_level', type=int, default=START_LEVEL)

    agent_group = parser.add_argument_group("Agent Args")
    PPOAgent.add_to_argparse(agent_group)

    model_group = parser.add_argument_group("Model Args")
    ImpalaPPO.add_to_argparse(model_group)

    return parser

def main():
    """
        Run experiemnt.
    """
    parser = _setup_parser()
    args = parser.parse_args()

    if args.flare:
        num_channels = 3
    else:
        num_channels = 3 * args.stack

    data_config = {'input_dims': (num_channels, 64, 64), 'num_classes': 15}

    env = ProcgenEnv(
        num_envs=args.num_envs, 
        env_name=args.env_name, 
        num_levels=args.num_levels, 
        start_level=args.start_level, 
        distribution_mode='easy', 
        render_mode='rgb_array'
    )
    env = VecExtractDictObs(env, 'rgb')
    env = VecMonitor(env)
    env = VecNormalize(env, ob=False)
    if args.stack > 1:
        env = VecFrameStack(env, args.stack)
    
    model = ImpalaPPO(data_config, args)
    
    agent = PPOAgent(env, model, data_config, args)
    if args.eval_model:
        agent.load(args.eval_model)
        agent.test(num_eval=5)
    else:
        agent.train()


# the main script
if __name__ == '__main__':
    main()
