import argparse
import numpy as np
import torch
import gym3
from procgen import ProcgenGym3Env

from .utils.schedule import LinearSchedule
from .utils.wrapper import RgbWrapper

# gym config
ENV_NAME = 'fruitbot'
NUM_ENVS = 1
NUM_LEVELS = 50
START_LEVEL = 500

# training config
TRAIN_STEPS = 4e6
TRAIN_RESUME = 0
LR_START = 2.5e-4
LR_END = 5e-5
LR_STEPS = 2e6
EPS_START = 1
EPS_END = 0.1
EPS_STEPS = 1e6


class QAgent():
    """
        The agent that performs Q-Learning.
    """
    def __init__(self, model, args):
        self.env = RgbWrapper(ProcgenGym3Env(
            args.num_envs, args.env_name, 
            num_levels=args.num_levels, 
            start_level=args.start_level, 
            distribution_mode='easy', 
            render_mode='rgb_array'
        ))
        self.model = model
        self.config = args
        self.step = self.config.train_resume
        self.eps_schedule = LinearSchedule(
            self.config.eps_start, self.config.eps_end, self.config.eps_steps
        )


    def _get_q_values(self, s):
        """
            Compute Q-values for all actions.
            Args
            ------
                s: the current state/frame of the game with shape (B, C, H, W)
            Returns
            -------
                (A, ) Q-values of all actions
        """
        return self.model(s)

    def get_action(self, s, epsilon_greedy=True):
        """
            Choose action based on Q-values using epsilon-greedy.
            Args
            ------
                s: the current state/frame of the game with shape (H, W, C)
                epsilon_greedy: whether to use epsilon_greedy
            Returns
            -------
                the chosen action
        """
        s = torch.from_numpy(s).permute(0, 3, 1, 2) # (1, C, H, W)
        q_vals = self._get_q_values(s).detach().numpy()
        if not epsilon_greedy or np.random.random() > self.eps_schedule.c:
            return np.argmax(q_vals, axis=1)
        else:
            return np.random.randint(q_vals.shape[1], size=(1, ))


    def evaluate(self, num_eval=5, render=False):
        """
            Evaluate the current model.
            Args
            -----
                render: whether to render the evaluation
        """
        if render:
            env = gym3.ViewerWrapper(self.env, info_key='rgb')
        else:
            env = self.env
        avg_reward = 0
        for i in range(num_eval):
            total_reward = 0
            step = 0
            s = env.observe()[1]
            while True:
                a = self.get_action(s, epsilon_greedy=False)
                env.act(a)
                r, s, first = env.observe()
                total_reward += r
                step += 1
                if first and step != 1:
                    avg_reward += total_reward / num_eval
                    break
            print('\tReward:', total_reward, 'Steps:', step)
        print('Evaluation ... Avg Reward:', avg_reward)
        return total_reward



    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--env_name', type=str, default=ENV_NAME)
        parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
        parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
        parser.add_argument('--start_level', type=int, default=START_LEVEL)

        parser.add_argument('--train_steps', type=int, default=TRAIN_STEPS)
        parser.add_argument('--train_resume', type=int, default=TRAIN_RESUME)

        parser.add_argument('--lr_start', type=float, default=LR_START)
        parser.add_argument('--lr_end', type=float, default=LR_END)
        parser.add_argument('--lr_steps', type=int, default=LR_STEPS)

        parser.add_argument('--eps_start', type=float, default=EPS_START)
        parser.add_argument('--eps_end', type=float, default=EPS_END)
        parser.add_argument('--eps_steps', type=int, default=EPS_STEPS)

        return parser


# test with a random policy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    agent_group = parser.add_argument_group('Agent Args')
    QAgent.add_to_argparse(agent_group)
    args = parser.parse_args()
    agent = QAgent(None, args)