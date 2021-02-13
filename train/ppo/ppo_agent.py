import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from procgen import ProcgenEnv

from .ppo import ImpalaPPO
from utils.vec_envs import VecExtractDictObs, VecNormalize, VecMonitor

# gym config
ENV_NAME = 'fruitbot'
NUM_ENVS = 64
NUM_LEVELS = 50
START_LEVEL = 500

# training config
TRAIN_STEPS = 5e6
TRAIN_RESUME = 0
UPDATE_FREQ = 256
BATCH_SIZE = 8
GAMMA = .999
LAMBDA = .95

categorical = torch.distributions.categorical.Categorical


def sf01(x):
    s = x.shape
    return x.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class PPOAgent:
    def __init__(self, data_config, args):
        self.env = ProcgenEnv(
            num_envs=args.num_envs, 
            env_name=args.env_name, 
            num_levels=args.num_levels, 
            start_level=args.start_level, 
            distribution_mode='easy', 
            render_mode='rgb_array'
        )
        self.env = VecExtractDictObs(self.env, 'rgb')
        # self.env = VecMonitor(self.env)
        self.env = VecNormalize(self.env, ob=False)

        self.s = self.env.reset() / 255
        self.d = np.array([False for _ in range(args.num_envs)])

        self.config = args
        self.step = self.config.train_resume

        self.model = ImpalaPPO(data_config, args)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using', self.device)

    def get_action(self, probs, deterministic=False):
        """
            Return the action chosen by greedy or sampling.
        """
        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            return categorical(probs).sample()

    def gather_trajectory(self):
        """
            Gather trajectory of length UPDATE_FREQ.
            B = num_envs * update_freq\\
            H = height of frame\\
            W = width of frame\\
            C = channel of frame\\
            A = action space dimension
            Returns
            -------
            states: (B, H, W, C)
                the normalized frames of the game
            returns: (B, )
                the generalized advantage estimation
            masks: (B, )
                the masks of first frame
            actions: (B, )
                the actions for each frame
            values: (B, )
                the estimated values for each frame
            neglogprobs: (B, A)
                the negative log probability of each action
        """
        self.model.eval()
        states, rewards, actions, values, dones, neglogprobs = [], [], [], [], [], []

        for _ in range(self.config.update_freq):
            s = torch.FloatTensor(self.s).permute(0, 3, 1, 2)
            a_prob, v = self.model(s)
            a = self.get_action(a_prob).numpy()
            n = -torch.log(a_prob)

            states.append(self.s.copy())
            actions.append(a)
            values.append(v.squeeze(1).detach().numpy())
            neglogprobs.append(n.detach().numpy())
            dones.append(self.d)

            self.s, r, self.d, _ = self.env.step(a)
            self.s = self.s / 255

            rewards.append(r)
        
        states = np.array(states, dtype=self.s.dtype)
        rewards = np.array(rewards, dtype=np.float32)
        actions = np.array(actions)
        values = np.array(values, dtype=np.float32)
        neglogprobs = np.array(neglogprobs, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)

        self.s = torch.FloatTensor(self.s).permute(0, 3, 1, 2) / 255.
        _, v = self.model(self.s)
        v = v.squeeze(1).detach().numpy()

        returns = self.compute_gae(rewards, dones, values, v)

        return tuple(map(sf01, (states, returns, dones, actions, values, neglogprobs)))

    def compute_gae(self, rewards, dones, values, v):
        """
            Compute the generalized advantage estimation.
        """
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        g = 0
        for t in reversed(range(self.config.update_freq)):
            if t == self.config.update_freq - 1:
                nonterminal = 1.0 - self.d
                next_v = v
            else:
                nonterminal = 1.0 - dones[t + 1]
                next_v = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_v * nonterminal - values[t]
            advs[t] = g = delta + self.config.gamma * self.config.lam * nonterminal * g
        returns = advs + values
        return returns

    def train(self):
        train_size = self.config.num_envs * self.config.update_freq




    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--env_name', type=str, default=ENV_NAME)
        parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
        parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
        parser.add_argument('--start_level', type=int, default=START_LEVEL)

        parser.add_argument('--train_steps', type=int, default=TRAIN_STEPS)
        parser.add_argument('--train_resume', type=int, default=TRAIN_RESUME)
        # parser.add_argument('--learning_start', type=int, default=LEARNING_START)
        # parser.add_argument('--learning_freq', type=int, default=LEARNING_FREQ)
        parser.add_argument('--update_freq', type=int, default=UPDATE_FREQ)
        # parser.add_argument('--saving_freq', type=int, default=SAVING_FREQ)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)

        # parser.add_argument('--lr_start', type=float, default=LR_START)
        # parser.add_argument('--lr_end', type=float, default=LR_END)
        # parser.add_argument('--lr_steps', type=int, default=LR_STEPS)
        # parser.add_argument('--eps_start', type=float, default=EPS_START)
        # parser.add_argument('--eps_end', type=float, default=EPS_END)
        # parser.add_argument('--eps_steps', type=int, default=EPS_STEPS)
        parser.add_argument('--gamma', type=float, default=GAMMA)
        parser.add_argument('--lam', type=float, default=LAMBDA)

        # parser.add_argument('--eval_freq', type=type, default=EVAL_FREQ)


        model_group = parser.add_argument_group("Model Args")
        ImpalaPPO.add_to_argparse(model_group)

        return parser



