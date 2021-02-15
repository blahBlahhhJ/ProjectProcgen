import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from procgen import ProcgenEnv

from .ppo import ImpalaPPO
from utils.vec_envs import VecExtractDictObs, VecNormalize, VecMonitor

# gym config
ENV_NAME = 'starpilot'
NUM_ENVS = 64
NUM_LEVELS = 50
START_LEVEL = 500

# training config
TRAIN_STEPS = 5e6
TRAIN_RESUME = 0
UPDATE_FREQ = 256   # steps per update
EVAL_FREQ = 10      # loops per eval
SAVING_FREQ = 10    # loops per save
NUM_BATCHES = 8     # number of splits of training size
NUM_EPOCHS = 3
CLIP_RANGE = 0.2
GAMMA = .999
LAMBDA = .95
ENT_COEF = .01
CL_COEF = 0.5
LR_START = 5e-4

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
        self.env = VecMonitor(self.env)
        self.env = VecNormalize(self.env, ob=False)

        self.s = self.env.reset() / 255

        self.config = args
        self.step = self.config.train_resume
        assert self.step % (args.num_envs * args.update_freq) == 0

        self.model = ImpalaPPO(data_config, args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_start)

        if self.step != 0:
            self.load()

        self.writer = torch.utils.tensorboard.SummaryWriter('results/logs/PPO/' + str(self.config.num_levels) + 'lvl')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using', self.device)

    def save(self):
        """
            Save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'results/weights/ppo/checkpoint-' + str(self.step))

    def load(self):
        """
            Load the model.
        """
        checkpoint = torch.load('results/weights/ppo/checkpoint-' + str(self.step))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("loaded")


    def get_action(self, probs, deterministic=False):
        """
            Return the action chosen by greedy or sampling.
        """
        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            return categorical(probs).sample()

    def _to_torch(self, states, returns, actions, values, neglogprobs):
        states = torch.FloatTensor(states).permute(0, 3, 1, 2)  # (B, C, H, W)
        returns = torch.FloatTensor(returns).unsqueeze(1)    # (B, )
        actions = torch.LongTensor(actions).unsqueeze(1) # (B, )
        values = torch.FloatTensor(values).unsqueeze(1)
        neglogprobs = torch.FloatTensor(neglogprobs)
        return states, returns, actions, values, neglogprobs
    
    def gather_trajectory(self, traj_bar):
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
            actions: (B, )
                the actions for each frame
            values: (B, )
                the estimated values for each frame
            neglogprobs: (B, A)
                the negative log probability of each action
        """
        self.model.eval()
        states, rewards, actions, values, dones, neglogprobs = [], [], [], [], [], []
        infos = []

        for _ in range(self.config.update_freq):
            traj_bar.update()
            s = torch.FloatTensor(self.s).permute(0, 3, 1, 2)
            a_prob, v = self.model(s)
            a = self.get_action(a_prob).numpy()
            n = -torch.log(a_prob)

            states.append(self.s.copy())
            actions.append(a)
            values.append(v.squeeze(1).detach().numpy())
            neglogprobs.append(n.detach().numpy())

            self.s, r, d, info = self.env.step(a)
            self.s = self.s / 255

            rewards.append(r)
            dones.append(d)

            for i in info:
                episode_info = i.get('episode')
                if episode_info:
                    infos.append(episode_info)
        avg_reward = np.mean([i['r'] for i in infos])
        self.writer.add_scalar('Reward/train', avg_reward, self.step)

        s = torch.FloatTensor(self.s).permute(0, 3, 1, 2) / 255.
        _, v = self.model(s)
        # one more estimate for values
        values.append(v.squeeze(1).detach().numpy())

        states = np.array(states, dtype=self.s.dtype)
        rewards = np.array(rewards, dtype=np.float32)
        actions = np.array(actions)
        values = np.array(values, dtype=np.float32)
        neglogprobs = np.array(neglogprobs, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)

        returns = self.compute_gae(rewards, dones, values, v)

        return self._to_torch(*map(sf01, (states, returns, actions, values, neglogprobs)))

    def compute_gae(self, rewards, dones, values):
        """
            Compute the generalized advantage estimation.
        """
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        g = 0
        for t in reversed(range(self.config.update_freq)):
            next_v = values[t + 1]
            nonterminal = 1.0 - dones[t]

            delta = rewards[t] + self.config.gamma * next_v * nonterminal - values[t]
            advs[t] = g = delta + self.config.gamma * self.config.lam * nonterminal * g
        returns = advs + values
        return returns

    def train_step(self, states, returns, actions, values, neglogprobs):
        """
            Update the model for one batch of training data.
        """
        with torch.no_grad():
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        a_prob, v = self.model(states)
        new_neglogprobs = -torch.log(a_prob).gather(1, actions)
        old_neglogprobs = neglogprobs.gather(1, actions)
        # pi_new / pi_old
        ratio = torch.exp(new_neglogprobs - old_neglogprobs).squeeze(1)

        # actor loss
        actor_loss_unclipped = -advs * ratio
        actor_loss_clipped = -advs * torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
        actor_loss = torch.max(actor_loss_clipped, actor_loss_unclipped).mean()

        # critic loss
        critic_loss_unclipped = torch.square(v - returns)
        critic_loss_clipped = torch.square(values + torch.clamp(v - values, -self.config.clip_range, self.config.clip_range))
        critic_loss = self.config.cl * 0.5 * torch.max(critic_loss_clipped, critic_loss_unclipped).mean()

        # entropy bonus for exploration
        entropy = new_neglogprobs * a_prob
        entropy = self.config.ent * entropy.sum(1).mean()

        # do the update
        total_loss = actor_loss + critic_loss - entropy
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return total_loss

    def evaluate(self, num_eval=50, record=False):
        eval_env = ProcgenEnv(
            num_envs=1, 
            env_name=self.config.env_name, 
            num_levels=500, 
            start_level=0, 
            distribution_mode='easy', 
            render_mode='rgb_array'
        )
        eval_env = VecExtractDictObs(eval_env, 'rgb')
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, ob=False)
        
        infos = []
        
        s = eval_env.reset() / 255
        while len(infos) < num_eval:
            s = torch.FloatTensor(s).permute(0, 3, 1, 2)
            a_prob, _ = self.model(s)
            a = self.get_action(a_prob, deterministic=True).numpy()
            s, r, done, info = eval_env.step(a)
            s = s / 255

            episode_info = info[0].get('episode')
            if episode_info:
                infos.append(episode_info)
        
        avg_reward = np.mean([i['r'] for i in infos])
        return avg_reward


    
    def train(self):
        # number of samples per train_step
        train_size = self.config.num_envs * self.config.update_freq
        batch_size = train_size // self.config.num_batches
        # number of times to call train_step
        num_loops = self.config.train_steps // train_size
        start_loop = self.step // train_size
        prog_bar = tqdm(total=int(num_loops), position=0, desc='Train Step')
        traj_bar = tqdm(total=self.config.update_freq, position=1, desc='Gathering Trajectory')
        train_bar = tqdm(total=self.config.num_epochs, position=2, desc='Updating Model')
        sub_train_bar = tqdm(total=self.config.num_batches, position=3, desc='Batches')
        performance_log = tqdm(total=0, position=4, bar_format='{desc}')
        eval_log = tqdm(total=0, position=5, bar_format='{desc}')

        for i in range(int(start_loop)+1, int(num_loops)+1):
            prog_bar.update(1)
            traj_bar.reset()
            states, returns, actions, values, neglogprobs, = self.gather_trajectory(traj_bar)
            idxs = np.arange(train_size)
            avg_loss = 0
            train_bar.reset()
            for __ in range(self.config.num_epochs):
                np.random.shuffle(idxs)
                ep_loss = []
                sub_train_bar.reset()
                for start in range(0, train_size, batch_size):
                    sub_train_bar.update(1)
                    end = start + batch_size
                    batch_idxs = idxs[start:end]
                    # get one batch of data
                    slices = (arr[batch_idxs] for arr in (states, returns, actions, values, neglogprobs))
                    ep_loss.append(self.train_step(*slices).detach().numpy())
                    performance_log.set_description_str(f'Average Loss: {ep_loss[-1]:.3f}')
                avg_loss += np.mean(ep_loss) / self.config.num_epochs
                train_bar.update(1)
            self.writer.add_scalar('Loss/train', avg_loss, i)

            self.step += train_size
            if i % self.config.eval_freq == 0:
                eval_reward = self.evaluate()
                self.writer.add_scalar('Reward/eval', eval_reward, self.step)
                eval_log.set_description_str(f'Eval Reward: {eval_reward}')
            if i % self.config.saving_freq == 0:
                self.save()
            


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--env_name', type=str, default=ENV_NAME)
        parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
        parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
        parser.add_argument('--start_level', type=int, default=START_LEVEL)

        parser.add_argument('--train_steps', type=int, default=TRAIN_STEPS)
        parser.add_argument('--train_resume', type=int, default=TRAIN_RESUME)
        parser.add_argument('--update_freq', type=int, default=UPDATE_FREQ)
        parser.add_argument('--saving_freq', type=int, default=SAVING_FREQ)
        parser.add_argument('--num_batches', type=int, default=NUM_BATCHES)
        parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
        parser.add_argument('--clip_range', type=float, default=CLIP_RANGE)
        parser.add_argument('--ent', type=float, default=ENT_COEF)
        parser.add_argument('--cl', type=float, default=CL_COEF)
        parser.add_argument('--lr_start', type=float, default=LR_START)
        parser.add_argument('--gamma', type=float, default=GAMMA)
        parser.add_argument('--lam', type=float, default=LAMBDA)
        parser.add_argument('--eval_freq', type=type, default=EVAL_FREQ)


        model_group = parser.add_argument_group("Model Args")
        ImpalaPPO.add_to_argparse(model_group)

        return parser



