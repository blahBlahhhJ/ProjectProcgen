import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from procgen import ProcgenEnv
from utils.vec_envs import VecExtractDictObs, VecNormalize, VecMonitor

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
    def __init__(self, env, model, data_config, args):
        self.env = env
        self.s = self.env.reset()

        self.config = args
        self.step = self.config.train_resume
        assert self.step % (args.num_envs * args.update_freq) == 0

        self.writer = torch.utils.tensorboard.SummaryWriter('results/logs/PPO/' + str(self.config.num_levels) + 'lvl')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using', self.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_start)

        if self.step != 0:
            self.load()

    def save(self):
        """
            Save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'env_mean': self.env.ret_rms.mean,
            'env_var': self.env.ret_rms.var
        }, 'results/weights/ppo/checkpoint-' + str(self.step))

    def load(self):
        """
            Load the model.
        """
        checkpoint = torch.load('results/weights/ppo/checkpoint-' + str(self.step))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.env.ret_rms.mean = checkpoint['env_mean']
        self.env.ret_rms.var = checkpoint['env_var']
        print("loaded")


    def get_action(self, a_logprob, deterministic=False):
        """
            Get the action according to the predicted probabilities.
            Args:
            ------
            a_logprob: torchFloatTensor
                the predicted action probabilities
            deterministic: boolean
                whether to choose max prob index or do random sampling

        """
        p = a_logprob.exp()
        if deterministic:
            return torch.argmax(p, dim=-1)
        else:
            return categorical(p).sample()

    def _to_torch(self, states, returns, actions, values, logprobs):
        """
            Turn numpy to torch, send to gpu.
            Normalize for states, unsqueeze axis for returns actions values.
        """
        states = states / 255
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device) # (B, C, H, W)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device) # (B, 1)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device) # (B, 1)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device) # (B, 1)
        logprobs = torch.FloatTensor(logprobs).to(self.device) # (B, A)
        return states, returns, actions, values, logprobs

    def _frame_to_torch(self, s):
        """
            Normalize, switch axis, send to gpu.
        """
        s = s / 255
        s = torch.FloatTensor(s).permute(0, 3, 1, 2)
        return s.to(self.device)
    
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
            states: torch.FloatTensor (B, C, H, W)
                the normalized frames of the game
            returns: torch.FloatTensor (B, 1)
                the generalized advantage estimation
            actions: torch.LongTensor(B, 1)
                the actions for each frame
            values: torch.FloatTensor(B, 1)
                the estimated values for each frame
            logprobs: torch.FloatTensor(B, A)
                the log probability of each action
        """
        with torch.no_grad():
            self.model.eval()
            states, rewards, actions, values, dones, logprobs = [], [], [], [], [], []
            infos = []

            for _ in range(self.config.update_freq):
                traj_bar.update()

                states.append(self.s)    # type numpy, un-normalized to store
                # type torch, normalized to predict
                a_logprob, v = self.model(self._frame_to_torch(self.s))
                a = self.get_action(a_logprob).cpu().numpy()
                v = v.squeeze(-1).detach().cpu().numpy()

                actions.append(a)
                values.append(v)
                logprobs.append(a_logprob.detach().cpu().numpy())

                self.s, r, d, info = self.env.step(a)

                rewards.append(r)
                dones.append(d)

                for i in info:
                    episode_info = i.get('episode')
                    if episode_info:
                        infos.append(episode_info)

            # put the value for the last "next state"
            _, last_v = self.model(self._frame_to_torch(self.s))
            values.append(last_v.squeeze(-1).detach().cpu().numpy())
                        
            states = np.asarray(states, dtype=self.s.dtype)
            actions = np.asarray(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.bool)
            values = np.asarray(values, dtype=np.float32)
            logprobs = np.asarray(logprobs, dtype=np.float32)

            returns = self.compute_gae(rewards, dones, values)

            # dump the "extra" last value
            values = values[:-1]

            avg_reward = np.mean([i['r'] for i in infos])
            self.writer.add_scalar('Reward/train', avg_reward, self.step)

        return self._to_torch(*map(sf01, (states, returns, actions, values, logprobs)))

    def compute_gae(self, rewards, dones, values):
        """
            Compute the generalized advantage estimation.
            Args:
            ------
            rewards: np.ndarray (update_freq, num_envs)
                rewards from trajectory. 
            dones: np.ndarray (update_freq, num_envs)
                done masks from trajectory. 
            values: np.ndarray (update_freq+1, num_envs)
                estimated values from trajectory. 
            Returns
            -------
            np.ndarray of shape (update_freq, num_envs)
        """
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(self.config.update_freq)):
            next_v = values[t + 1]
            nonterminal = 1.0 - dones[t]

            delta = rewards[t] + self.config.gamma * next_v * nonterminal - values[t]
            lastgaelam = delta + self.config.gamma * self.config.lam * nonterminal * lastgaelam
            advs[t] = lastgaelam
        returns = advs + values[:-1]
        return returns

    def train_step(self, states, returns, actions, values, logprobs):
        """
            Update the model for one batch of training data.
            Args:
            -------
            states: (B, C, H, W)
                the normalized frames of the game
            returns: (B, 1)
                the generalized advantage estimation
            actions: (B, 1)
                the actions for each frame
            values: (B, 1)
                the estimated values for each frame
            logprobs: (B, A)
                the log probability of each action
        """
        # normalize advantages
        with torch.no_grad():
            advs = returns - values     # (B, 1)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        new_logprobs, v = self.model(states)    # (B, A), (B, 1)

        # entropy bonus for exploration (entropy=-∑p*logp)
        entropy = -new_logprobs.exp() * new_logprobs        # (B, A)
        entropy = self.config.ent * entropy.sum(1).mean()   # scalar

        new_logprobs = new_logprobs.gather(1, actions)  # (B, 1)
        old_logprobs = logprobs.gather(1, actions)      # (B, 1)
        # pi_new / pi_old
        ratio = torch.exp(new_logprobs - old_logprobs)  # (B, 1)

        # actor loss
        actor_loss_unclipped = -advs * ratio            # (B, 1)
        actor_loss_clipped = -advs * torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)                         # (B, 1)
        actor_loss = torch.maximum(actor_loss_clipped, actor_loss_unclipped).squeeze(1).mean()  # scalar

        # critic loss
        critic_loss_unclipped = torch.square(v - returns)   # (B, 1)
        critic_loss_clipped = torch.square(values + torch.clamp(v - values, -self.config.clip_range, self.config.clip_range) - returns)      # (B, 1)
        critic_loss = self.config.cl * 0.5 * torch.maximum(critic_loss_clipped, critic_loss_unclipped).squeeze(1).mean()            # scalar

        # do the update
        total_loss = actor_loss + critic_loss - entropy
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return total_loss

    def evaluate(self, num_eval=50, record=False):
        """
            Evaluate the model on unseen levels.
        """
        self.model.eval()
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
            a_prob, _ = self.model(s.to(self.device))
            a = self.get_action(a_prob, deterministic=True).cpu().numpy()
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
        idxs = np.arange(train_size)    # for splitting batches

        # number of times to call gather_trajectory & train_step
        num_loops = self.config.train_steps // train_size
        start_loop = self.step // train_size

        # tqdm progress bars & logs
        prog_bar = tqdm(total=int(num_loops), position=0, desc='Train Step')
        traj_bar = tqdm(total=self.config.update_freq, position=1, desc='Gathering Trajectory')
        train_bar = tqdm(total=self.config.num_epochs, position=2, desc='Updating Model')
        sub_train_bar = tqdm(total=self.config.num_batches, position=3, desc='Batches')
        performance_log = tqdm(total=0, position=4, bar_format='{desc}')
        eval_log = tqdm(total=0, position=5, bar_format='{desc}')

        # the big training loop
        for i in range(int(start_loop)+1, int(num_loops)+1):
            prog_bar.update(1)
            traj_bar.reset()
            # get data
            states, returns, actions, values, logprobs = self.gather_trajectory(traj_bar)
            losses = []
            train_bar.reset()
            self.model.train()
            # the epoch loop
            for __ in range(self.config.num_epochs):
                # randomize indices for picking batches
                np.random.shuffle(idxs)
                sub_train_bar.reset()
                # the batch loop
                for b in range(self.config.num_batches):
                    sub_train_bar.update(1)
                    batch_idxs = idxs[b*batch_size:(b+1)*batch_size]
                    # get one batch of data
                    slices = (arr[batch_idxs] for arr in (states, returns, actions, values, logprobs))
                    losses.append(self.train_step(*slices).item())
                    performance_log.set_description_str(f'Loss: {np.mean(losses):.3f}')
                train_bar.update(1)
            self.writer.add_scalar('Loss/train', np.mean(losses), i)

            self.step += train_size

            if i % self.config.eval_freq == 0:
                eval_reward = self.evaluate()
                self.writer.add_scalar('Reward/eval', eval_reward, self.step)
                eval_log.set_description_str(f'Eval Reward: {eval_reward}')
            if i % self.config.saving_freq == 0:
                self.save()
        self.save()

    @staticmethod
    def add_to_argparse(parser):
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

        return parser



