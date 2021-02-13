import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gym3 import ViewerWrapper
from gym3.extract_dict_ob import ExtractDictObWrapper
from procgen import ProcgenGym3Env

from train.common.schedule import LinearSchedule
from train.common.memory import ReplayBuffer

# gym config
ENV_NAME = 'fruitbot'
NUM_ENVS = 1
NUM_LEVELS = 50
START_LEVEL = 500

# training config
TRAIN_STEPS = 4e6
TRAIN_RESUME = 0
LEARNING_START = 5e4
LEARNING_FREQ = 4
UPDATE_FREQ = 1e4
SAVING_FREQ = 2.5e5
BATCH_SIZE = 32
LR_START = 2.5e-4
LR_END = 5e-5
LR_STEPS = 2e6
EPS_START = 1
EPS_END = 0.1
EPS_STEPS = 1e6
GAMMA = 0.99

# eval config
EVAL_FREQ = 2.5e4


class QAgent():
    """
        The agent that performs Q-Learning.
    """
    def __init__(self, model, target_model, args):
        self.env = ProcgenGym3Env(
            num=args.num_envs, 
            env_name=args.env_name, 
            num_levels=args.num_levels, 
            start_level=args.start_level, 
            distribution_mode='easy', 
            render_mode='rgb_array'
        )
        self.env = ExtractDictObWrapper(self.env, 'rgb')


        self.memory = ReplayBuffer()
        self.config = args
        self.step = self.config.train_resume

        self.model = model
        self.target = target_model

        self.eps_scheduler = LinearSchedule(
            self.config.eps_start, self.config.eps_end, self.config.eps_steps
        )
        # for safety reasons
        self.eps_scheduler.step(self.step)

        self.lr_scheduler = LinearSchedule(
            self.config.lr_start, self.config.lr_end, self.config.lr_steps
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr_scheduler.step(self.step)
        )

        # if it's a resume, then load model
        if self.step != 0:
            self.load()
        self.update_network()

        self.model.train()
        self.target.eval()

        self.writer = SummaryWriter(log_dir='results/logs/ImpalaDQN')


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
        s = torch.from_numpy(s.astype('float32')).permute(0, 3, 1, 2) # (1, C, H, W)
        q_vals = self._get_q_values(s).detach().numpy()
        if not epsilon_greedy or np.random.random() > self.eps_scheduler.c:
            return np.argmax(q_vals, axis=1)
        else:
            return np.random.randint(q_vals.shape[1], size=(1, ))

    def evaluate(self, num_eval=50, render=False):
        """
            Evaluate the current model.
            Args
            -----
                render: whether to render the evaluation
        """
        if render:
            env = ViewerWrapper(self.env, info_key='rgb')
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
                r, s, done = env.observe()
                total_reward += r
                step += 1
                if done:
                    avg_reward += total_reward / num_eval
                    break
            # print('\tReward:', total_reward, 'Steps:', step)
        self.writer.add_scalar('Reward/eval', avg_reward, self.step)
        # print('Evaluation ... Avg Reward:', avg_reward)
        return avg_reward

    def save(self):
        """
            Save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'results/weights/checkpoint')

    def load(self):
        """
            Load the model.
        """
        checkpoint = torch.load('results/weights/checkpoint')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("loaded")

    def update_network(self):
        """
            Update the target network.
        """
        self.target.load_state_dict(self.model.state_dict())
    
    def train_step(self):
        """
            Perform one batch update for the model.
        """
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self.memory.sample(self.config.batch_size)

        state_batch = torch.from_numpy(np.array(sampled_states).astype('float32')).squeeze(1).permute(0, 3, 1, 2)
        next_state_batch = torch.from_numpy(np.array(sampled_next_states).astype('float32')).squeeze(1).permute(0, 3, 1, 2)
        action_batch = torch.from_numpy(np.array(sampled_actions).astype('int64'))
        reward_batch = torch.from_numpy(np.array(sampled_rewards).astype('float32'))
        done_batch = np.array(sampled_dones)

        # the q values for the action taken
        q_vals = self.model(state_batch)
        q_vals = q_vals.gather(1, action_batch)
        # the q values for the next state
        next_q_vals = self.target(next_state_batch).max(1)[0].detach()
        next_q_vals[done_batch.reshape(-1)] = 0
        target_vals = (next_q_vals.unsqueeze(1) * self.config.gamma) + reward_batch

        loss = F.smooth_l1_loss(q_vals, target_vals)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.optimizer.param_groups[0]['lr'] = self.lr_scheduler.step(self.step)
        return loss

    def train(self):
        """
            The main training loop for the agent
        """
        prog_bar = tqdm(total=self.config.train_steps, initial=self.step)
        hyperparam_log = tqdm(total=0, position=1, bar_format='{desc}')
        performance_log = tqdm(total=0, position=2, bar_format='{desc}')
        eval_log = tqdm(total=0, position=3, bar_format='{desc}')
        since_last_save = 0
        since_last_eval = 0

        s = self.env.observe()[1]
        ep_reward = 0
        ep_loss = 0
        while self.step < self.config.train_steps:
            a = self.get_action(s)
            self.env.act(a)
            r, sp, d = self.env.observe()
            self.memory.store(s, a, r, sp, d)
            ep_reward += r[0]
            s = sp
            if d:
                self.writer.add_scalar('Reward/train', ep_reward, self.step)
                self.writer.add_scalar('Loss/train', ep_loss, self.step)
                performance_log.set_description_str(f'Ep Reward: {ep_reward}, Ep Loss: {ep_loss:.3f}')
                ep_reward = 0
                ep_loss = 0

            self.step += 1
            since_last_save += 1
            since_last_eval += 1

            # if start training
            if (self.step - self.config.train_resume) >= self.config.learning_start:
                if self.step % self.config.learning_freq == 0:
                    ep_loss += self.train_step()
                if self.step % self.config.update_freq == 0:
                    self.update_network()
                if since_last_save >= self.config.saving_freq:
                    self.save()
                    since_last_save = 0
                if since_last_eval >= self.config.eval_freq:
                    eval_reward = self.evaluate()
                    eval_log.set_description_str(f'Eval Reward: {eval_reward}')
                    since_last_eval = 0

            self.eps_scheduler.step(self.step)
            prog_bar.update(1)
            hyperparam_log.set_description_str(f"LR: {self.optimizer.param_groups[0]['lr']:.7f}, Epsilon: {self.eps_scheduler.c:.3f}")


            

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--env_name', type=str, default=ENV_NAME)
        parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
        parser.add_argument('--num_levels', type=int, default=NUM_LEVELS)
        parser.add_argument('--start_level', type=int, default=START_LEVEL)

        parser.add_argument('--train_steps', type=int, default=TRAIN_STEPS)
        parser.add_argument('--train_resume', type=int, default=TRAIN_RESUME)
        parser.add_argument('--learning_start', type=int, default=LEARNING_START)
        parser.add_argument('--learning_freq', type=int, default=LEARNING_FREQ)
        parser.add_argument('--update_freq', type=int, default=UPDATE_FREQ)
        parser.add_argument('--saving_freq', type=int, default=SAVING_FREQ)
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)

        parser.add_argument('--lr_start', type=float, default=LR_START)
        parser.add_argument('--lr_end', type=float, default=LR_END)
        parser.add_argument('--lr_steps', type=int, default=LR_STEPS)
        parser.add_argument('--eps_start', type=float, default=EPS_START)
        parser.add_argument('--eps_end', type=float, default=EPS_END)
        parser.add_argument('--eps_steps', type=int, default=EPS_STEPS)
        parser.add_argument('--gamma', type=float, default=GAMMA)

        parser.add_argument('--eval_freq', type=type, default=EVAL_FREQ)

        return parser


# test with a random policy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    agent_group = parser.add_argument_group('Agent Args')
    QAgent.add_to_argparse(agent_group)
    args = parser.parse_args()
    agent = QAgent(None, None, args)