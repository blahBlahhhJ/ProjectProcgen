import argparse
import numpy as np
import torch
import gym3
from gym3.extract_dict_ob import ExtractDictObWrapper
from procgen import ProcgenGym3Env

from .utils.schedule import LinearSchedule
from .utils.memory import ReplayBuffer

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
EVAL_FREQ = 2.5e5


class QAgent():
    """
        The agent that performs Q-Learning.
    """
    def __init__(self, model, args):
        self.env = ProcgenGym3Env(
            num=args.num_envs, 
            env_name=args.env_name, 
            num_levels=args.num_levels, 
            start_level=args.start_level, 
            distribution_mode='easy', 
            render_mode='rgb_array'
        )
        self.env = ExtractDictObWrapper(self.env, 'rgb')

        self.model = model
        self.memory = ReplayBuffer()
        self.config = args
        self.step = self.config.train_resume
        # if it's a resume, then load model
        if self.step != 0:
            self.load()

        self.eps_scheduler = LinearSchedule(
            self.config.eps_start, self.config.eps_end, self.config.eps_steps
        )
        # for safety reasons
        self.eps_scheduler.step(self.step)

        self.lr_schedule = LinearSchedule(
            self.config.lr_start, self.config.lr_end, self.config.lr_steps
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr_schedule(self.step)
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.lr_schedule
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
        s = torch.from_numpy(s.astype('float32')).permute(0, 3, 1, 2) # (1, C, H, W)
        q_vals = self._get_q_values(s).detach().numpy()
        if not epsilon_greedy or np.random.random() > self.eps_scheduler.c:
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
            print(env.observe()[2])
            while True:
                a = self.get_action(s, epsilon_greedy=False)
                env.act(a)
                r, s, done = env.observe()
                total_reward += r
                step += 1
                if done:
                    avg_reward += total_reward / num_eval
                    break
            print('\tReward:', total_reward, 'Steps:', step)
        print('Evaluation ... Avg Reward:', avg_reward)
        return total_reward

    def save(self):
        pass

    def load(self):
        pass

    def update_network(self):
        pass
    
    def train_step(self):
        pass

    def train(self):
        since_last_save = 0
        since_last_eval = 0

        s = self.env.observe()[1]
        while self.step < self.config.train_steps:
            a = self.get_action(s)
            self.env.act(a)
            r, sp, d = self.env.observe()
            self.memory.store(s, a, r, sp, d)




            self.step += 1
            since_last_save += 1
            since_last_eval += 1

            if (self.step - self.config.train_resume) >= self.config.learning_start:
                if self.step % self.config.learning_freq == 0:
                    self.train_step()
                    since_last_learn = 0
                if self.step % self.config.update_freq == 0:
                    self.update_network()
                if since_last_save >= self.config.saving_freq:
                    self.save()
                    since_last_save = 0
                if since_last_eval >= self.config.eval_freq:
                    self.evaluate()
                    since_last_eval = 0

            self.lr_scheduler.step()
            self.eps_scheduler.step()

            

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
    agent = QAgent(None, args)