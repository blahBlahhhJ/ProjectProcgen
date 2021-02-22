# ProjectProcgen

This project allows quick and easy experiment on [OpenAI's Procgen Benchmark](https://github.com/openai/procgen).

Unlike [OpenAI's baseline](https://github.com/openai/baselines), this project is implemented using PyTorch.

Currently supported methods:
- Proximal Policy Optimization (PPO)
- Deep Q Learning [probably won't work]

## Getting started
### Install Dependencies (untested)
```shell
$ conda env create -f environment.yml
$ conda activate procgen
```
- essential packages are: pytorch, procgen, tensorboard, tqdm
### Run Experiment
```shell
$ cd train
$ python run.py
```
## Optional Arguments:
### Environment Arguments
| Argument | Default | Description |
| -- | --- | --- |
| `--env_name` | 'fruitbot' | The name of the environment |
| `--num_envs` | 64 | The number of copies for the environment |
| `--num_levels` | 50 | The number of levels for the agent to train |
| `--start_level` | 500 | The starting level for the agent to train |

### PPO Agent Arguments
| Argument | Default | Description |
| -- | --- | --- |
| `--train_step` | 5e6 | The total number of frames for the agent to train |
| `--train_resume` | 0 | The checkpoint for agent to resume training |
| `--update_freq` | 256 | The number of frames for each environment to gather for training |
| `--eval_freq` | 10 | The frequency (per training loop) to evaluate performance |
| `--saving_freq` | 10 | The frequency (per training loop) to save model |
| `--num_batches` | 8 | The number of batches in one epoch (not batch size) |
| `--num_epochs` | 3 | The number of epochs in one train step |
| `--clip_range` | 0.2 | The range to clip policy deviation and value estimate deviation |
| `--gamma` | 0.999 | The discount factor |
| `--lam` | 0.95 | The hyperparameter in GAE |
| `--ent` | 0.01 | The coefficient for entropy penalty |
| `--cl` | 0.5 | The coefficient for value estimation |
| `--lr_start` | 5e-4 | The learning rate for Adam |

### PPO Model Arguments
| Argument | Default | Description |
| -- | --- | --- |
| `--conv_dims` | [16, 32, 32] | The number of filters in each Impala block |
| `--fc_dims` | [256] | The number of hidden units in the fully connected layer
- Note that although these are passed in as lists, this project doesn't support customizing the number of layers (by now). So the length of these two arguments should match the length of the default ones.