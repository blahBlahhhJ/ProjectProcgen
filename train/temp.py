from procgen import ProcgenEnv
from matplotlib import pyplot as plt
import numpy as np

from utils.vec_envs import VecExtractDictObs, VecNormalize, VecMonitor


env = ProcgenEnv(
    num_envs=1, 
    env_name='fruitbot', 
    num_levels=0, 
    start_level=0, 
    distribution_mode='easy', 
    render_mode='rgb_array'
)
env = VecExtractDictObs(env, 'rgb')
env = VecMonitor(env)
env = VecNormalize(env, ob=False)
s = env.reset()
d = False
frames = [s]
while True:
    a = np.ones(1)
    s, r, d, _ = env.step(a)
    frames.append(s)
    if d:
        break

for f in frames[-5:]:
    plt.imshow(f[0])
    plt.show()
