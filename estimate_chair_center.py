import gym
import pickle
import numpy as np

# from policies.chair_push_new import get_center
from policies.chair_push_nudge import get_center


def quick_get_pc(obs):
    red_max_thresh = 0.7
    min_chair_height = 0.08
    max_chair_height = 2
    chair_red, = np.where(obs['pointcloud']['seg'][:, 0] == 0)
    chair_above, = np.where(obs['pointcloud']['xyz'][:, 2] > min_chair_height)
    chair_below, = np.where(obs['pointcloud']['xyz'][:, 2] < max_chair_height)
    chair_not_red, = np.where((obs['pointcloud']['rgb'][:, 0] < red_max_thresh))
    chair_idx = list(
        set(chair_red).intersection(set(chair_above)).intersection(chair_not_red).intersection(chair_below))
    return obs['pointcloud']['xyz'][chair_idx, :]

env = gym.make('PushChair-v0')
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
error_old = 0
error_new = 0
for level_idx in range(0, 250): # level_idx is a random seed

    print('#### Level {:d}'.format(level_idx))
    obs = env.reset(level=level_idx)

    chair_pc = quick_get_pc(obs)
    mean_pc = np.mean(chair_pc, axis=0)[:2]
    print('old', mean_pc)

    obs = env.reset(level=level_idx)
    my_center = get_center(obs)[:2]
    print('new', my_center)

    for i_step in range(1000000):
        env.render('human') # a display is required to use this function, rendering will slower the running speed
        # action = env.action_space.sample()
        action = [0 for _ in range(22)]
        obs, reward, done, info = env.step(action) # take a random action
        break

    env.close()
