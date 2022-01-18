import gym
import pickle
from mani_skill import env

env = gym.make('OpenCabinetDoor-v0')
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(env.observation_space)  # this shows the structure of the observation, openai gym's format
print(env.action_space)  # this shows the action space, openai gym's format

# for level_idx in range(0, 5): # level_idx is a random seed
# level_idx = 1002

num = 10
for level_idx in range(1003, 1010):
    obs = env.reset(level=level_idx)
    print('#### Level {:d}'.format(level_idx))
    for i_step in range(1000000):
        env.render('human')  # a display is required to use this function, rendering will slower the running speed
        # action = env.action_space.sample()
        action = [0 for _ in range(13)]
        action[0] = 1
        action[1] = -1
        # print("action")
        # print(action)
        obs, reward, done, info = env.step(action)  # take a random action

        with open(f'temp/{level_idx}_{i_step}.pkl', 'wb') as handle:
            pickle.dump(obs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if i_step >= num:
            break
    env.close()
