import gymnasium as gym
import ale_py
gym.register_envs(ale_py) # register the Atari environments

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import time

from agent import Agent
from wrappers import AtariImage, ClipReward, NoopResetEnv, FireResetEnv, EpisodicLifeEnv, BreakoutActionTransform
from eval import evaluate
from log import plot_logs

def create_checkpoints_dir(dir_path='saved_models/'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")







# directory creation for checkpoints
checkpoints_dir_path = create_checkpoints_dir()

# cofiguration of the environment
game_id = 'ALE/Breakout-v5'
frame_skip = 4
env = gym.make(id=game_id, frameskip=1)
wrappers_lst = [ClipReward, EpisodicLifeEnv, NoopResetEnv, FireResetEnv, AtariImage, BreakoutActionTransform]
wrapped_env = env
for wrapper in wrappers_lst:
    wrapped_env = wrapper(wrapped_env)
print(f'The Environment for the Game {game_id} has been Initialized.')

# configuration of the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# configuration of the agent
agent = Agent(num_of_actions=4, device=device) # we keep the arguments as default


# parameters of the training loop 
max_total_interactions = 5000000
total_interactions = 0 # total number of the interactions, that the agent had so far (each stack of the frames is counted once).


# logging variables (accumulated over all episodes)
history_of_total_losses = []
history_of_total_rewards = []
episode_cnt = 0
num_of_last_episodes_to_avg = 100
log_display_step = 10000
start_time = time.time()

print(f'Starting the Training...')
while total_interactions < max_total_interactions: 
    episode_finished = False
    episode_total_loss = 0.0
    episode_total_reward = 0.0

    # initializing a new episode
    obs, info = wrapped_env.reset()
    obs = torch.tensor(obs)

    while not episode_finished:
        # chosing action - observing the outcome - storing in replay buffer - learning 
        action = agent.choose_action(obs.unsqueeze(0).to(device))
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        next_obs, action = torch.tensor(next_obs), torch.tensor(action)
        
        agent.store_transition(obs, action, reward, terminated or truncated, next_obs)
        loss = agent.learn()
        
        if loss == None: # it means that the replay buffer has not stored a sufficient number of transitions yet
            continue

        obs = next_obs

        # logging (accumlated over each episode)
        total_interactions += 1
        episode_finished = terminated or truncated
        episode_total_loss += loss
        episode_total_reward += reward

        # display logs every log_display_step + saving
        if (total_interactions % log_display_step) == 0 and (total_interactions > 0) and (episode_cnt >= num_of_last_episodes_to_avg):
            end_time = time.time()
            avg_loss_of_last_episodes = np.average(history_of_total_losses[-num_of_last_episodes_to_avg:])
            avg_reward_of_last_episodes = np.average(history_of_total_rewards[-num_of_last_episodes_to_avg:])
            print(f'Displaying Logs at the Frame {total_interactions}, Episode {episode_cnt}, Delta Time: {end_time - start_time}')
            print(f'Avg Loss Across {num_of_last_episodes_to_avg} Last Episodes = {avg_loss_of_last_episodes:.4f}')
            print(f'Avg Reward Across {num_of_last_episodes_to_avg} Last Episodes = {avg_reward_of_last_episodes:.4f}')
            start_time = end_time
            
            print(f'Evaluation:')
            evaluate(wrapped_env, agent, device)
            agent.save_model(f'{checkpoints_dir_path}agent_it_{total_interactions}.pt')


    # logging (accumulated over all episodes)
    history_of_total_losses.append(episode_total_loss)
    history_of_total_rewards.append(episode_total_reward)
    episode_cnt += 1
print(f'Training has been Finished!')

print(f'Storing the Model...')
agent.save_model(f'{checkpoints_dir_path}agent_{game_id.replace('/', '_')}.pt')

print(f'Plotting the Logs...')
plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards)