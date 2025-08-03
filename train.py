import gymnasium as gym
import ale_py
gym.register_envs(ale_py) # register the Atari environments

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from agent import Agent
from wrappers import AtariImage, ClipReward

def plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()

    x = np.arange(1, episode_cnt + 1)
    sns.lineplot(x=x, y=history_of_total_losses, ax=axs[0])
    axs[0].set_title('Total Loss over Different Episodes')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total MSE Loss')
    # axs[0].legend()

    sns.lineplot(x=x, y=history_of_total_rewards, ax=axs[1])
    axs[1].set_title('Total Rewards over each Episodes')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Total Reward')
    # axs[1].legend()

    plt.suptitle('Total Loss & Reward over each Episodes \n ' )

    plt.tight_layout()
    plt.show()


# cofiguration of the environment
game_id = 'ALE/Assault-v5'
max_total_interactions = 5000000
frame_skip = 4
env = gym.make(id=game_id)
clip_reward_wrapper = ClipReward(env)
atari_image_wrapper = AtariImage(clip_reward_wrapper)
# add other wrappers if needed
# ...
wrapped_env = atari_image_wrapper # set to the last applied wrapper for more convinent naming 

print(f'The Environment for the Game {game_id} has been Initialized.')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# configuration of the agent
agent = Agent(device=device) # we keep the arguments as default


# parameters of the training loop 
total_interactions = 0 # total number of the interactions, that the agent had so far (each stack of the frames is counted once).


# logging variables (accumulated over all episodes)
history_of_total_losses = []
history_of_total_rewards = []
episode_cnt = 0
num_of_last_episodes_to_avg = 100
log_display_step = 10000

print(f'Starting the Training...')
while total_interactions < max_total_interactions: 
    episode_finished = False
    episode_total_loss = 0.0
    episode_total_reward = 0.0

    # initializing a new episode
    obs, info = wrapped_env.reset()
    obs = torch.tensor(obs).to(device)

    while not episode_finished:
        # chosing action - observing the outcome - storing in replay buffer - learning 
        action = agent.choose_action(obs.unsqueeze(0))
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        next_obs, action = torch.tensor(next_obs).to(device), torch.tensor(action).to(device)
        
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
            avg_loss_of_last_episodes = np.average(history_of_total_losses[-num_of_last_episodes_to_avg:])
            avg_reward_of_last_episodes = np.average(history_of_total_rewards[-num_of_last_episodes_to_avg:])
            print(f'Displaying Logs at the Frame {total_interactions} and Episode {episode_cnt}:')
            print(f'Avg Loss Across {num_of_last_episodes_to_avg} Last Episodes = {avg_loss_of_last_episodes:.4f}')
            print(f'Avg Reward Across {num_of_last_episodes_to_avg} Last Episodes = {avg_reward_of_last_episodes:.4f}')

            agent.save_model(f'/saved_models/agent_{game_id}_it_{total_interactions}.pt')


    # logging (accumulated over all episodes)
    history_of_total_losses.append(episode_total_loss)
    history_of_total_rewards.append(episode_total_reward)
    episode_cnt += 1

print(f'Training has been Finished!')

print(f'Storing the Model...')
agent.save_model(f'/saved_models/agent_{game_id}.pt')

print(f'Plotting the Logs...')
plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards)