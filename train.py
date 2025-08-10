import gymnasium as gym
import ale_py
gym.register_envs(ale_py) # register the Atari environments
from gymnasium.wrappers import TimeLimit

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import time
import math

from agent import Agent
from wrappers import AtariImage, ClipReward, FireResetWithoutEpisodicLife, FireResetWithEpisodicLife, EpisodicLifeEnv, BreakoutActionTransform
from eval import evaluate
from train_log import plot_logs

def create_proj_dirs(checkpoints_dir_path='saved_models/', plots_dir_path='plots/'):
    print(f'Creating directory for saving checkpoints & plots...')
    if not os.path.exists(checkpoints_dir_path):
        os.makedirs(checkpoints_dir_path)
        print(f"Directory '{checkpoints_dir_path}' created for saving checkpoints!")
    else:
        print(f"Directory '{checkpoints_dir_path}' already exists for saving checkpoints!")
    
    if not os.path.exists(plots_dir_path):
        os.makedirs(plots_dir_path)
        print(f"Directory '{plots_dir_path}' created for saving plots!")
    else:
        print(f"Directory '{plots_dir_path}' already exists for saving plots!")

    return checkpoints_dir_path, plots_dir_path







# directory creation for checkpoints
checkpoints_dir_path, plots_dir_path = create_proj_dirs()

# cofiguration of the environment
game_id = 'BreakoutNoFrameskip-v4'
num_of_lives_in_each_game = 5
env = gym.make(id=game_id, frameskip=1)
wrappers_lst = [(EpisodicLifeEnv, {}), 
                (FireResetWithEpisodicLife, {}), 
                (ClipReward, {}), 
                (AtariImage, {'image_shape':(84, 84), 'frame_skip': 4}), 
                (TimeLimit, {'max_episode_steps': 10000})] # each stack of frames is counted once
wrapped_env = env
for wrapper, kwargs in wrappers_lst:
    wrapped_env = wrapper(wrapped_env, **kwargs)
print(f'The environment for the game {game_id} has been initialized!')

# configuration of the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# configuration of the agent
agent = Agent(num_of_actions=wrapped_env.action_space.n, device=device) # we keep the arguments as default


# parameters of the training loop 
max_total_interactions = 5000000
total_interactions = 0 # total number of the interactions, that the agent had so far (each stack of the frames is counted once).

# variables to keep track of the best parameters based on evaluation scores
best_eval_mean = - math.inf


# logging variables
history_of_total_losses = []
history_of_total_rewards = []
episode_cnt = 0
using_episodic_life = EpisodicLifeEnv in [t[0] for t in wrappers_lst]
scaling_factor = num_of_lives_in_each_game if using_episodic_life else 1
num_of_last_episodes_to_avg = 100 * scaling_factor
log_display_step = 10000
start_time = time.time()

print(f'Starting the training...')
while total_interactions < max_total_interactions: 
    episode_finished = False
    episode_total_loss = 0.0
    episode_total_reward = 0.0

    # initializing a new episode
    obs, info = wrapped_env.reset()
    obs = torch.tensor(obs)

    while not episode_finished:
        # chosing action - observing the outcome
        action = agent.choose_action(obs.unsqueeze(0).to(device))
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        next_obs, action = torch.tensor(next_obs), torch.tensor(action)
        
        # storing in replay buffer - learning 
        agent.store_transition(obs, action, reward, terminated or truncated, next_obs)
        loss = agent.learn()
        
        # updating current last observation
        obs = next_obs

        # updating logs per step in each episode
        total_interactions += 1
        episode_finished = terminated or truncated
        episode_total_loss += loss if loss is not None else 0
        episode_total_reward += reward
 

        if (total_interactions % log_display_step) == 0 and (total_interactions > 0) and (episode_cnt >= num_of_last_episodes_to_avg):
            print(f'<------- Displaying logs at the frame {total_interactions}, episode {episode_cnt} ------->')

            # printing training logs
            print(f'Training logs over the last {num_of_last_episodes_to_avg} episodes:')
            avg_loss_of_last_episodes = np.mean(history_of_total_losses[-num_of_last_episodes_to_avg:])
            avg_reward_of_last_episodes = np.mean(history_of_total_rewards[-num_of_last_episodes_to_avg:])
            std_loss_of_last_episodes = np.std(history_of_total_losses[-num_of_last_episodes_to_avg:])
            std_reward_of_last_episodes = np.std(history_of_total_rewards[-num_of_last_episodes_to_avg:])
            print(f'Loss: mean:{avg_loss_of_last_episodes:.4f}, std:{std_loss_of_last_episodes:.4f}')
            print(f'Reward: mean:{avg_reward_of_last_episodes:.4f}, std:{std_reward_of_last_episodes:.4f}')

            # printing evaluation logs            
            eval_mean, eval_var = evaluate(wrapped_env, agent, device, num_of_lives_in_each_game=num_of_lives_in_each_game, using_episodic_life=using_episodic_life)

            # TODO - plotting training and storing plots as images
            plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards, plots_dir_path)

            # TODO - updating the best model according to mean evaluation scores and saving
            if eval_mean > best_eval_mean:
                print(f"Changing best model: evaluation reward mean improved by {eval_mean - best_eval_mean:.4f}! (showing up to 4 decimal places)")
                best_eval_mean = eval_mean
                agent.save_model(f'{checkpoints_dir_path}best_model_it_{total_interactions}.pt')

            # saving the checkpoint
            agent.save_model(f'{checkpoints_dir_path}agent_it_{total_interactions}.pt')

            end_time = time.time()
            print(f"delta time {end_time - start_time:.1f}")
            start_time = end_time

            print('\n')


    # updating logs per episode
    history_of_total_losses.append(episode_total_loss)
    history_of_total_rewards.append(episode_total_reward)
    episode_cnt += 1
print(f'Training has been finished!')

print(f'Storing the model...')
agent.save_model(f'{checkpoints_dir_path}agent_{game_id.replace("/", "_")}.pt')

print(f'Plotting the final logs...')
plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards)