import argparse
import torch
import numpy as np
import gymnasium as gym
from agent import Agent
from wrappers import AtariImage, ClipReward, FireResetWithoutEpisodicLife, FireResetWithEpisodicLife, EpisodicLifeEnv


def test_model(model_path, env_name, total_games=3):
    # configuration of the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(id=env_name, render_mode="human", frameskip=1)
    wrappers_lst = [ClipReward, AtariImage] # Add other wrappers if it's used when we trained the agent
    wrapped_env = env
    for wrapper in wrappers_lst:
        wrapped_env = wrapper(wrapped_env)
    print(f'The Environment for the Game {game_id} has been Initialized.')
    num_of_actions = wrapped_env.action_space.n
    agent = Agent(num_of_actions=num_of_actions, device=device)
    agent.load_model(model_path)

    for i in range(total_games):
        obs, info = wrapped_env.reset()
        done = False

        while not done:
            action_index = agent.choose_action(torch.from_numpy(obs).unsqueeze(0).to(device), eps=0.05)
            obs, reward, done, truncated, info = wrapped_env.step(action_index)

            if done:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_id', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()
    game_id = args.game_id
    model_path = args.model_path

    test_model(model_path, game_id)
