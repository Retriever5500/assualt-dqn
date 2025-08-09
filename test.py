import argparse
import torch
import numpy as np
import gymnasium as gym
from agent import Agent
from wrappers import AtariImage, ClipReward, FireResetWithoutEpisodicLife, FireResetWithEpisodicLife, EpisodicLifeEnv


def test_model(model_path, env_name, total_games=3, num_of_lives_in_each_game=1, using_episodic_life=False):
    scaling_factor = num_of_lives_in_each_game if using_episodic_life else 1

    # configuration of the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = gym.make(id=game_id, frameskip=1, repeat_action_probability=0, render_mode='human')
    wrappers_lst = [FireResetWithoutEpisodicLife, ClipReward, AtariImage] # Add other wrappers if it's used when we trained the agent
    wrapped_env = env
    for wrapper in wrappers_lst:
        wrapped_env = wrapper(wrapped_env)
    print(f'The Environment for the Game {game_id} has been Initialized.')
    num_of_actions = wrapped_env.action_space.n
    agent = Agent(num_of_actions=num_of_actions, device=device)
    agent.load_model(model_path)

    for i in range(total_games * scaling_factor):
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
    parser.add_argument('--num_of_lives', type=int, required=True)
    parser.add_argument('--using_episodic_life', type=bool, required=True)

    args = parser.parse_args()
    game_id = args.game_id
    model_path = args.model_path
    num_of_lives_in_each_game = args.num_of_lives
    using_episodic_life = args.using_episodic_life

    test_model(model_path, game_id, 3, num_of_lives_in_each_game, using_episodic_life)
