import torch


def evaluate(env, agent, device, games_count=10, num_of_lives_in_each_game=1, using_episodic_life=False):
    scaling_factor = num_of_lives_in_each_game if using_episodic_life else 1

    scores = []
    for i in range(games_count * scaling_factor):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_index = agent.choose_action(torch.tensor(obs).unsqueeze(0).to(device), eps=0.05)

            obs, reward, done, truncated, info = env.step(action_index)
            total_reward += reward

        scores.append(total_reward)

    mean_scores = sum(scores)/len(scores)

    print(f"Mean score: {mean_scores:.1f}")
