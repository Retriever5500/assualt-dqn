import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def rolling_average(data, window_size):
    """Calculate the rolling average of a given data list."""
    averages = []
    for i in range(len(data)):
        # Calculate the average from the start to the current index if i < window_size
        if i < window_size:
            avg = np.mean(data[:i + 1])  # Average from the start to the current index
        else:
            avg = np.mean(data[i - window_size + 1:i + 1])  # Average of the last 'window_size' elements
        averages.append(avg)
    return averages


def plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards, plot_dir_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    x = np.arange(1, episode_cnt + 1)

    rolling_avg_losses = rolling_average(history_of_total_losses, 200)
    rolling_avg_rewards = rolling_average(history_of_total_rewards, 200)

    sns.lineplot(x=x, y=rolling_avg_losses, ax=axs[0])
    axs[0].set_title('Total Training Loss over each Episode')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total MSE Loss')
    # axs[0].legend()

    sns.lineplot(x=x, y=rolling_avg_rewards, ax=axs[1])
    axs[1].set_title('Total Training Rewards over each Episode')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Total Reward')
    # axs[1].legend()

    plt.suptitle(f'Total Training Loss & Reward over each Episode, over {total_interactions} iterations and {episode_cnt} episodes \n {game_id}' )

    plt.tight_layout()

    plot_path = f'{plot_dir_path}plot_{game_id.replace("/", "-")}_it_{total_interactions}'
    plt.savefig(plot_path)
    plt.show()