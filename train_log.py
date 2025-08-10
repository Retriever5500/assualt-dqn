import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_logs(game_id, total_interactions, episode_cnt, history_of_total_losses, history_of_total_rewards, plot_dir_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()
    print(total_interactions, episode_cnt, len(history_of_total_losses), len(history_of_total_rewards))
    x = np.arange(1, episode_cnt + 1)
    print(len(x), len(history_of_total_losses))
    sns.lineplot(x=x, y=history_of_total_losses, ax=axs[0])
    axs[0].set_title('Total Training Loss over each Episode')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total MSE Loss')
    # axs[0].legend()

    sns.lineplot(x=x, y=history_of_total_rewards, ax=axs[1])
    axs[1].set_title('Total Training Rewards over each Episode')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Total Reward')
    # axs[1].legend()

    plt.suptitle(f'Total Training Loss & Reward over each Episode, over {total_interactions} iterations and {episode_cnt} episodes \n {game_id}' )

    plt.tight_layout()

    plot_path = f'{plot_dir_path}plot_{game_id.replace("/", "-")}_it_{total_interactions}'
    plt.savefig(plot_path)
    plt.show()