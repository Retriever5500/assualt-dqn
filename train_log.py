import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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