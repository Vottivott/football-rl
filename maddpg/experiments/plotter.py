import numpy as np
import matplotlib.pyplot as plt

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def produce_graph(episode_rewards_smooth, episode_lengths_smooth, episode_rewards_hi, episode_lengths_hi, episode_rewards_lo, episode_lengths_lo, filename, games_per_expfile):
    episode_groups = range(0,len(episode_rewards_smooth)*games_per_expfile,games_per_expfile)
    plt.clf()
    plt.figure(1, figsize=(5, 6 * 3.13), facecolor='whitesmoke')
    num_plots = 2
    current_plot = 1
    plt.subplot(num_plots, 1, current_plot)
    plt.subplots_adjust(hspace=0.3) #0.2 = default
    plt.title('Total abs(reward) for each episode')
    plt.fill_between(episode_groups, episode_rewards_lo, episode_rewards_hi,color="lightgray")
    plt.plot(episode_groups, episode_rewards_smooth, "r")#,
             #stats["generations"], stats["best_fitness_last_only"], "m",
             #stats["generations"], stats["avg_fitness"], "b")
    current_plot += 1

    plt.subplot(num_plots, 1, current_plot)
    plt.title('Episode length')
    plt.fill_between(episode_groups, episode_lengths_lo, episode_lengths_hi,color="lightgray")
    plt.plot(episode_groups, episode_lengths_smooth, "c")
    current_plot += 1

    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":

    episode_rewards_smooth = [0, 4, 4, 5, 9]
    episode_lengths_smooth = [0, 4, 3, 5, 9]
    episode_rewards_hi = [0, 2.5, 1.5, 2.3, 5.1]
    episode_lengths_hi = [1.4, 1.35, 1.32, 1.28, 1.23]
    episode_rewards_lo = [0.2, 0.2, 0.2, 0.3, 1.4]
    episode_lengths_lo = [0.2, 0.2, 0.2, 1.3, 0.4]
    
    produce_graph(episode_rewards_smooth, episode_lengths_smooth, episode_rewards_hi, episode_lengths_hi, episode_rewards_lo, episode_lengths_lo, "./test.png")
