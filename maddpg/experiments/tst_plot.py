import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_single(varname):
    fname = varname + ".pkl"
    with open("../../saved_variables/" + fname, 'rb') as fp:
        return pickle.load(fp)


def load(*varnames):
    return tuple(load_single(varname) for varname in varnames)

episode_rewards, episode_lengths, episode_rewards_smooth, episode_lengths_smooth = load("episode_rewards", "episode_lengths", "episode_rewards_smooth", "episode_lengths_smooth")
plt.plot(episode_lengths)
plt.show()
