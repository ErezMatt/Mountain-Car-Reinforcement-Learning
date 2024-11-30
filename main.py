import numpy as np
#Workaround for gym compatibility
np.bool8 = np.bool_
import gym

import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt

from environment import DiscreteEnvironment
from q_learning import QLearning

if __name__ == '__main__':
    #Discrete environment creation
    env = gym.make('MountainCar-v0',  render_mode="rgb_array")
    partitions = [20] * len(env.observation_space.high)
    action_number = env.action_space.n
    discrete_env = DiscreteEnvironment(env, env.observation_space.low, env.observation_space.high, partitions, action_number)

    #QLearning parameters
    alpha = 0.1
    gamma = 0.95
    episodes = 25000
    epsilon = 0.2
    epsilon_decreas_start = 1
    epsilon_decreas_end = episodes // 2
    interval_size = 500

    q_learning = QLearning(discrete_env=discrete_env, alpha=alpha, gamma=gamma, epsilon=epsilon)
    history = q_learning.train(episodes=episodes, 
                               epsilon_decreas_start=epsilon_decreas_start, 
                               epsilon_decreas_end=epsilon_decreas_end,
                               interval_size=interval_size)
    q_learning.save_strategy("strategy.npy")
    q_learning.load_strategy("strategy.npy")

    q_learning.run_strategy()
    env.close()

    #Plot training results
    plt.plot(history['episode'], history['avg'], label = "avg")
    plt.plot(history['episode'], history['min'], label = "min")
    plt.plot(history['episode'], history['max'], label = "max")
    plt.title("Training results")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
