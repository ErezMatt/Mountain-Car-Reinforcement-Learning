import argparse
import numpy as np
#Workaround for gym compatibility
# np.bool8 = np.bool_
import gymnasium as gym

import matplotlib
matplotlib.get_backend()
import matplotlib.pyplot as plt

from environment import DiscreteEnvironment
from q_learning import QLearning

from models import DQN
from deep_q_learning import DeepQLearning
import torch

from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=['q_learning', 'deep_q_learning'], default="deep_q_learning", 
                        help="algorithm for training, possible values: (q_learning, deep_q_learning)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    env = gym.make('MountainCar-v0',  render_mode="rgb_array")
    
    #QLearning parameters
    if args.algorithm=="q_learning":
        #Discrete environment creation
        partitions = [20] * len(env.observation_space.high)
        action_number = env.action_space.n
        discrete_env = DiscreteEnvironment(env, env.observation_space.low, env.observation_space.high, partitions, action_number)

        #Qlearning Parameters
        alpha = 0.1
        gamma = 0.95
        episodes = 25000
        epsilon = 0.2
        epsilon_decay_value = 0.999
        interval_size = 500

        algorithm = QLearning(discrete_env=discrete_env, alpha=alpha, gamma=gamma)
    else:
        #Deep Qlearning Parameters
        alpha = 0.001
        gamma = 0.99
        episodes = 25000
        epsilon = 0.99
        epsilon_decay_value = 0.999
        interval_size = 500
        batch_size = 64
        replay_buffer_max_len = 1000
        replay_buffer_min_len = 500

        algorithm = DeepQLearning(env, DQN, batch_size, replay_buffer_max_len, replay_buffer_min_len, alpha, gamma, device)

    trainer = Trainer(algorithm=algorithm, epsilon=epsilon, epsilon_decay_value=0.999, interval_size=interval_size)
    history = trainer.train(episodes=episodes)

    trainer.test()
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
