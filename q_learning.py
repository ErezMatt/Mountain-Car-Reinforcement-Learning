import numpy as np
import os
import cv2
from PIL import Image

class QLearning:
    def __init__(self, discrete_env, alpha, gamma):
        self.discrete_env = discrete_env
        self.alpha = alpha
        self.gamma = gamma

        self.q_table = np.random.uniform(low=-2, high=0, size=(discrete_env.number_of_partitions + [discrete_env.action_number]))
    
    def select_action(self, state, epsilon):
        state_actions = self.q_table[state]

        if np.random.random() > epsilon:
            return np.argmax(state_actions)
        else:
            return np.random.randint(0, self.discrete_env.action_number)
    
    def train(self, state, action, new_state, reward, terminate_state):
        #Update Q Table
        q_prime = np.max(self.q_table[new_state])
        current_q = self.q_table[state + (action, )]

        new_q = current_q + self.alpha * (reward + self.gamma * q_prime - current_q)
        self.q_table[state + (action, )] = new_q

    def update(self, episode):
        pass
    
    def save_checkpoint(self, path, episode):
        file_path = f'{path}/q_learning/'
        if not(os.path.exists(file_path)):
            os.mkdir(file_path)

        file_path += f'ep_{episode}.npy'
        np.save(file_path, self.q_table)

    def load_checkpoint(self, file_path):
        self.q_table = np.load(file_path)

    def step(self, action):
        return self.discrete_env.step(action)
    
    def reset_env(self):
        return self.discrete_env.reset()
    
    def render_env(self):
        return self.discrete_env.env.render()
