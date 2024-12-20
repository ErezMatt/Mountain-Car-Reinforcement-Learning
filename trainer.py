import os
import numpy as np
import cv2
from PIL import Image

class Trainer:
    def __init__(self, algorithm, epsilon, epsilon_decay_value, interval_size):
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.epsilon_decay_value = epsilon_decay_value
        self.min_epsilon_value = 0.001
        self.interval_size = interval_size

        self.episodes_rewards = []
        self.history = {'episode' : [], 'avg' : [], 'min' : [], 'max': []}

    def train(self, episodes):
        for episode in range(episodes+1):
            episode_rewards = 0
            terminate_state = False
            state = self.algorithm.reset_env()
            
            while not terminate_state:
                #Select action
                action = self.algorithm.select_action(state, self.epsilon)

                #Get new state and reward
                new_state, reward, term, trun, _  = self.algorithm.step(action)
                episode_rewards += reward

                if term or trun:
                    terminate_state = True

                self.algorithm.train(state, action, new_state, reward, terminate_state)
                state = new_state

                #Show env animation
                if episode % self.interval_size == 0:
                        frame = cv2.cvtColor(self.algorithm.render_env(), cv2.COLOR_RGB2BGR)
                        cv2.imshow("frame", frame)
                        cv2.waitKey(50)

            self.algorithm.update(episode)

            if self.epsilon > self.min_epsilon_value:
                self.epsilon *= self.epsilon_decay_value
            else:
                 self.epsilon = 0

            self.episodes_rewards.append(episode_rewards)

            if episode % self.interval_size == 0:
                self.save_checkpoint("./checkpoints", episode)

                history_interval = self.episodes_rewards[-self.interval_size:]

                average_interval_reward = sum(history_interval)/len(history_interval)
                min_interval_reward = min(history_interval)
                max_interval_reward = max(history_interval)

                self.history['episode'].append(episode)
                self.history['avg'].append(average_interval_reward)
                self.history['min'].append(min_interval_reward)
                self.history['max'].append(max_interval_reward)

                print(f' Episode: {episode} reward: {episode_rewards} avg: {average_interval_reward} min: {min_interval_reward} max: {max_interval_reward}')

        return self.history

    def save_checkpoint(self, path, episode):
         if not(os.path.exists(path)):
            os.mkdir(path)

         self.algorithm.save_checkpoint(path, episode)

    def load_checkpoint(self, file_path):
         self.algorithm.load_checkpoint(file_path)

    def test(self):
        episode_rewards = 0
        terminate_state = False
        state = self.algorithm.reset_env()
        while not terminate_state:
                #Select best action
                action = self.algorithm.select_action(state, 0)

                #Get new state and reward
                state, reward, term, trun, _  = self.algorithm.step(action)
                episode_rewards += reward
                
                #Show env animation
                frame = cv2.cvtColor(self.algorithm.render_env(), cv2.COLOR_RGB2BGR)
                cv2.imshow("frame", frame)
                cv2.waitKey(50)

                if term or trun:
                    terminate_state = True
  
        print(f'Strategy rewards: {episode_rewards}')
