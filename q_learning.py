import numpy as np
import cv2
from PIL import Image

class QLearning:
    def __init__(self, discrete_env, alpha, gamma, epsilon):
        self.discrete_env = discrete_env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.random.uniform(low=-2, high=0, size=(discrete_env.number_of_partitions + [discrete_env.action_number]))

        self.episodes_rewards = []
        self.history = {'episode' : [], 'avg' : [], 'min' : [], 'max': []}

    def train(self, episodes, epsilon_decreas_start, epsilon_decreas_end, interval_size):
        epsilon_decreas_value = self.epsilon / (epsilon_decreas_end - epsilon_decreas_start)
        for episode in range(episodes):
            episode_rewards = 0
            terminate_state = False
            state = self.discrete_env.reset()

            while not terminate_state:
                #Select action
                state_actions = self.q_table[state]

                if np.random.random() > self.epsilon :
                    action = np.argmax(state_actions)
                else:
                    action = np.random.randint(0, self.discrete_env.action_number)

                #Get new state and reward
                new_state, reward, term, trun, _  = self.discrete_env.step(action)
                episode_rewards += reward

                #Update Q Table
                q_prime = np.max(self.q_table[new_state])
                current_q = self.q_table[state + (action, )]

                new_q = current_q + self.alpha * (reward + self.gamma * q_prime - current_q)
                self.q_table[state + (action, )] = new_q

                state = new_state

                if term or trun:
                    terminate_state = True

                #Show env animation
                if episode % interval_size == 0:
                        frame = cv2.cvtColor(self.discrete_env.env.render(), cv2.COLOR_RGB2BGR)
                        cv2.imshow("frame", frame)
                        cv2.waitKey(50)

            if epsilon_decreas_end >= episode >= epsilon_decreas_start:
                self.epsilon  -= epsilon_decreas_value

            self.episodes_rewards.append(episode_rewards)

            if episode % interval_size == 0:
                history_interval = self.episodes_rewards[-interval_size:]

                average_interval_reward = sum(history_interval)/len(history_interval)
                min_interval_reward = min(history_interval)
                max_interval_reward = max(history_interval)

                self.history['episode'].append(episode)
                self.history['avg'].append(average_interval_reward)
                self.history['min'].append(min_interval_reward)
                self.history['max'].append(max_interval_reward)

                print(f' Episode: {episode} reward: {episode_rewards} avg: {average_interval_reward} min: {min_interval_reward} max: {max_interval_reward}')

        return self.history

    def save_strategy(self, file_name):
         np.save(file_name, self.q_table)

    def load_strategy(self, file_name):
         self.q_table = np.load(file_name)

    def run_strategy(self):
        episode_rewards = 0
        terminate_state = False
        state = self.discrete_env.reset()
        while not terminate_state:
                #Select best action
                state_actions = self.q_table[state]
                action = np.argmax(state_actions)

                #Get new state and reward
                state, reward, term, trun, _  = self.discrete_env.step(action)
                episode_rewards += reward

                frame = cv2.cvtColor(self.discrete_env.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("frame", frame)
                cv2.waitKey(50)

                if term or trun:
                    terminate_state = True
  
        print(f'Strategy rewards: {episode_rewards}')
