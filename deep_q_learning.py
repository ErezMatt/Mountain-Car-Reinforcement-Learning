import copy
import os
import numpy as np
import cv2
import random
import torch
from torch import nn
from torch import optim
from collections import deque

import models

class DeepQLearning:
    def __init__(self, env, network, batch_size, replay_buffer_max_len, replay_buffer_min_len, alpha, gamma, device="cpu"):
        self.device = device

        self.env = env
        
        self.model = network(env.observation_space.shape[0], env.action_space.n).to(self.device)
        
        self.target_model = network(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.replay_buffer_min_len = replay_buffer_min_len
        self.replay_buffer_max_len = replay_buffer_max_len
        self.replay_buffer = deque(maxlen=self.replay_buffer_max_len)

        self.alpha = alpha
        self.gamma = gamma
        self.target_update_steps = 0

    def select_action(self, state, epsilon):
        if np.random.random() > epsilon:
            state_tensor = torch.Tensor(state).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()
        else:
            return np.random.randint(0, self.env.action_space.n)
    
    def train(self, state, action, new_state, reward, terminate_state):
        self.replay_buffer.append((state, action, reward, new_state, terminate_state))
        self.train_network()
 
    def train_network(self):
        if len(self.replay_buffer) < self.replay_buffer_min_len:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)

        for (state, action, reward, new_state, done) in batch:
            state = torch.Tensor(state).to(self.device)
            new_state = torch.Tensor(new_state).to(self.device)
            reward = torch.Tensor([reward]).to(self.device)

            if not done:
                new_q = reward + self.gamma * torch.max(self.target_model(new_state))
            else:
                new_q = reward

            model_prediction = self.model(state)
            true_value = model_prediction.clone()
            true_value[action] = new_q

            loss = self.criterion(model_prediction, true_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update(self, episode):
        if episode % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save_checkpoint(self, path, episode):
        file_path = f'{path}/deep_q_learning/'
        if not(os.path.exists(file_path)):
            os.mkdir(file_path)
        file_path += f'model_ep_{episode}.pth'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)

    def load_checkpoint(self, file):
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def step(self, action):
        return self.env.step(action)
    
    def reset_env(self):
        return self.env.reset()[0]
    
    def render_env(self):
        return self.env.render()
