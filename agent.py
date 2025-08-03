import random
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from network import DQNNet
from memory import Memory
import copy

class Agent:
    def __init__(self, num_of_actions=4, network=None, lr=0.00025, gamma=0.99, eps=1.0,
        eps_fframe=1e6, eps_final=0.1, minibatch_size=32, min_training_step=1000,
        max_num_transitions=50000, target_interval=10000, device="cpu"):   

        self.num_of_actions = num_of_actions
        if(network == None):
            network = DQNNet(num_of_actions)
        self.network = network
        self.target_network = copy.deepcopy(network)
        self.target_interval = target_interval
        self.learn_count = 0
        # Hyperparameters taken from the paper
        self.optim = torch.optim.RMSprop(network.parameters(), lr=lr, alpha=0.95, eps=0.01, momentum=0.95)
        self.minibatch_size = minibatch_size

        self.eps = eps
        self.eps_final = eps_final
        self.eps_step = (eps - eps_final) / eps_fframe
        self.gamma = gamma
        self.min_training_step = min_training_step

        self.memory = Memory(max_num_transitions=max_num_transitions)
        self.device = device

    def load_model(self, model_path):
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_model(self, model_path):
        torch.save(self.network.state_dict(), model_path)

    def store_transition(self, obs, action, reward, done, next_obs):
        self.memory.append(obs, action, reward, done, next_obs)

    def choose_action(self, obs, eps=None):
        if(eps == None):
            eps = self.eps

        if(random.random() < eps):
            return random.randint(0, self.num_of_actions - 1)
        else:
            with torch.no_grad():
                action_values = self.network(obs)
                return torch.argmax(action_values).item()

    def learn(self):
        if(self.memory.size < self.min_training_step):
            return

        obss, actions, rewards, dones, next_obss = self.memory.sample(self.minibatch_size)

        ys = rewards + 0.0
    
        with torch.no_grad():
            next_qvals = self.target_network(next_obss)
            ys[dones == 0] += self.gamma * torch.max(next_qvals, dim=1)[0][dones == 0]
        
        qvals = self.network(obss)
        ys_p = qvals[torch.arange(qvals.size(0), device=self.device), actions]

        loss = F.mse_loss(ys, ys_p)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.eps = max(self.eps - self.eps_step, self.eps_final)

        self.learn_count += 1

        if(self.learn_count % self.target_interval == 0):
            self.target_network = copy.deepcopy(self.network)
            print("Updated target network")

        return loss.item()
