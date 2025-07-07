"""
ddpg_agent.py
-------------
Defines the actor-critic architecture and training logic for DDPG-style RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters
BUFFER_SIZE = 200000
BATCH_SIZE = 128
GAMMA = 0.98
TAU = 0.005
ACTOR_LR = 5e-5
CRITIC_LR = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []

    def size(self):
        return len(self.buffer)


class OUNoise:
    def __init__(self, size, mu=0, theta=0.10, sigma=0.8, decay=0.999995):
        self.mu = np.ones(size) * mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.decay = decay

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * (np.random.randn(len(self.state)))
        self.state += dx
        return self.state

    def decay_noise(self):
        self.sigma *= self.decay


class DDPGAgent:
    def __init__(self, state_size, action_size, max_action):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.target_actor = Actor(state_size, action_size, max_action).to(device)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_size)

        self.soft_update(self.target_actor, self.actor, tau=1.0)
        self.soft_update(self.target_critic, self.critic, tau=1.0)

    def act(self, state, noise=True):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise:
            action += self.noise.sample()
        return np.tanh(action)

    def update(self):
        if self.replay_buffer.size() < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_Q = rewards + GAMMA * self.target_critic(next_states, next_actions) * (1 - dones)

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, TAU)
        self.soft_update(self.target_critic, self.critic, TAU)

        self.noise.decay_noise()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
