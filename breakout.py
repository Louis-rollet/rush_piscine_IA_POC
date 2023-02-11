import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Create fully-connected layers with ReLU activations
        self.fc1 = torch.nn.Linear(33600 * 3, 210)
        self.fc2 = torch.nn.Linear(210, 4)

        self.actions, self.states, self.rewards = [], [], []

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = x.view(-1, 33600 * 3)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def remember(self, Action, State, Reward):
        self.actions.append(Action)
        self.states.append(State)
        self.rewards.append(Reward)

    def policy_action(self, state):
        # Get the probabilities for each action, using the current state
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action

    def discount_rewards(self):
        rewards = []
        for t in range(len(self.rewards)):
            gt = 0
            pw = 0
            for r in self.rewards[t:]:
                gt += gamma ** (pw - t - 1) * r
                pw += 1
            rewards.append(gt)
        return rewards

    def gradient_ascent(self, discounted_rewards):
        for State, Action, G in zip(self.states, self.actions, discounted_rewards):
            probs = self.forward(State)
            if Action >= probs.size(0):
                print(Action, probs.size(0))
                Action = 0
            loss = -torch.log(probs[Action]) * G

            optim.zero_grad()
            loss.backward()
            optim.step()


# Initialize empty lists for rewards and losses
recent_rewards = deque(maxlen=100)
train_rewards = []
train_loss = []


lr = 0.001
gamma = 0.995
env_name = 'ALE/Breakout-v5'
env = gym.make(env_name, render_mode="rgb_array")
network = NeuralNetwork(env)
optim = torch.optim.Adam(network.parameters(), lr=lr)
episodes = 1000

for episode in range(episodes):
    # Reset the environment and initialize empty lists for actions, states, and rewards
    state = env.reset()
    network.actions, network.states, network.rewards = [], [], []

    # Train the agent for a single episode
    for _ in range(1000):
        action = network.policy_action(state)

        # Take the action in the environment and get the new state, reward, and done flag
        new_state, reward, termination, truncation = env.step(action.item())

        # Save the action, state, and reward for later
        network.remember(action, state, reward)
        state = new_state

        # If the episode is done or the time limit is reached, stop training
        if termination or truncation:
            break

    # Perform gradient ascent
    network.gradient_ascent(network.discount_rewards())

    # Save the total reward for the episode and append it to the recent rewards queue
    train_rewards.append(np.sum(network.rewards))
    recent_rewards.append(train_rewards[-1])

    # Print the mean recent reward every 50 episodes
    if episode % 50 == 0:
        print(f"Episode {episode:>6}: \tR:{np.mean(recent_rewards):>6.3f}")

    if np.mean(recent_rewards) > 400:
        print(f"Episode {episode:>6}: \tR:{np.mean(recent_rewards):>6.3f}")
        break
env.close()
