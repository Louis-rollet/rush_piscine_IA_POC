import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision import transforms as T
from collections import deque


class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.conv1 = nn.Conv2d(84, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(33600 * 3, 210)
        self.fc2 = torch.nn.Linear(210, 4)

        self.actions, self.states, self.rewards = [], [], []

    def forward(self, x):
        x = torch.FloatTensor(x)
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        print(x.shape)
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
            print(State.shape)
            print(State)
            probs = self.forward(State)
            loss = -torch.log(probs[Action]) * G

            optim.zero_grad()
            loss.backward()
            optim.step()


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Initialize empty lists for rewards and losses
recent_rewards = deque(maxlen=100)
train_rewards = []
train_loss = []


lr = 0.001
gamma = 0.995
env_name = 'ALE/Breakout-v5'
env = gym.make(env_name, render_mode="rgb_array")
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
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
        state, total_reward, done, trunk = env.step(action)

        # Save the action, state, and reward for later
        network.remember(action, state, total_reward)
        state = state

        # If the episode is done or the time limit is reached, stop training
        if done or trunk:
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
