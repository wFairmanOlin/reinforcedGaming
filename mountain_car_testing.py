import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make('MountainCar-v0')
env.goal_position = 0
# print(env.goal_position)
#MountainCar-v0
#CartPole-v1
env.seed(1)
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = .99


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        num_hidden = 128

        self.l1 = nn.Linear(state_space, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, action_space, bias=False)

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.7),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


def predict(state):
    # Select an action (0 or 1) by running policy model
    # and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action_probs = policy(state)
    distribution = Categorical(action_probs)
    action = distribution.sample()

    # Add log probability of our chosen action to our history
    policy.episode_actions = torch.cat([
        policy.episode_actions,
        distribution.log_prob(action).reshape(1)
    ])

    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)

    # print(rewards)
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.episode_rewards))
    policy.reset()


def train(episodes):
    scores = []
    max_positions = []
    for episode in range(episodes):
        # Reset environment and record the starting state
        state = env.reset()
        max_pos = state[0]
        # env.gravity = 0

        for time in range(250):
            if episode % 1000 < 1:
                env.render()
            action = predict(state)

            # Uncomment to render the visual state in a window
            # env.render()

            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())
            reward = state[0] + 0.5
            if state[0] >= env.goal_position:
                reward += 50
            # print(state[0])
            if state[0] > max_pos:
                max_pos = state[0]
            # Save reward
            # if state[0] >= env.goal_position:
            #     reward = 50
            policy.episode_rewards.append(reward)

            if done or state[0] > env.goal_position:
                # if state[0] > env.goal_position:
                # print(state[0], env.goal_position)
                break

        update_policy()
        # print(policy.reward_history)

        # Calculate score to determine when the environment has been solved
        scores.append(time)
        max_positions.append(max_pos)
        mean_score = np.mean(scores[-100:])
        mean_pos = np.mean(max_positions[-100:])

        if episode % 100 == 0:
            print('Episode {}\Average length (last 100 episodes): {:.2f}'.format(
                episode, mean_score))
            print('Episode {}\ max pos (last 100 episodes): {:.2f}'.format(
                episode, mean_pos))

        # if mean_score > env.spec.reward_threshold:
        #     print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
        #           .format(episode, mean_score, time))
        #     break


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
train(episodes=10001)
