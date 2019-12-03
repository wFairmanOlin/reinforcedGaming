import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import time as t
env = gym.make('MountainCar-v0')

# env.seed(1)
# torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.005
gamma = 0.99


def saveModel(model, fileName):
    fileName = str(fileName)
    path = 'mountainNets/' + fileName + '.pth'
    torch.save(model.state_dict(), path)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        num_hidden = 128

        self.l1 = nn.Linear(state_space, 64, bias=False)
        self.l2 = nn.Linear(64, 128, bias=False)
        self.l3 = nn.Linear(128, action_space, bias=False)

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
            nn.Dropout(p=0.4),
            nn.ReLU(),
            self.l2,
            nn.Dropout(p=0.2),
            nn.ReLU(),
            self.l3,
            nn.Softmax(dim=-1)
        )
        return model(x)


def predict(policy, state):
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
    # print(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1))
    # print(loss)

        # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.episode_rewards))
    policy.reset()

    return loss.item()


def train(episodes):
    scores = []
    max_pos = []
    losses = []
    for episode in range(episodes):

        #save every 250 episodes
        if episode % 250 == 0:
            saveModel(policy, episode)

        # Reset environment and record the starting state
        state = env.reset()
        maxp = -.5
        for time in range(2500):
            if episode % 200 == 0:
                env.render()
            action = predict(policy, state)


            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())

            if state[0] > maxp:
                maxp = state[0]

            # Save reward
            if state[0] > .5:
                policy.episode_rewards.append(100)
                break

            else:
                # -((.5-abs(state[0] + .5))) + abs(state[1])
                policy.episode_rewards.append(reward)
                

        losses.append(update_policy())

        # Calculate score to determine when the environment has been solved
        scores.append(time)
        max_pos.append(maxp)
        mean_score = np.mean(scores[-100:])
        mean_max = np.mean(max_pos[-100:])
        mean_loss = np.mean(losses[-100:])

        if episode % 50 == 0:
            print('Episode {} Average time (last 100 episodes): {:.2f} avg loss: {:.8f} avg max-pos: {:.2f}'.format(
                episode, mean_score, mean_loss, mean_max))

        if mean_score < 200:
            saveModel(policy, 'winning')
            print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
                  .format(episode, mean_score, time))
            break



if __name__ == "__main__":
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    train(episodes=20000)

