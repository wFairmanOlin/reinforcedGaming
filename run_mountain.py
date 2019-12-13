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
import gym_mountain
env = gym.make('MountainCar-v0')
#MountainCar-v0
#CartPole-v1
# env.seed(1)
# torch.manual_seed(1)

# Hyperparameter


fileName = str(input('episode (increments of 500): '))

print('correct file')

policy = gym_mountain.Policy()
policy.load_state_dict(torch.load('mountainNets/' + fileName + '.pth'))
policy.eval()


def run(episodes):
    for episode in range(episodes):

        # Reset environment and record the starting state
        state = env.reset()
        for time in range(1000):
            env.render()
            action = gym_mountain.predict(policy, state)

            state, reward, done, _ = env.step(action.item())

            if(state[0] > .5):
                print('won in: ', time)
                break
        if (state[0] < .5):
            print('lost episode: ', episode)


run(episodes=100)

