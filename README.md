
# reinforcedGaming
For our final project we wanted to focus on how reinforcement learning can be used to train game-solving neural nets. Throughout the course of project we utilized OpenAI's [Gym library](https://gym.openai.com/) to create game environments that were easily compatible with the algorithms we were implementing.

## Overview
Reinforcement learning is a branch of machine learning that aims to train a model by taking action and observing how the environment reacts to any and all actions taken. A very common depiction of this feedback loop is shown below.

![overview](/pics/basicRL.png)

In general, a model's actions are determined by the model's agent. The agent makes decisions based on the previous state of the environment. After taking a new action as input, the environment enters a new state and outputs a reward. In most cases a reinforcement algorithm wants to maximize its reward by tuning the agent to take an action with the highest probability of generating a reward.

### Q Function
In a RL model, the action an agent takes is based off of the **Q function**. This function takes the current state and a future action as inputs and returns the probability of that future action leading to a high reward.
```
Q(state, action) = probability of high reward
```
### Deep Q-Learning
In deep Q-Learning, the **Q function** is defined as a neural network with an input layer connecting to each environmental variable and an output layer connecting to all valid actions. For any given state, the neural network will output the probability of each action generating a high reward.

## Training
In order to create an effective neural network Q function we have to either maximize the reward or minimize the loss of our network. This is achieved by stepping the network towards higher rewards or lower losses. For our project, we focused on generating a loss function and performing gradient descent to minimize the amount of loss our Q function would return.

### Loss Function
Our loss function is constantly computing the difference between our current q function and the ideal q function.
```
loss.backwards()
```

## Process
We started by finding an environment we could use to start trying out some reinforcement learning techniques without having to build an entire game from scratch. We settled on the Gym library as described above, because its many games have all the utilities we were looking for. Each game can be imported into a python script, which means we could easily create something to control the "character" in the game. Additionally, the games also have information about the state of the environment (like where the character is or which direction it is moving) which is crucial for the reinforcement algorithm to have something to learn from.

## CartPole
The first game we decided to try out is CartPole, This game involves a cart with a pole attached to the top, and the goal is to keep the pole balanced and upright as long as possible. The cart can either move left or right, and you lose if the pole tilts further than 15 degrees in either direction.

The reward scheme the algorithm worked to maximize was very simple, with a small reward given each time step the algorithm lasted without losing. It started out with just random moves to the left and right, which was not a very effective method of balancing the pole.

![Before training](/cartGifs/0.gif)

In the training, the specific information it used was the cart's position, its velocity, the pole angle, and the pole's velocity. After each try, it essentially evaluates how those factors affected its ability to continue growing the reward by staying up for longer.

## MountainCar

## Installing Dependencies
Run this good'ol pip install script:

`pip install -r requirements.txt`
