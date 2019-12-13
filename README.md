# reinforcedGaming
For our final project we wanted to focus on how reinforcement learning can be used to train game-solving neural nets. Throughout the course of project we utilized OpenAI's [Gym library](https://gym.openai.com/) to create game environments that were easily compatible with the algorithms we were implementing.

## Overview
Reinforcement learning is a branch of machine learning that aims to train a model by taking action and observing how the environment reacts to any and all actions taken. A very common depiction of this feedback loop is shown below.

![overview](/pics/basicRL.png)

In general, a model's actions are determined by the model's agent. The agent makes decisions based on the previous state of the environment. After taking a new action as input, the environment enters a new state and outputs a reward. In most cases a reinforcement algorithm wants to maximize its reward by tuning the agent to take an action with the highest probability of generating a reward.

### Deep Q Learning
For this project, we specifically focused on deep Q-learning. Similar to general reinforcement learning, the goal of deep-Q learning is to optimize an agent (neural network) to take an action that has the highest probability of producing a reward. The The agent in deep Q-learning is defined as a neural network that takes in environmental variables and outputs actions. 
## Process
We started by finding an environment we could use to start trying out some reinforcement learning techniques without having to build an entire game from scratch. We settled on the Gym library as described above, because its many games have all the utilities we were looking for. Each game can be imported into a python script, which means we could easily create something to control the "character" in the game. Additionally, the games also have information about the state of the environment (like where the character is or which direction it is moving) which is crucial for the reinforcement algorithm to have something to learn from.

## CartPole
The first game we decided to try out is CartPole, This game involves a cart with a pole attached to the top, and the goal is to keep the pole balanced and upright as long as possible. The cart can either move left or right, and you lose if the pole tilts further than 15 degrees in either direction.

The reward scheme the algorithm worked to maximize was very simple, with a small reward given each time step the algorithm lasted without losing. It started out with just random moves to the left and right, which was not a very effective method of balancing the pole. In the training, the specific information it used was the cart's position, its velocity, the pole angle, and the pole's velocity. After each try, it essentially evaluates how those factors affected its ability to continue growing the reward by staying up for longer.

## MountainCar

## Installing Dependencies
Run this good'ol pip install script:

`pip install -r requirements.txt`
