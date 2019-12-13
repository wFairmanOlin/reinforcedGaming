
# reinforcedGaming
For our final project we wanted to focus on how reinforcement learning can be used to train game-solving neural nets. Throughout the course of project we utilized OpenAI's [Gym library](https://gym.openai.com/) to create game environments that were easily compatible with the algorithms we were implementing.

Much of our initial code base and understanding of reinforcement learning came from Tim's article [*Policy Gradient Reinforcement Learning in PyTorch*](https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf). It is important to note, however, that this article uses code that is now depreciated due to newer versions of **PyTorch**.
## Running Our Code
To interact with the code made for this assignment, make sure you have python 3 and run the following script to install all necessary dependencies.

```
pip install -r requirements.txt
```
## Overview
Reinforcement learning (RL) is a branch of machine learning that aims to train a model by taking an action and observing how the environment reacts to the action taken. A very common depiction of this feedback loop is shown below.

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
The goal of the loss function is to accurately portray how far off our Q function is from acting ideal. For our project, we defined an ideal Q-function as one that would return the maximum reward immediately. While this is an unachievable goal, the model would train to get as close as possible to this goal.

The loss function used in our project is simply defined as the negative of the dot product between our reward value and max(q function).
```
loss = -(rewardVector . max(Q)Vector)
```

## Process
We started by finding an environment we could use to try some reinforcement learning techniques without having to build an entire game from scratch. We settled on the Gym library as described above, because its many games have all the utilities we were looking for. Each game can be imported into a python script, which means we could easily create something to control the "character" in the game. Additionally, the games also have information about the state of the environment (like where the character is or which direction it is moving) which is crucial for the reinforcement algorithm to have something to learn from.

### CartPole
The first game we decided to try out is known as CartPole. This game involves a cart with a pole attached to the top, and the goal is to keep the pole balanced and upright as long as possible. The cart can either move left or right, and you lose if the pole tilts further than 15 degrees in either direction or if the cart moves off screen.

The reward scheme of the game was very simple for our algorithm to take in, with a small reward given each time step the model lasts without losing. Because our initial agent was assigned with random weights it started out with just random moves to the left and right, which was not a very effective method of balancing the pole.

#### Before Training
![Before training](/pics/cart0.gif)

To train the model, we had access to the cart's position, velocity, the pole angle, and the pole's velocity. After each try, our algorithm essentially evaluates how those factors affect its ability to continue growing the reward by staying up for longer.

The neural net is comprised of 4 input nodes (one for each of the environment descriptors), a hidden layer of 128 nodes, and two output nodes. The two output nodes are the probabilities for each action to lead to a higher total reward. At each step of the game, the algorithm choses an action based on the probabilities the neural net gives out given the environment variables at that time.
```

```

After each attempt, the neural net is updated based on the newest information it is given, and it adjusts the weights of the nodes in order to maximize its reward.

After playing the game around 500 times, the algorithm was able to consistently stay standing up to the maximum time we gave it to run, averaging about 460 time steps out of a maximum 500 possible.

#### After Training
![After training](/pics/cartwinning.gif)

#### Effects of Learning Rate
With a small learning rate of `.005` we were able to achieve our winning state in relatively small amount of time. This can be seen in the first image below. When increasing the learning rate to `.02` you can see strong difference in the episode length over time graph. With a large learning rate, the neural net is more likely to skip over the optimal gradients and radically swing between high and low episode lengths.

##### Small Learning Rate
![Small Learning Rate](/pics/cartPole_low.png)

##### Large Learning Rate
![Large Learning Rate](/pics/cartPole_high.png)
### MountainCar

Next, we tackled MountainCar, a game where a cart tries to drive up a hill. However, it does not have the power to drive straight up and it has to be more clever in order to get all the way to the top.

In this game, the cart can either apply a force to the left, to the right, or none at all. The algorithm has information on the position and the velocity of the cart throughout each attempt, and the game ends once it hits its goal or it hits the maximum amount of time allowed.


This neural net was larger, going from the 2 input nodes representing position and velocity to a hidden layer of 64 nodes, to another hidden layer of 128, to 3 output nodes for each possible action.

The way the reward is set up is quite different for this game compared to CartPole. Initially, we opted for a system that gave a small negative reward for each time step where it had not hit the goal yet, and gave a comparatively large reward once it finally did hit the goal. However, with this setup we realized that it was difficult for the algorithm to learn, because unless it happened to randomly reach the goal every run that timed out would look the same. In every one of those failed attempts, there would be the same loss accrued from the time it ran and nothing meaningful could be changed. An alternate strategy we tried was to make the reward more explicitly encourage getting closer and closer to the reward, so giving a small reward for each "new record" in that attempt.

We found that the second reward scheme showed effective results after several thousand attempts, but we felt like we were giving the algorithm too much of a crutch by giving it more explicit instructions, and not allowing it to explore its own potential solutions. Because of this, we reverted to the original reward structure again, and simply let it run for longer each time to make it more likely that it would reach the reward. Once it did hit the reward, it was actually more effective, probably because it was able to explore a more diverse range of options that would get it to its goal.


![Before training](/pics/mountain0.gif)

**Before training**

![After 250 tries](/pics/mountain250.gif)

**After 250 tries**

![After 1800 tries](/pics/mountainwinning.gif)

**After 1800 tries**
