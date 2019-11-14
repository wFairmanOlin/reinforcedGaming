import gym
import time
from gym import spaces

# Observation: 
#     Type: Box(4)
#     Num Observation                 Min         Max
#     0   Cart Position             -4.8            4.8
#     1   Cart Velocity             -Inf            Inf
#     2   Pole Angle                 -24 deg        24 deg
#     3   Pole Velocity At Tip      -Inf            Inf
    
# Actions:
#     Type: Discrete(2)
#     Num Action
#     0   Push cart to the left
#     1   Push cart to the right


#define our action space with values 0 or 1
# move = spaces.Discrete(2)
# x = move.sample() #randomly assign x a valid integer


env = gym.make('CartPole-v0')

for i_episode in range(3):
    observation = env.reset()
    for t in range(1000):
        env.render()
        
        #basic if swinging right push cart right 
        if observation[3] > 0:
            action = 1
        else:
            action = 0
            
        # action = env.action_space.sample() #randomly assigning an action
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        # time.sleep(.05)

env.close()