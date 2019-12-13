import gym
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time as t
import gym_cartPole
import ffmpy
env = gym.make('CartPole-v1')

# Hyperparameter

fileName = str(input('episode (increments of 500): '))

print('correct file')

policy = gym_cartPole.Policy()
policy.load_state_dict(torch.load('cartNets/' + fileName + '.pth'))
policy.eval()


def run(episodes):
    rec = VideoRecorder(env, base_path=('pics/cart' + fileName),
                        enabled=True)
    for episode in range(episodes):

        # Reset environment and record the starting state
        state = env.reset()
        for time in range(1000):
            env.render()
            #Uncomment to record
            # rec.capture_frame()
            action = gym_cartPole.predict(policy, state)

            state, reward, done, _ = env.step(action.item())

            if done:
                print('lost in', time)
                break
    #Uncomment to record
    # rec.close()

run(episodes=5)

#Uncomment to record
# ff = ffmpy.FFmpeg(
#     inputs={("pics/cart" + fileName + '.mp4'): None},
#     outputs={("pics/cart" + fileName + ".gif"): None})
# ff.run()
