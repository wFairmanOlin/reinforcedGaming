import gym
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time as t
import gym_medium
import ffmpy
env = gym.make('CartPole-v1')
#MountainCar-v0
#CartPole-v1
# env.seed(1)
# torch.manual_seed(1)

# Hyperparameter

fileName = str(input('episode (increments of 500): '))

print('correct file')

policy = gym_medium.Policy()
policy.load_state_dict(torch.load('cartNets/' + fileName + '.pth'))
policy.eval()


def run(episodes):
    rec = VideoRecorder(env, base_path=('cartVideo/' + fileName),
                        enabled=True)
    for episode in range(episodes):
        # enable = False

        # Reset environment and record the starting state
        state = env.reset()
        for time in range(1000):
            env.render()
            rec.capture_frame()
            action = gym_medium.predict(policy, state)

            state, reward, done, _ = env.step(action.item())

            if done:
                print('lost in', time)
                break
            # if(state[0] > .5):
            #     print('won in: ', time)
            #     break
        # if (state[0] < .5):
        #     print('lost episode: ', episode)
        # if episode == 0:
        #
    rec.close()

run(episodes=10)
ff = ffmpy.FFmpeg(
    inputs={("cartVideo/" + fileName + '.mp4'): None},
    outputs={("cartGifs/" + fileName + ".gif"): None})
ff.run()
