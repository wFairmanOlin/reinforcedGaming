import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import gym_mountain
import ffmpy

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
    rec = VideoRecorder(env, base_path=('pics/mountain' + fileName),
                        enabled=True)
    for episode in range(episodes):

        # Reset environment and record the starting state
        state = env.reset()
        for time in range(1000):
            # env.render()
            rec.capture_frame()
            action = gym_mountain.predict(policy, state)

            state, reward, done, _ = env.step(action.item())

            if(state[0] > .5):
                print('won in: ', time)
                break
        if (state[0] < .5):
            print('lost episode: ', episode)
    rec.close()


run(episodes=1)
ff = ffmpy.FFmpeg(
    inputs={("pics/mountain" + fileName + '.mp4'): None},
    outputs={("pics/mountain" + fileName + ".gif"): None})
ff.run()
