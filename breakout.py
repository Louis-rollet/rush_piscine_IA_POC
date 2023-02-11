from time import sleep
import gym


env_name = 'ALE/Breakout-v5'
env = gym.make(env_name, render_mode="human")

for _ in range(5):
    env.reset()
    termination = False
    while termination is not True:
        _, _, termination, _ = env.step(env.action_space.sample())
env.close()
