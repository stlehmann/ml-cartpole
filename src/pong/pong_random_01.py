import gym
import click


env = gym.make("PongNoFrameskip-v4")
obs = env.reset()
done = False

while not done:
    env.render()
    obs, reward, done, _ = env.step(env.action_space.sample())

env.close
