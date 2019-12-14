from __future__ import annotations
from typing import NamedTuple, Tuple, List, Iterable

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    """The neural network.

    We only use one hidden layer with 128 units.

    """

    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Episode(NamedTuple):
    """A full episode with the final reward and a list of steps."""

    reward: float
    steps: List[EpisodeStep]


class EpisodeStep(NamedTuple):
    """A step with observation and the action."""

    observation: Tuple[float]
    action: int


# noinspection PyArgumentList
def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> Iterable[List[Episode]]:
    """Generate batches by stepping through the environment.

    Each batch contains a number of full episodes.

    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    softmax = nn.Softmax(dim=1)

    while True:
        # turn observation in tensor
        obs_v = torch.FloatTensor([obs])
        # calculate probabilities of next actions according to neural net
        act_probs_v = softmax(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        # choose next action by probability
        action = np.random.choice(len(act_probs), p=act_probs)
        # perform next step
        next_obs, reward, is_done, _ = env.step(action)
        # add reward and save step (observation and action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        # if episode is finished
        if is_done:
            # append episode to batch
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            # set reward and steps to zero and reset environment
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            # yield the batch if batch_size is reached
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


# noinspection PyArgumentList
def filter_batch(
    batch: List[Episode], percentile: float
) -> Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    """Filter the batch for episodes with a reward in the upper percentile."""
    # get all rewards from current batch
    rewards = [episode.reward for episode in batch]
    # calculate bound and mean
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs: List[Tuple[float]] = []
    train_act: List[int] = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend([step.observation for step in example.steps])
        train_act.extend([step.action for step in example.steps])

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    # create environment and store observations size and number of actions
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # create neural net, learning objective and optimizer for learning
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # generate batches and iterate over them
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # filter the batches for learning (only upper percentile)
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # reset gradients
        optimizer.zero_grad()
        # get action scores from neural net
        action_scores_v = net(obs_v)
        # calculate loss to actual actions
        loss_v = objective(action_scores_v, acts_v)
        # back propagate
        loss_v.backward()
        # optimize net parameters
        optimizer.step()
        print(
            "%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f"
            % (iter_no, loss_v.item(), reward_m, reward_b)
        )
        if reward_m > 199:
            print("Solved!")
            break
