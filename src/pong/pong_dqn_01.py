import collections
import time
from typing import NamedTuple, Any, Tuple, Optional, List

import click
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from .lib import wrappers
from .lib import dqn_model

ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5


GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Action = Any
State = Any


class Experience(NamedTuple):
    state: State
    action: Action
    reward: int
    done: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        """Reset the environment."""
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(
        self, net: torch.nn.Module, epsilon: float = 0.0, device: torch.device = "cpu"
    ) -> Optional[float]:
        """Play one step.

        :return: None if not finished, else the reward

        """
        assert device.type in ("cpu", "gpu")
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # convert state in numpy array and in tensor
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            # get q values from neural net
            q_vals_v = net(state_v)
            # choose the action with max q
            _, act_v = torch.max(q_vals_v, dim=1)
            # convert to integer
            action = int(act_v.item())

        # do environment step and add reward
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # save experience in experience buffer
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(
    batch: Tuple[np.ndarray, ...],
    net: nn.Module,
    tgt_net: nn.Module,
    device: torch.device = "cpu",
) -> torch.Tensor:
    # extract batch parts
    states, actions, rewards, dones, next_states = batch

    # covert all batch parts to tensors
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # noinspection PyArgumentList
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


@click.command()
@click.option("--cuda", is_flag=True, help="Enable cuda support")
@click.option(
    "--env",
    "env_name",
    default=ENV_NAME,
    type=str,
    help="Set a gym environment, default: " + ENV_NAME,
)
@click.option(
    "--reward",
    "reward_stop",
    default=MEAN_REWARD_BOUND,
    type=float,
    help="Mean reward boundary to stop training, default: {:0.2f}".format(
        MEAN_REWARD_BOUND
    ),
)
def main(cuda: bool, env_name: str, reward_stop: float):
    device = torch.device("cuda" if cuda else "cpu")
    # create environment
    env: gym.Env = wrappers.make_env(env_name)

    # create both neural networks
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # create summary writer for tensorboard
    writer = SummaryWriter(comment="-" + env_name)

    # create buffer and agent and init epsilon
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards: List[float] = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward: Optional[float] = None

    while True:
        frame_idx += 1
        # update epsilon
        epsilon = max(
            EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME
        )
        # play one step
        reward = agent.play_step(net, epsilon, device)

        if reward is not None:
            # add reward to total and calculate mean
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])

            # meter speed
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            # print and write information
            print(
                "%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s"
                % (frame_idx, len(total_rewards), float(mean_reward), epsilon, speed)
            )
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), env_name + "-bast.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, float(mean_reward)))
                best_mean_reward = float(mean_reward)
            if mean_reward > reward_stop:
                print("Solved in {0} frames!".format(frame_idx))
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        # sync target net with training net
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    writer.close()


if __name__ == "__main__":
    main()
