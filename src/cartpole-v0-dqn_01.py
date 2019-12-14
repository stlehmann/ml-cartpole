import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import ptan


GAMMA = 0.99
LEARNING_RATE = 0.01
N_HIDDEN = 128
BATCH_SIZE = 8

EPSILON_START = 1.0
EPSILON_STOP = 0.02
EPSILON_STEPS = 5000

REPLAY_BUFFER = 50000


class DQN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.FloatTensor):
        return self.net(x)


def calc_target(net, local_reward, next_state):
    if next_state is None:
        return local_reward

    state_v = torch.tensor([next_state], dtype=torch.float32)
    next_q_v = net(state_v)
    best_q = next_q_v.max(dim=1)[0].item()
    return local_reward + GAMMA * best_q


def main():
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = DQN(obs_size, n_actions)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_BUFFER)

    # noinspection PyUnresolvedReferences
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    while True:
        step_idx += 1
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        replay_buffer.populate(1)

        if len(replay_buffer) < BATCH_SIZE:
            continue

        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states = [exp.state for exp in batch]
        batch_actions = [exp.action for exp in batch]
        batch_targets = [calc_target(net, exp.reward, exp.last_state) for exp in batch]

        optimizer.zero_grad()
        # noinspection PyArgumentList
        states_v = torch.FloatTensor(batch_states)
        net_q_v = net(states_v)
        target_q = net_q_v.data.numpy().copy()
        target_q[range(BATCH_SIZE), batch_actions] = batch_targets
        target_q_v = torch.tensor(target_q)
        loss_v = mse_loss(net_q_v, target_q_v)
        loss_v.backward()
        optimizer.step()

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, selector.epsilon, done_episodes))
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

if __name__ == "__main__":
    from pyannotate_runtime import collect_types
    collect_types.init_types_collection()
    with collect_types.collect():
        main()
    collect_types.dump_stats('type_info.json')
