from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gymnasium as gym
import random

from branchingdqn.model import DuelingNetwork, BranchingQNetwork
from branchingdqn.utils import (
    TensorEnv,
    ExperienceReplayMemory,
    AgentConfig,
    BranchingTensorEnv,
)
import branchingdqn.utils


class BranchingDQN(nn.Module):

    def __init__(self, obs, ac, config):
        super().__init__()

        self.q = BranchingQNetwork(obs, ac)
        self.target = BranchingQNetwork(obs, ac)

        self.target.load_state_dict(self.q.state_dict())
        self.target_net_update_freq = config["target_net_update_freq"]
        self.update_counter = 0

    def get_action(self, state):
        with torch.no_grad():
            q_values, durations = self.q(state)  # Get Q-values and durations

            # Select discrete actions (highest Q-value per branch)
            actions = [torch.argmax(q, dim=-1).item() for q in q_values]

            # Convert durations from tensor to numpy
            durations = [d.item() for d in durations]

        # Combine actions and durations into a tuple
        return np.array(actions), np.array(durations)

    def update_policy(self, adam, memory, params):
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(
            params.batch_size
        )

        states = torch.tensor(b_states).float()
        actions = (
            torch.tensor(b_actions[0]).long().reshape(states.shape[0], -1, 1)
        )  # Discrete actions
        durations = (
            torch.tensor(b_actions[1]).float().reshape(states.shape[0], -1, 1)
        )  # Continuous durations
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        # Get Q-values and durations from the Q network for current states
        qvals, predicted_durations = self.q(states)

        current_q_values = self.q(states)[0].gather(2, actions).squeeze(-1)

        with torch.no_grad():
            # Get max Q-values for next states and their corresponding actions
            next_qvals, next_durations = self.q(next_states)
            argmax = torch.argmax(next_qvals, dim=2)

            max_next_q_vals = (
                self.target(next_states)[0].gather(2, argmax.unsqueeze(2)).squeeze(-1)
            )
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True)

        expected_q_vals = rewards + max_next_q_vals * 0.99 * masks
        loss_q = F.mse_loss(expected_q_vals, current_q_values)

        # For durations, you can compute a separate loss (e.g., MSE loss)
        predicted_durations = torch.cat(
            predicted_durations, dim=-1
        )  # Combine duration predictions
        loss_durations = F.mse_loss(predicted_durations, durations)

        # Total loss is the combination of Q-value loss and duration loss
        loss = loss_q + loss_durations

        adam.zero_grad()
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        for p in self.q.parameters():
            p.grad.data.clamp_(-1.0, 1.0)
        adam.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())


# args = utils.arguments()

# bins = 6
# env = BranchingTensorEnv(args.env, bins)

# config = AgentConfig()
# memory = ExperienceReplayMemory(config.memory_size)
# agent = BranchingDQN(
#     env.observation_space.shape[0], env.action_space.shape[0], bins, config
# )
# adam = optim.Adam(agent.q.parameters(), lr=config.lr)


# s = env.reset()
# ep_reward = 0.0
# recap = []

# p_bar = tqdm(total=config.max_frames)
# for frame in range(config.max_frames):

#     epsilon = config.epsilon_by_frame(frame)

#     if np.random.random() > epsilon:
#         action = agent.get_action(s)
#     else:
#         action = np.random.randint(0, bins, size=env.action_space.shape[0])

#     ns, r, done, infos = env.step(action)
#     ep_reward += r

#     if done:
#         ns = env.reset()
#         recap.append(ep_reward)
#         p_bar.set_description("Rew: {:.3f}".format(ep_reward))
#         ep_reward = 0.0

#     memory.push(
#         (
#             s.reshape(-1).numpy().tolist(),
#             action,
#             r,
#             ns.reshape(-1).numpy().tolist(),
#             0.0 if done else 1.0,
#         )
#     )
#     s = ns

#     p_bar.update(1)

#     if frame > config.learning_starts:
#         agent.update_policy(adam, memory, config)

#     if frame % 1000 == 0:
#         utils.save(agent, recap, args)


# p_bar.close()
