import os
import random
import re
import sys
import threading

from tqdm import tqdm

from branchingdqn.branching_dqn import BranchingDQN
from branchingdqn.model import BranchingQNetwork
from branchingdqn.utils import BranchingTensorEnv, ExperienceReplayMemory


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        return random.choice(self.action_space)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov5"))


class DQN(nn.Module):
    def __init__(self, state_shape, action_size, duration_size):
        # action_size + duration_size should be equal to the number of Q-values
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_shape, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size + duration_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)

        # Split Q-values into action Q-values and duration Q-values
        action_q_values = q_values[:, :-1]
        duration_q_values = torch.nn.functional.softplus(q_values[:, -1:])

        return torch.cat((action_q_values, duration_q_values), dim=1)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append([experience])

    def sample(self, batch_size):
        sample = []
        for _ in range(batch_size):
            sample += self.buffer.pop()
        return sample

    def size(self):
        return len(self.buffer)


def encode_detections(detections, max_detections=10, detection_size=6):
    encoded_detections = []
    for detection in detections[:max_detections]:
        encoded_detections.extend(detection["bbox"])
        encoded_detections.append(detection["confidence"])
        encoded_detections.append(detection["class"])

    while len(encoded_detections) < max_detections * detection_size:
        encoded_detections.append(0.0)

    return encoded_detections[: max_detections * detection_size]


def encode_state(state):
    return np.concatenate(
        [
            state["cooldowns"],
            [state["health"]],
            [state["money"]],
            [state["time"]],
            [state["previous_health"]],
            encode_detections(state["detections"]),
        ]
    )


class DQNAgent:
    def __init__(
        self,
        state_shape,
        action_size,
        replay_buffer,
        batch_size=2,
        gamma=0.99,
        lr=0.001,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        # self.model = DQN(state_shape, action_size, 3)
        self.model = BranchingQNetwork(state_shape, action_size, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, epsilon):
        if random.random() < epsilon:

            return [
                (
                    random.choice(range(self.action_size)),
                    random.uniform(0.1, 2.0),
                    random.uniform(0.1, 2.0),
                )
            ]
        else:
            state_concat = encode_state(state)
            state_tensor = torch.FloatTensor(state_concat).unsqueeze(0)
            q_values = self.model(state_tensor)
            duration = q_values[0, -1].item()
            start_time = q_values[0, -2].item()

            amount_of_actions = int(q_values[0, -3].item())

            return [(torch.argmax(q_values[:, :-1]).item(), start_time, duration)]

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        print(f"batch {batch}")
        states, actions, rewards, next_states, dones = zip(*batch)

        print(states)

        states_concat = [encode_state(state) for state in states]
        next_states_concat = [encode_state(next_state) for next_state in next_states]

        states_tensor = torch.FloatTensor(np.array(states_concat))
        next_states_tensor = torch.FloatTensor(np.array(next_states_concat))
        actions_tensor = torch.LongTensor([action[0] for action in actions])
        start_time_tensor = torch.FloatTensor([action[1] for action in actions])
        durations_tensor = torch.FloatTensor([action[2] for action in actions])
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)

        current_q_values = self.model(states_tensor)
        current_action_q_values = current_q_values[:, :-1]  # Action Q-values
        current_duration_q_values = current_q_values[:, -1]  # Duration Q-values

        # Compute target values for actions
        next_q_values = self.model(next_states_tensor)
        max_next_q_values = next_q_values[:, :-1].max(1)[0]
        expected_action_q_values = rewards_tensor + (
            self.gamma * max_next_q_values * (1 - dones_tensor)
        )

        # Compute action loss
        print(f"current_action_q_values {current_action_q_values}")
        print(actions_tensor.unsqueeze(1))
        current_action_q_values_for_actions = current_action_q_values.gather(
            1, actions_tensor.unsqueeze(1)
        ).squeeze(1)
        action_loss = self.loss_fn(
            current_action_q_values_for_actions, expected_action_q_values
        )

        # Compute target values for durations (assuming you want to match durations to a fixed target, e.g., average duration)
        duration_target = torch.mean(durations_tensor)  # Example duration target
        duration_loss = self.loss_fn(
            current_duration_q_values,
            torch.full(durations_tensor.shape, duration_target),
        )

        # Total loss
        total_loss = action_loss + duration_loss

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def preprocess_state(state):
    state = state[0]
    cooldowns = state["cooldowns"]
    health = np.array([state["health"]])
    money = np.array([state["money"]])
    time = np.array([state["time"]])
    previous_health = np.array([state["previous_health"]])

    # Normalize and reshape the screen image
    screen = state["screen"].astype(np.float32) / 255.0  # Normalize pixel values
    screen = (
        torch.tensor(screen).permute(2, 0, 1).unsqueeze(0)
    )  # [H, W, C] -> [C, H, W] and add batch dimension

    # Flatten everything except the screen
    flattened_state = np.concatenate([cooldowns, health, money, time, previous_health])

    # Convert to tensors
    flattened_state_tensor = torch.tensor(
        flattened_state, dtype=torch.float32
    ).unsqueeze(
        0
    )  # Add batch dimension

    return flattened_state_tensor


def train_dqn(
    environment,
    agent,
    episodes,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
):
    # epsilon = epsilon_start
    # # agent.model.load_state_dict(torch.load("models/model.pth"))
    # for episode in range(episodes):
    #     state, _ = environment.reset()
    #     done = False
    #     while not done:
    #         action = agent.get_action(state, epsilon)
    #         print(action)
    #         next_state, reward, done, _ = environment.step(action)
    #         # print(next_state)
    #         agent.replay_buffer.add((state, action, reward, next_state, done))
    #         agent.train()
    #         torch.save(agent.model.state_dict(), "models/model.pth")
    #         state = next_state
    #         epsilon = max(epsilon_end, epsilon * epsilon_decay)
    #         if done:

    #             print(f"Episode {episode + 1} finished with epsilon {epsilon:.4f}.")
    #             epsilon = epsilon_start
    #             environment.reset()
    num_episodes = episodes
    adam = optim.Adam(agent.q.parameters())
    config = {
        "lr": 1e-4,  # Learning rate
        "batch_size": 64,  # Batch size for training
        "memory_size": 100000,  # Size of experience replay buffer
        "target_net_update_freq": 1000,  # Frequency to update the target network
        "max_frames": 1000000,  # Total number of frames to train on
        "epsilon_by_frame": lambda frame: max(0.01, 1 - frame / 50000),  # Epsilon decay
        "learning_starts": 1000,  # Number of frames before training starts
        "save_freq": 10000,  # Frequency to save the model
        "num_episodes": 1000,  # Number of episodes for training
    }
    # Replay memory buffer for experience replay
    memory = ExperienceReplayMemory(config["memory_size"])
    num_episodes = config["num_episodes"]

    # For progress bar tracking
    pbar = tqdm(total=config["max_frames"])

    # Loop over episodes
    for episode in range(num_episodes):
        state = environment.reset()
        state = preprocess_state(state)
        episode_reward = 0
        epsilon = 0
        done = False

        while not done:
            # Epsilon-greedy exploration strategy
            epsilon = max(
                epsilon_end, epsilon * epsilon_decay
            )  # Decrease epsilon over time

            # Select action
            if np.random.random() > epsilon:
                print(state)
                # state = state[0]
                discrete_actions, durations = agent.get_action(state)
            else:
                # Random exploration for both discrete actions and durations
                discrete_actions = np.random.randint(0, [5, 5, 3, 3])
                durations = np.random.uniform(0.0, 1.0, size=(4,))

            # Execute action in the environment
            action = (
                discrete_actions,
                durations,
            )  # Tuple containing both discrete and continuous actions
            next_state, reward, done, _ = environment.step(action)
            next_state = preprocess_state(next_state)
            # Store the experience in replay buffer
            memory.push(
                (
                    state,
                    (discrete_actions, durations),
                    reward,
                    next_state,
                    0.0 if done else 1.0,
                )
            )

            # Accumulate reward for the episode
            episode_reward += reward

            # Move to the next state
            state = next_state

            # Update the policy if enough frames have passed
            if pbar.n > config["learning_starts"]:
                agent.update_policy(adam, memory, config)

            # Update the progress bar
            pbar.update(1)

        # Logging
        pbar.set_description(f"Episode: {episode}, Reward: {episode_reward:.2f}")


# Assuming you have registered 'RiskOfRain-v0' environment
gym.register(
    id="RiskOfRain-v0",
    entry_point="env:RiskOfRainEnv",  # Adjust the entry point if defined in another file/module
)
# print(gym.spec("RiskOfRain-v0"))
# print(gym.envs.registry.keys())
env = gym.make("RiskOfRain")
state_shape = 8
print(state_shape)
# action_size = env.action_space.n
bins = [5, 5, 3, 3]
replay_buffer = ReplayBuffer(buffer_size=100)
# env = BranchingTensorEnv("RiskOfRain", 4)
config = {
    "lr": 1e-4,  # Learning rate
    "batch_size": 64,  # Batch size for training
    "memory_size": 100000,  # Size of experience replay buffer
    "target_net_update_freq": 1000,  # Frequency to update the target network
    "max_frames": 1000000,  # Total number of frames to train on
    "epsilon_by_frame": lambda frame: max(0.01, 1 - frame / 50000),  # Epsilon decay
    "learning_starts": 1000,  # Number of frames before training starts
    "save_freq": 10000,  # Frequency to save the model
    "num_episodes": 1000,  # Number of episodes for training
}
agent = BranchingDQN(state_shape, bins, config)
# agent = DQNAgent(state_shape, action_size, replay_buffer, batch_size=64)
env_thread = threading.Thread(
    target=train_dqn,
    daemon=True,
    kwargs={"environment": env, "agent": agent, "episodes": 1000},
)
env_thread.start()
env_thread.join()
# Start the Tkinter bounding box overlay on the main thread
# if env.bbox_overlay:
#     env.bbox_overlay.run()
# train_dqn(env, agent, episodes=1000, epsilon_start=0.9)
