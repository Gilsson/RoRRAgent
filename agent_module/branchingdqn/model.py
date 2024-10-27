import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class DuelingNetwork(nn.Module):

    def __init__(self, obs, ac):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(obs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x):

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1, 1)
        return q_val


class BranchingQNetwork(nn.Module):
    def __init__(self, obs_size, action_bins):
        super(BranchingQNetwork, self).__init__()
        self.obs_size = obs_size

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # Separate Q-value heads for discrete actions
        self.q_heads = nn.ModuleList([nn.Linear(128, bins) for bins in action_bins])

        # Separate heads for predicting durations (one per action branch)
        self.duration_heads = nn.ModuleList(
            [nn.Linear(128, 1) for _ in range(len(action_bins))]
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        # Get Q-values for each discrete action branch
        q_values = [head(features) for head in self.q_heads]

        # Get predicted durations (continuous) for each action branch
        durations = [
            torch.sigmoid(head(features)) for head in self.duration_heads
        ]  # Scaled to [0, 5] seconds

        return q_values, durations


# b = BranchingQNetwork(5, 4, 6)

# b(torch.rand(10, 5))
