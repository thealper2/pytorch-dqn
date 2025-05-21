from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for GridWorld.

    Input: 4-channel grid observation
    Output: Q-values for each action
    """

    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Initialize the DQN model.

        Args:
            input_shape: Input shape of the observation (channels, height, width).
            n_actions: Number of possible actions.
        """
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the feature maps after convolutions
        conv_output_size = input_shape[1] * input_shape[2] * 64

        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Q-values for each action.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
