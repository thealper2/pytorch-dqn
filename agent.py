import logging
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import Action
from environment import GridWorld
from models import DQN
from replay_buffer import ReplayBuffer

logger = logging.getLogger("DQN_GridWorld")


class DQNAgent:
    """
    DQN Agent for GridWorld.
    """

    def __init__(
        self,
        env: GridWorld,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        target_update: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the DQN agent.

        Args:
            env: GridWorld environment.
            buffer_size: Size of the replay buffer.
            batch_size: Size of the training batch.
            gamma: Discount factor.
            epsilon_start: Initial epsilon value for ε-greedy policy.
            epsilon_end: Final epsilon value for ε-greedy policy.
            epsilon_decay: Decay rate for epsilon.
            learning_rate: Learning rate for the optimizer.
            target_update: Number of episodes between target network updates.
            device: Device to run the model on.
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = device

        # Observation shape: (channels, height, width)
        observation = env.reset()
        self.input_shape = observation.shape
        self.n_actions = len(Action)

        # Initialize networks
        self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Initialize step counter
        self.steps_done = 0

        # Track rewards
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state.

        Returns:
            Selected action.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def update_epsilon(self) -> None:
        """Update epsilon value for ε-greedy policy."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def optimize_model(self) -> float:
        """
        Update the policy network using a batch from the replay buffer.

        Returns:
            Loss value.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_q_values = reward_batch + (
            self.gamma * next_q_values * (1 - done_batch)
        )

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    def train(
        self, num_episodes: int, max_steps_per_episode: int = 100
    ) -> Tuple[List[float], List[int]]:
        """
        Train the agent.

        Args:
            num_episodes: Number of episodes to train for.
            max_steps_per_episode: Maximum number of steps per episode.

        Returns:
            Tuple of (episode_rewards, episode_lengths).
        """
        self.episode_rewards = []
        self.episode_lengths = []

        pbar = tqdm(range(num_episodes), desc="Training")
        for episode in pbar:
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0

            for step in range(max_steps_per_episode):
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(Action(action))

                # Store the transition in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update state
                state = next_state

                # Update statistics
                episode_reward += reward

                # Perform one step of optimization
                loss = self.optimize_model()
                episode_loss += loss

                if done:
                    break

            # Update epsilon
            self.update_epsilon()

            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)

            # Update progress bar
            pbar.set_postfix(
                {
                    "episode": episode,
                    "reward": f"{episode_reward:.2f}",
                    "length": step + 1,
                    "epsilon": f"{self.epsilon:.2f}",
                    "avg_loss": f"{episode_loss / (step + 1):.5f}"
                    if step > 0
                    else "0.0",
                }
            )

            # Log statistics
            logger.info(
                f"Episode {episode}: "
                f"reward={episode_reward:.2f}, "
                f"length={step + 1}, "
                f"epsilon={self.epsilon:.2f}, "
                f"avg_loss={episode_loss / (step + 1):.5f}"
                if step > 0
                else "0.0"
            )

        return self.episode_rewards, self.episode_lengths

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of episodes to evaluate for.

        Returns:
            Tuple of (average_reward, average_length).
        """
        total_reward = 0
        total_length = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.env.config.max_steps):
                # Select action with greedy policy
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()

                next_state, reward, done, _ = self.env.step(Action(action))
                state = next_state
                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward
            total_length += step + 1

            logger.info(
                f"Evaluation Episode {episode}: reward={episode_reward:.2f}, length={step + 1}"
            )

        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes

        logger.info(
            f"Evaluation: avg_reward={avg_reward:.2f}, avg_length={avg_length:.2f}"
        )

        return avg_reward, avg_length

    def save_model(self, path: str) -> None:
        """
        Save the model.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "input_shape": self.input_shape,
                "n_actions": self.n_actions,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load the model.

        Args:
            path: Path to load the model from.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.input_shape = checkpoint["input_shape"]
            self.n_actions = checkpoint["n_actions"]

            self.policy_net = DQN(self.input_shape, self.n_actions).to(self.device)
            self.target_net = DQN(self.input_shape, self.n_actions).to(self.device)

            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def plot_training_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot training results.

        Args:
            save_path: Path to save the plot. If None, the plot is shown.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Length")

        # Plot moving average of rewards
        window_size = min(100, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ax3.plot(moving_avg)
            ax3.set_title(f"Moving Average of Rewards (Window Size: {window_size})")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Average Reward")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training results plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
