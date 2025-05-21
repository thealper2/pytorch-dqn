import logging
import os
import random

import numpy as np
import torch
import typer

from agent import DQNAgent
from config import GridWorldConfig
from environment import GridWorld
from visualization import create_visualization, save_trajectory_visualization

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dqn_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("DQN_GridWorld")

app = typer.Typer(help="PyTorch - Deep Q-Learning (DQN)")


@app.command()
def train(
    width: int = typer.Option(5, help="Width of the grid"),
    height: int = typer.Option(5, help="Height of the grid"),
    obstacle_density: float = typer.Option(
        0.2, help="Density of obstacles in the grid"
    ),
    episodes: int = typer.Option(1000, help="Number of episodes to train for"),
    buffer_size: int = typer.Option(10000, help="Size of the replay buffer"),
    batch_size: int = typer.Option(64, help="Size of the training batch"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    epsilon_start: float = typer.Option(1.0, help="Initial epsilon value"),
    epsilon_end: float = typer.Option(0.05, help="Final epsilon value"),
    epsilon_decay: float = typer.Option(0.995, help="Decay rate for epsilon"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    target_update: int = typer.Option(
        10, help="Number of episodes between target network updates"
    ),
    model_path: str = typer.Option(
        "models/dqn_model.pt", help="Path to save the model"
    ),
    results_dir: str = typer.Option("results", help="Directory to save results"),
    evaluate: bool = typer.Option(True, help="Evaluate the model after training"),
    eval_episodes: int = typer.Option(10, help="Number of episodes to evaluate for"),
    seed: int = typer.Option(None, help="Random seed"),
) -> None:
    """Train a DQN agent on the GridWorld environment."""

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create environment
    config = GridWorldConfig(
        width=width,
        height=height,
        obstacle_positions=None,  # Will be generated randomly
        max_steps=width * height,  # Maximum steps is the size of the grid
    )
    env = GridWorld(config)

    # Print environment information
    logger.info(f"Environment created with dimensions {width}x{height}")
    logger.info(f"Start position: {config.start_position}")
    logger.info(f"Goal position: {config.goal_position}")
    logger.info(f"Number of obstacles: {len(env.config.obstacle_positions)}")

    # Create agent
    agent = DQNAgent(
        env=env,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        learning_rate=learning_rate,
        target_update=target_update,
    )

    # Train agent
    logger.info(f"Starting training for {episodes} episodes")
    agent.train(episodes)

    # Save model
    agent.save_model(model_path)

    # Plot training results
    results_path = os.path.join(results_dir, "training_results.png")
    agent.plot_training_results(results_path)

    # Create policy visualization
    policy_path = os.path.join(results_dir, "policy_visualization.png")
    create_visualization(env, agent, policy_path)

    # Create trajectory visualization
    trajectory_path = os.path.join(results_dir, "trajectory_visualization.png")
    save_trajectory_visualization(env, agent, trajectory_path)

    # Evaluate agent
    if evaluate:
        logger.info(f"Evaluating agent for {eval_episodes} episodes")
        avg_reward, avg_length = agent.evaluate(eval_episodes)
        logger.info(
            f"Evaluation results: avg_reward={avg_reward:.2f}, avg_length={avg_length:.2f}"
        )

    logger.info("Training completed")


@app.command()
def evaluate(
    model_path: str = typer.Option(
        "models/dqn_model.pt", help="Path to load the model from"
    ),
    width: int = typer.Option(5, help="Width of the grid"),
    height: int = typer.Option(5, help="Height of the grid"),
    obstacle_density: float = typer.Option(
        0.2, help="Density of obstacles in the grid"
    ),
    episodes: int = typer.Option(10, help="Number of episodes to evaluate for"),
    results_dir: str = typer.Option("results", help="Directory to save results"),
    seed: int = typer.Option(None, help="Random seed"),
) -> None:
    """Evaluate a trained DQN agent on the GridWorld environment."""

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)

    # Create environment
    config = GridWorldConfig(
        width=width,
        height=height,
        obstacle_positions=None,  # Will be generated randomly
        max_steps=width * height,  # Maximum steps is the size of the grid
    )
    env = GridWorld(config)

    # Print environment information
    logger.info(f"Environment created with dimensions {width}x{height}")
    logger.info(f"Start position: {config.start_position}")
    logger.info(f"Goal position: {config.goal_position}")
    logger.info(f"Number of obstacles: {len(env.config.obstacle_positions)}")

    try:
        # Create agent
        agent = DQNAgent(env=env)

        # Load model
        agent.load_model(model_path)

        # Evaluate agent
        logger.info(f"Evaluating agent for {episodes} episodes")
        avg_reward, avg_length = agent.evaluate(episodes)
        logger.info(
            f"Evaluation results: avg_reward={avg_reward:.2f}, avg_length={avg_length:.2f}"
        )

        # Create policy visualization
        policy_path = os.path.join(results_dir, "policy_visualization.png")
        create_visualization(env, agent, policy_path)

        # Create trajectory visualization
        trajectory_path = os.path.join(results_dir, "trajectory_visualization.png")
        save_trajectory_visualization(env, agent, trajectory_path)

        logger.info("Evaluation completed")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    app()
