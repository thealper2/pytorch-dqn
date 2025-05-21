import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DQNAgent
from config import Action
from environment import GridWorld

logger = logging.getLogger("DQN_GridWorld")


def create_visualization(env: GridWorld, agent: DQNAgent, save_path: str) -> None:
    """
    Create a visualization of the agent's policy.

    Args:
        env: GridWorld environment.
        agent: DQN agent.
        save_path: Path to save the visualization.
    """
    # Create a grid to visualize the Q-values
    q_values = np.zeros((env.config.height, env.config.width, len(Action)))

    # For each cell in the grid
    for y in range(env.config.height):
        for x in range(env.config.width):
            # Skip obstacles
            if (x, y) in env.config.obstacle_positions:
                continue

            # Create a state for this position
            state = np.zeros((4, env.config.height, env.config.width), dtype=np.float32)
            state[0, y, x] = 1.0  # Agent position
            state[1, env.config.goal_position[1], env.config.goal_position[0]] = (
                1.0  # Goal position
            )

            # Mark obstacles
            for ox, oy in env.config.obstacle_positions:
                if 0 <= ox < env.config.width and 0 <= oy < env.config.height:
                    state[2, oy, ox] = 1.0

            # Get Q-values for this state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values[y, x] = agent.policy_net(state_tensor).cpu().numpy()

    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create a grid
    ax.set_xticks(np.arange(-0.5, env.config.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.config.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits
    ax.set_xlim(-0.5, env.config.width - 0.5)
    ax.set_ylim(env.config.height - 0.5, -0.5)  # Invert y-axis

    # Plot obstacles
    for x, y in env.config.obstacle_positions:
        if 0 <= x < env.config.width and 0 <= y < env.config.height:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))

    # Plot start and goal
    start_x, start_y = env.config.start_position
    goal_x, goal_y = env.config.goal_position
    ax.add_patch(
        plt.Rectangle((start_x - 0.5, start_y - 0.5), 1, 1, color="blue", alpha=0.3)
    )
    ax.add_patch(
        plt.Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1, color="green", alpha=0.3)
    )

    # Create a meshgrid for quiver plot
    X, Y = np.meshgrid(np.arange(env.config.width), np.arange(env.config.height))

    # Initialize arrays for quiver plot
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    C = np.zeros_like(X, dtype=float)  # For coloring arrows

    # Set arrow directions and colors based on Q-values
    for y in range(env.config.height):
        for x in range(env.config.width):
            # Skip obstacles
            if (x, y) in env.config.obstacle_positions:
                continue

            # Set arrow direction based on max Q-value action
            best_action = np.argmax(q_values[y, x])

            if best_action == Action.UP:
                U[y, x] = 0
                V[y, x] = 1
            elif best_action == Action.LEFT:
                U[y, x] = -1
                V[y, x] = 0
            elif best_action == Action.RIGHT:
                U[y, x] = 1
                V[y, x] = 0
            elif best_action == Action.DOWN:
                U[y, x] = 0

            # Set color based on max Q-value
            C[y, x] = np.max(q_values[y, x])

    # Normalize colors
    if np.max(C) > np.min(C):
        C = (C - np.min(C)) / (np.max(C) - np.min(C))

    # Plot arrows
    quiver = ax.quiver(
        X,
        Y,
        U,
        V,
        C,
        scale=20,
        scale_units="width",
        headwidth=5,
        headlength=5,
        width=0.005,
        cmap="viridis",
    )

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label("Max Q-value")

    # Add title
    ax.set_title("Policy Visualization")

    # Save figure
    plt.savefig(save_path)
    logger.info(f"Policy visualization saved to {save_path}")
    plt.close()


def save_trajectory_visualization(
    env: GridWorld, agent: DQNAgent, save_path: str
) -> None:
    """
    Save a visualization of the agent's trajectory.

    Args:
        env: GridWorld environment.
        agent: DQN agent.
        save_path: Path to save the visualization.
    """
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create a grid
    ax.set_xticks(np.arange(-0.5, env.config.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.config.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits
    ax.set_xlim(-0.5, env.config.width - 0.5)
    ax.set_ylim(env.config.height - 0.5, -0.5)  # Invert y-axis

    # Plot obstacles
    for x, y in env.config.obstacle_positions:
        if 0 <= x < env.config.width and 0 <= y < env.config.height:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))

    # Plot start and goal
    start_x, start_y = env.config.start_position
    goal_x, goal_y = env.config.goal_position
    ax.add_patch(
        plt.Rectangle((start_x - 0.5, start_y - 0.5), 1, 1, color="blue", alpha=0.3)
    )
    ax.add_patch(
        plt.Rectangle((goal_x - 0.5, goal_y - 0.5), 1, 1, color="green", alpha=0.3)
    )

    # Run an episode and track the agent's trajectory
    state = env.reset()
    trajectory = [(env.state.x, env.state.y)]

    while True:
        # Select action with greedy policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor)
            action = q_values.max(1)[1].item()

        next_state, _, done, _ = env.step(Action(action))
        state = next_state

        trajectory.append((env.state.x, env.state.y))

        if done:
            break

    # Plot trajectory
    x_coords, y_coords = zip(*trajectory)
    ax.plot(x_coords, y_coords, "r-", linewidth=2)
    ax.scatter(x_coords, y_coords, color="red", s=50)

    # Add title
    ax.set_title("Agent Trajectory Visualization")

    # Save figure
    plt.savefig(save_path)
    logger.info(f"Trajectory visualization saved to {save_path}")
    plt.close()
