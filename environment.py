import random
from typing import Any, Dict, List, Tuple

import numpy as np

from config import Action, GridWorldConfig, GridWorldState


class GridWorld:
    """
    GridWorld environemnt where an agent needs to navigate from a start position to a goal.

    The agent can move four directions: up, right, down, left.
    There are obstacles that the agent needss to avoid.
    """

    def __init__(self, config: GridWorldConfig):
        """
        Initialize the GridWorld environment.

        Args:
            config: Configuration for the GridWorld environment.
        """
        self.config = config
        if config.obstacle_positions is None:
            self.config.obstacle_positions = self._generate_random_obstacles(
                int(config.width * config.height * 0.2)
            )

        self.state = GridWorldState(
            x=config.start_position[0], y=config.start_position[1], steps=0
        )
        self.done = False

    def _generate_random_obstacles(self, num_obstacles: int) -> List[Tuple[int, int]]:
        """
        Generate random obstacle positions.

        Args:
            num_obstacles: Number of obstacles to generate.

        Returns:
            List of obstacle positions.
        """
        obstacles = []
        all_positions = [
            (x, y) for x in range(self.config.width) for y in range(self.config.height)
        ]

        # Remove start and goal positions
        all_positions.remove(self.config.start_position)
        all_positions.remove(self.config.goal_position)

        obstacles = random.sample(all_positions, min(num_obstacles, len(all_positions)))
        return obstacles

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the inital state.

        Returns:
            Observation of the initial state.
        """
        self.state = GridWorldState(
            x=self.config.start_position[0], y=self.config.start_position[1], steps=0
        )
        self.done = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation of the current state.

        Returns:
            A 4-channel binary grid representing the environment:
                Channel 0: Agent position
                Channel 1: Goal position
                Channel 2: Obstacles
                Channel 3: Visited positions
        """
        observation = np.zeros(
            (4, self.config.height, self.config.width), dtype=np.float32
        )

        # Agent position
        observation[0, self.state.y, self.state.x] = 1.0

        # Goal position
        observation[1, self.config.goal_position[1], self.config.goal_position[0]] = 1.0

        # Obstacles
        for x, y in self.config.obstacle_positions:
            if 0 <= x < self.config.width and 0 <= y < self.config.height:
                observation[2, y, x] = 1.0

        return observation

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The action to take.

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0.0, True, {"steps": self.state.steps}

        # Get next position based on action
        next_x, next_y = self.state.x, self.state.y

        if action == Action.UP:
            next_y = max(0, self.state.y - 1)
        elif action == Action.RIGHT:
            next_x = min(self.config.width - 1, self.state.x + 1)
        elif action == Action.DOWN:
            next_y = min(self.config.height - 1, self.state.y + 1)
        elif action == Action.LEFT:
            next_x = max(0, self.state.x - 1)

        # Check if next position is an obstacle
        if (next_x, next_y) in self.config.obstacle_positions:
            reward = self.config.obstacle_penalty
            # Stay in the same position
            next_x, next_y = self.state.x, self.state.y
        else:
            reward = self.config.step_penalty

        # Update state
        self.state = GridWorldState(x=next_x, y=next_y, steps=self.state.steps + 1)

        # Check if goal is reached
        if (self.state.x, self.state.y) == self.config.goal_position:
            reward = self.config.goal_reward
            self.done = True

        # Check if max steps reached
        if self.state.steps >= self.config.max_steps:
            self.done = True

        return self._get_observation(), reward, self.done, {"steps": self.state.steps}
