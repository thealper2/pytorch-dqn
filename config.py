from dataclasses import dataclass
from enum import IntEnum
from typing import List, NamedTuple, Tuple


# Define constants
class Action(IntEnum):
    """Possible actions in the GridWorld environment."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class GridWorldConfig:
    """Configuration for the GridWorld environment."""

    width: int = 5
    height: int = 5
    start_position: Tuple[int, int] = (0, 0)
    goal_position: Tuple[int, int] = (4, 4)
    obstacle_positions: List[Tuple[int, int]] = None
    max_steps: int = 100
    obstacle_penalty: float = -1.0
    goal_reward: float = 10.0
    step_penalty: float = -0.1


class GridWorldState(NamedTuple):
    """State representation for the GridWorld environment."""

    x: int
    y: int
    steps: int
