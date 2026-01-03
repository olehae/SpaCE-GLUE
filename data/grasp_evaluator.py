"""
GRASP Game Evaluator

Evaluates a sequence of steps in the GRASP grid world game.
Based on game rules:
- Grid world with cells containing energy (E) or obstacles (O)
- Agent (A) starts at a specific position
- Goal: collect energy and return it to starting cell
- 20 steps maximum
- Actions: UP, DOWN, LEFT, RIGHT, TAKE, DROP
- Can't move through obstacles or boundaries
- TAKE collects energy from current cell (if present)
- DROP puts all collected energy in current cell
"""

from typing import List, Dict, Tuple, Any
import re


class GRASPEvaluator:
    """Evaluator for GRASP grid world game."""

    def __init__(self, grid_string: str):
        self.grid = []
        self.grid_size = 0
        self.agent_pos = [0, 0]
        self.start_pos = [0, 0]
        self.agent_energy = 0
        self.max_steps = 20
        self.parse_grid(grid_string)

    def parse_grid(self, grid_string: str) -> None:
        """
        Parse the grid string and initialize game state.

        Args:
            grid_string: String representation of the grid
        """
        lines = grid_string.strip().split("\n")

        # Find the actual grid lines (skip borders)
        grid_lines = []
        for line in lines:
            # Grid lines contain cell markers like E, O, A, or spaces
            if "|" in line and len(line) > 2:
                grid_lines.append(line)

        self.grid_size = len(grid_lines)
        self.grid = [
            [{"energy": 0, "obstacle": False} for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        # Parse each cell
        for row_idx, line in enumerate(grid_lines):
            # Extract cell contents between '|' separators
            cells = [cell.strip() for cell in line.split("|")[1:-1]]

            for col_idx, cell in enumerate(cells):
                if "O" in cell:
                    self.grid[row_idx][col_idx]["obstacle"] = True
                if "E" in cell:
                    self.grid[row_idx][col_idx]["energy"] = 1
                if "A" in cell:
                    self.agent_pos = [row_idx, col_idx]
                    self.start_pos = [row_idx, col_idx]

    def move(self, direction: str) -> bool:
        """
        Move the agent in the specified direction.

        Args:
            direction: One of UP, DOWN, LEFT, RIGHT

        Returns:
            True if move was successful, False otherwise
        """
        row, col = self.agent_pos
        new_row, new_col = row, col

        if direction == "UP":
            new_row = row - 1
        elif direction == "DOWN":
            new_row = row + 1
        elif direction == "LEFT":
            new_col = col - 1
        elif direction == "RIGHT":
            new_col = col + 1
        else:
            return False

        # Check boundaries
        if (
            new_row < 0
            or new_row >= self.grid_size
            or new_col < 0
            or new_col >= self.grid_size
        ):
            return False

        # Check obstacles
        if self.grid[new_row][new_col]["obstacle"]:
            return False

        # Move is valid
        self.agent_pos = [new_row, new_col]
        return True

    def take(self) -> bool:
        """
        Take energy from current cell.

        Returns:
            True if energy was taken, False otherwise
        """
        row, col = self.agent_pos
        if self.grid[row][col]["energy"] > 0:
            self.grid[row][col]["energy"] -= 1
            self.agent_energy += 1
            return True
        return False

    def drop(self) -> bool:
        """
        Drop all energy in current cell.

        Returns:
            True if energy was dropped, False otherwise
        """
        if self.agent_energy > 0:
            row, col = self.agent_pos
            self.grid[row][col]["energy"] += self.agent_energy
            self.agent_energy = 0
            return True
        return False

    def execute_step(self, step: str) -> bool:
        """
        Execute a single step.

        Args:
            step: The step to execute (UP, DOWN, LEFT, RIGHT, TAKE, DROP)

        Returns:
            True if step was valid and executed, False otherwise
        """
        step = step.strip().upper()

        if step in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return self.move(step)
        elif step == "TAKE":
            return self.take()
        elif step == "DROP":
            return self.drop()
        else:
            return False

    def evaluate(self, steps: List[str]) -> Dict[str, Any]:
        """
        Evaluate a sequence of steps on the given grid.

        Args:
            steps: List of steps to execute

        Returns:
            Dictionary with evaluation results:
            - final_energy: Energy at starting position after execution
            - steps_taken: Number of steps executed
            - valid_steps: Number of valid steps executed
        """
        # Execute steps (max 20)
        steps_to_execute = steps[: self.max_steps]
        valid_step_count = 0

        for step in steps_to_execute:
            if self.execute_step(step):
                valid_step_count += 1
            else:
                # Invalid step, stop execution
                break

        # Calculate results
        start_row, start_col = self.start_pos
        final_energy = self.grid[start_row][start_col]["energy"]

        return {
            "final_energy": final_energy,
            "steps_taken": len(steps_to_execute),
            "valid_steps": valid_step_count,
        }


def parse_steps_from_response(response: str) -> List[str]:
    """
    Parse steps from a model response.

    Handles various formats:
    - [UP, DOWN, LEFT, RIGHT, TAKE]
    - ["UP", "DOWN", "LEFT"]
    - UP, DOWN, LEFT, RIGHT
    - UP DOWN LEFT RIGHT

    Args:
        response: The model's response string

    Returns:
        List of step strings
    """
    # Remove common wrappers
    response = response.strip()

    # Try to find content in brackets/braces
    bracket_match = re.search(r"[\[\{]([^\]\}]+)[\]\}]", response)
    if bracket_match:
        response = bracket_match.group(1)

    # Remove quotes
    response = response.replace('"', "").replace("'", "")

    # Split by comma or whitespace
    if "," in response:
        steps = [s.strip() for s in response.split(",")]
    else:
        steps = response.split()

    # return False if step not in valid actions
    valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "TAKE", "DROP"]
    for step in steps:
        step = step.strip().upper()
        if step.strip().upper() not in valid_actions:
            return []

    return steps
