"""
Base action executor for Phase 0 of the Clash Royale RL Agent.

This module provides the BaseActionExecutor class that defines the interface
for action execution. The BootstrapActionExecutor in src/bootstrap/executor.py
extends this class to implement ADB-based action execution with humanization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import time


class BaseActionExecutor(ABC):
    """Base class for action executors in the Clash Royale RL Agent.

    This class defines the interface that all action executors must implement.
    It provides the basic structure for executing actions in the game
    environment.
    """

    @abstractmethod
    def execute(self, action: dict[str, Any]) -> bool:
        """Executes an action in the game environment.

        Args:
            action: A dictionary containing action parameters, including:
                - "card_slot": An integer (0-3) for card selection.
                - "grid_x": An integer (0-31) for the x-coordinate on the grid.
                - "grid_y": An integer (0-17) for the y-coordinate on the grid.

        Returns:
            True if the action was executed successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Connects to the game environment.

        Returns:
            True if the connection was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the game environment."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Checks if connected to the game environment.

        Returns:
            True if connected, False otherwise.
        """
        pass


class ActionResult:
    """Represents the result of an action execution.

    This data class encapsulates the outcome of an action, including its success
    status, timing information, and any relevant metadata.

    Attributes:
        success: Whether the action was executed successfully.
        execution_time_ms: The time taken to execute the action in milliseconds.
        error_message: An error message if the action failed.
        metadata: Additional metadata about the action execution.
        timestamp: The timestamp of when the action result was created.
    """

    def __init__(
        self,
        success: bool,
        execution_time_ms: float,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initializes an ActionResult.

        Args:
            success: Whether the action was executed successfully.
            execution_time_ms: The time taken to execute the action in
                milliseconds.
            error_message: An error message if the action failed.
            metadata: Additional metadata about the action execution.
        """
        self.success = success
        self.execution_time_ms = execution_time_ms
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """Returns a string representation of the ActionResult."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"ActionResult({status}, {self.execution_time_ms:.2f}ms)"
    
    def __repr__(self) -> str:
        """Returns a detailed string representation of the ActionResult."""
        return (f"ActionResult(success={self.success}, "
                f"execution_time_ms={self.execution_time_ms:.2f}, "
                f"error_message={self.error_message}, "
                f"metadata={self.metadata})")