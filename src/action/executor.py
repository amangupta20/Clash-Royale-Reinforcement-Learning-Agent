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
    """
    Base class for action executors in the Clash Royale RL Agent.
    
    This class defines the interface that all action executors must implement.
    It provides the basic structure for executing actions in the game environment.
    """
    
    @abstractmethod
    def execute(self, action: Dict[str, Any]) -> bool:
        """
        Execute an action in the game environment.
        
        Args:
            action: Dictionary containing action parameters:
                - card_slot: Integer (0-3) for card selection from 4 visible cards
                - grid_x: Integer (0-31) for x position on the grid
                - grid_y: Integer (0-17) for y position on the grid
                
        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the game environment.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the game environment.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to the game environment.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass


class ActionResult:
    """
    Data class representing the result of an action execution.
    
    This class encapsulates the outcome of an action execution, including
    success status, timing information, and any relevant metadata.
    """
    
    def __init__(self, 
                 success: bool, 
                 execution_time_ms: float,
                 error_message: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an ActionResult.
        
        Args:
            success: Whether the action was executed successfully
            execution_time_ms: Time taken to execute the action in milliseconds
            error_message: Error message if the action failed
            metadata: Additional metadata about the action execution
        """
        self.success = success
        self.execution_time_ms = execution_time_ms
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String representation of the ActionResult."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"ActionResult({status}, {self.execution_time_ms:.2f}ms)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the ActionResult."""
        return (f"ActionResult(success={self.success}, "
                f"execution_time_ms={self.execution_time_ms:.2f}, "
                f"error_message={self.error_message}, "
                f"metadata={self.metadata})")