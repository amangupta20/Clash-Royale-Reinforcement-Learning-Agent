"""
Environment module for Clash Royale RL Agent.

This module contains the environment implementations for the Clash Royale RL Agent.
It includes the basic Gymnasium environment wrapper that integrates all Phase 0 components.
"""

from .clash_royale_env import ClashRoyaleEnv

__all__ = ['ClashRoyaleEnv']