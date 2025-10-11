"""
Policy module for Clash Royale RL Agent.

This module contains policy implementations for the Clash Royale reinforcement learning agent.
Includes structured MLP policies with shared card encoders for efficient learning.
"""

from .interfaces import Policy, PolicyConfig

__all__ = [
    'Policy',
    'PolicyConfig'
]