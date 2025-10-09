"""
Phase 0 Bootstrap Module

This module contains the minimal perception and execution components for Phase 0 of the
Clash Royale RL Agent project. Phase 0 focuses on fast MVP development using simplified
perception techniques before upgrading to full computer vision pipelines in Phase 1.

Components:
- capture: Screen capture wrapper for BlueStacks emulator
- template_matcher: Template matching for hand card detection
- minimal_perception: Pixel counting for elixir and OCR for tower health

Phase 0 Requirements:
- <50ms capture latency
- >80% card detection accuracy
- 100% elixir accuracy
- >95% tower health OCR accuracy
- 30-40% win rate vs easy AI
"""