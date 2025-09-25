"""
LUKi Core Agent Package

Primary LUKi brain: dialogue manager, tool orchestration, prompt packs & safety layers
"""

__version__ = "0.1.0"
__author__ = "ReMeLife Team"
__email__ = "dev@remelife.com"

# Only import what's actually used by the current system
try:
    from .context_builder import ContextBuilder
except ImportError:
    ContextBuilder = None

__all__ = ["ContextBuilder"]
