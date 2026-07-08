"""
Top-level package initialization. Exposes main utilities for easy import.
"""
from .channel_colors import get_channel_colors
from .converters import BatchConverter, OmeZarrConverter

__version__ = "0.1.0"

__all__ = [
    "OmeZarrConverter",
    "BatchConverter",
    "get_channel_colors",
]
