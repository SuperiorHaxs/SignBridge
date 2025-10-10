"""
Configuration package for ASL-v1 project
Provides centralized path management for datasets, models, and outputs
"""

from .paths import PathConfig, get_config

__all__ = ['PathConfig', 'get_config']
