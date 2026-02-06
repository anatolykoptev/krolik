"""Configuration module for krolik."""

from krolik.config.loader import load_config, get_config_path
from krolik.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
