"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Any

from krolik.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path (~/.krolik/config.json)."""
    krolik_cfg = Path.home() / ".krolik" / "config.json"
    legacy_cfg = Path.home() / ".nanobot" / "config.json"
    # Prefer .krolik; fall back to legacy if it exists
    if not krolik_cfg.exists() and legacy_cfg.exists():
        return legacy_cfg
    return krolik_cfg


def get_data_dir() -> Path:
    """Get the krolik data directory."""
    from krolik.utils.helpers import get_data_path
    return get_data_path()


def find_env_file() -> Path | None:
    """
    Find .env file in priority order:
    1. Current working directory
    2. Project root (where pyproject.toml exists)
    3. User home directory (~/.krolik/.env)
    4. User home directory (~/.nanobot/.env)
    
    Returns:
        Path to .env file or None if not found.
    """
    # Check current directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    
    # Check project root (look for pyproject.toml)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            project_env = parent / ".env"
            if project_env.exists():
                return project_env
            break
    
    # Check user config directories
    home_env = Path.home() / ".krolik" / ".env"
    if home_env.exists():
        return home_env
    
    legacy_env = Path.home() / ".nanobot" / ".env"
    if legacy_env.exists():
        return legacy_env
    
    return None


def load_env_file(env_path: Path | None = None) -> dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Optional path to .env file. Will auto-find if not provided.
    
    Returns:
        Dictionary of loaded environment variables.
    """
    path = env_path or find_env_file()
    
    if not path or not path.exists():
        return {}
    
    env_vars = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=value
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")  # Remove quotes
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        env_vars[key] = value
    except Exception as e:
        print(f"Warning: Failed to load .env from {path}: {e}")
    
    return env_vars


def _migrate_legacy_env_vars() -> None:
    """Copy NANOBOT_* env vars to KROLIK_* if not already set (backward compat)."""
    for key, value in list(os.environ.items()):
        if key.startswith("NANOBOT_") and not key.startswith("NANOBOT_TMUX_"):
            new_key = "KROLIK_" + key[len("NANOBOT_"):]
            if new_key not in os.environ:
                os.environ[new_key] = value


def _flatten_dict_to_env(data: dict, prefix: str = "KROLIK_") -> dict[str, str]:
    """
    Flatten nested dict to KROLIK__-style env vars.
    
    E.g. {"providers": {"openrouter": {"api_key": "X"}}}
    -> {"KROLIK_PROVIDERS__OPENROUTER__API_KEY": "X"}
    """
    result = {}
    
    def _flatten(obj: Any, path: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{path}{k.upper()}__" if path else f"{prefix}{k.upper()}__")
        elif isinstance(obj, list):
            # Skip lists for env var flattening
            pass
        elif obj is not None and str(obj):
            # Remove trailing __
            env_key = path.rstrip("_")
            result[env_key] = str(obj)
    
    _flatten(data, "")
    return result


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    """
    Load configuration from .env file and/or config.json.
    
    Priority (highest first):
    1. Real environment variables (set by shell)
    2. .env file variables
    3. config.json values (set as env defaults)
    4. Pydantic defaults
    
    Args:
        config_path: Optional path to config file. Uses default if not provided.
        env_path: Optional path to .env file. Auto-finds if not provided.
    
    Returns:
        Loaded configuration object.
    """
    # Step 0: Migrate legacy NANOBOT_* env vars to KROLIK_*
    _migrate_legacy_env_vars()
    
    # Step 1: Load config.json as lowest-priority defaults
    path = config_path or get_config_path()
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            # Flatten JSON to env vars (only set if not already present)
            flat = _flatten_dict_to_env(convert_keys(data))
            for key, value in flat.items():
                if key not in os.environ and value:
                    os.environ[key] = value
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
    
    # Step 2: Load .env file (higher priority than config.json)
    loaded_env = load_env_file(env_path)
    if loaded_env:
        print(f"Loaded {len(loaded_env)} variables from .env")
    
    # Step 3: Create Config via pydantic-settings (reads env vars automatically)
    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(item) for item in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(item) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
