"""Test configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Shared fixtures can be added here
@pytest.fixture(scope="session")
def project_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def krolik_path(project_path):
    """Return the krolik package path."""
    return project_path / "krolik"


@pytest.fixture(scope="session")
def nanobot_path(project_path):
    """Return the nanobot package path."""
    return project_path / "nanobot"
